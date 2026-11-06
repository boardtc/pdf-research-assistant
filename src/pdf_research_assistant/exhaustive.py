"""Exhaustive search: check every chunk of every in-scope paper against a question.

Scoping and planning read the saved index directly and make no API calls. Nothing
here calls PaperQA's get_directory_index: that creates missing index folders, and
with build=True a narrower manifest removes entries from the index.

The run calls map_fxn_summary per chunk rather than Docs.aget_evidence, which drops
failed chunks silently and loses the whole run on one uncaught error such as a rate
limit. Calling it directly keeps "failed" separate from "scored 0".
"""

import argparse
import asyncio
import csv
import html
import json
import logging
import pickle
import re
import sys
import time
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import limits
import litellm
from paperqa import Docs, Settings
from paperqa.core import llm_parse_json, map_fxn_summary

from pdf_research_assistant import bootstrap
from pdf_research_assistant.bootstrap import FAILED_DOCUMENT_ADD_ID

STATUS_LOADABLE = "loadable"
STATUS_FAILED = "failed_to_index"
STATUS_NOT_INDEXED = "not_indexed"
STATUS_UNREADABLE = "unreadable"

RESULT_RELEVANT = "relevant"
RESULT_NO_EVIDENCE = "no_relevant_evidence"
RESULT_POSSIBLE_MISSES = "possible_misses_only"
RESULT_INCOMPLETE = "incomplete"
RESULT_NOT_PROCESSED = "not_processed"

TIER_BEST = "best"
TIER_LIKELY = "likely"
TIER_POSSIBLE = "possible"

STATUS_LABELS = {
    RESULT_RELEVANT: "Relevant evidence found",
    RESULT_POSSIBLE_MISSES: "Possible misses only",
    RESULT_NO_EVIDENCE: "No relevant evidence",
    RESULT_INCOMPLETE: "Incomplete",
    RESULT_NOT_PROCESSED: "Not processed",
    STATUS_FAILED: "Failed to index",
    STATUS_NOT_INDEXED: "Not indexed",
    STATUS_UNREADABLE: "Unreadable",
}
TIER_LABELS = {TIER_BEST: "Best matches", TIER_LIKELY: "Likely matches", TIER_POSSIBLE: "Possibly related"}
PROBLEM_STATUSES = {
    RESULT_INCOMPLETE,
    RESULT_NOT_PROCESSED,
    STATUS_FAILED,
    STATUS_NOT_INDEXED,
    STATUS_UNREADABLE,
}
REPORT_NOTICE = (
    "This lists evidence, not a conclusion. It is more thorough than a normal query, but the model"
    " can still misjudge relevance, so \"no relevant evidence\" does not prove a paper has nothing."
    " Scanned pages and figures are not searched. Check anything important against the PDF."
)

EVIDENCE_SUMMARY_LENGTH = "about 200 words"
MAX_CONCURRENT_REQUESTS = 40
# Stays under the account's 2,000,000 tokens-a-minute limit, measured on 2026-11-06.
TOKENS_PER_MINUTE = 1_500_000
MAX_CONSECUTIVE_FAILED_CHUNKS = 20
ABANDONED_CHUNK_REASON = "PaperQA abandoned this chunk after a timeout, bad request, or unreadable reply."
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_*-]+")

# PaperQA's prompt rates the model's own summary, which scored most chunks 8 or higher.
# This one rates the excerpt against a defined scale, and asks for a quote that can be checked.
STRICT_SUMMARY_SYSTEM_PROMPT = (
    "You are screening an excerpt from a research paper for a literature review."
    " Judge only what the excerpt itself says, not what the works it cites might say."
    " Never stretch an excerpt to fit the question."
    " Respond with the following JSON format:"
    '\n\n{{\n  "quote": "...",\n  "summary": "...",\n  "relevance_score": 0-10\n}}'
    "\n\n`quote` is the single sentence from the excerpt, copied word for word, that best"
    " supports your score. Leave it empty if the score is 0."
    "\n\n`summary` states what the excerpt says that answers the question - {summary_length} words."
    " Leave it empty if the score is 3 or lower."
    "\n\n`relevance_score` is an integer for how well the excerpt itself answers the question:"
    "\n0: not about the question's topic. Reference lists, bibliographies, acknowledgements,"
    " and author details are always 0."
    "\n1-3: a passing mention only."
    "\n4-6: substantially about the topic, but no idea that answers the question."
    " If the excerpt covers only part of the question, the score is at most 6."
    "\n7-8: an idea that answers the question indirectly, including a barrier or problem"
    " that implies what should change."
    "\n9-10: explicitly proposes an idea that answers the question."
    "\n\nMost excerpts should score low. Use the whole scale."
)

# Words too common in these papers to show that a rejected chunk is on topic.
GENERIC_QUESTION_WORDS = {
    "about", "does", "from", "have", "idea", "ideas", "improve", "improved", "improving", "paper", "papers",
    "propose", "proposed", "research", "student", "students", "studies", "study", "their", "what", "when",
    "where", "which", "with",
}  # fmt: skip
KEY_TERM_LENGTH = 6
# Tuned on 200 scored chunks: flags the wrongly rejected ones without flagging most rejections.
POSSIBLE_MISS_MENTIONS = 5
# Citations per 100 words; above this a chunk is treated as a reference list.
REFERENCE_LIST_DENSITY = 3
CITATION_PATTERN = re.compile(r"\(\d{4}[a-z]?\)|\b(?:19|20)\d{2}[a-z]?[.,;)]|doi|https?://", re.IGNORECASE)
QUOTE_NOISE_PATTERN = re.compile(r"[\s\-­‐]+")

# Output length is unknown before a run, so the estimate is a range. Typical is measured.
OUTPUT_TOKENS_PER_CHUNK = {"low": 30, "typical": 160, "high": 300}
# Median measured for gpt-4o-mini with 200-word summaries.
SECONDS_PER_CALL = 5.5
CHARACTERS_PER_TOKEN = 4
# The rate limiter counts characters / 4, which runs about 10% above real input tokens.
LIMITER_TOKEN_FACTOR = 1.1

TokenCounter = Callable[[str], int]


def build_exhaustive_settings() -> Settings:
    """Build settings for scoring every chunk, leaving the index and normal query mode unchanged."""
    settings = bootstrap.build_settings()
    settings.summary_llm_config = {
        **settings.summary_llm_config,
        "rate_limit": {settings.summary_llm: f"{TOKENS_PER_MINUTE} per 1 minute"},
        # Without this, lmi defaults to temperature 1.0 and relevance scores vary between runs.
        "temperature": 0,
    }
    settings.answer.evidence_retrieval = False
    settings.answer.evidence_summary_length = EVIDENCE_SUMMARY_LENGTH
    settings.answer.max_concurrent_requests = MAX_CONCURRENT_REQUESTS
    settings.prompts.summary_json_system = STRICT_SUMMARY_SYSTEM_PROMPT
    return settings


@dataclass
class PaperScope:
    """One in-scope paper and whether its saved chunks can be used."""

    file_location: str
    status: str
    filehash: str | None = None
    chunk_count: int = 0
    chunk_tokens: int = 0
    citation_tokens: int = 0
    detail: str = ""


def read_index_files(index_dir: Path) -> tuple[dict[str, str], str | None]:
    """Return the index's file entries and an error message if they cannot be read."""
    file_index = index_dir / "files.zip"
    if not file_index.exists():
        return {}, None
    try:
        return pickle.loads(zlib.decompress(file_index.read_bytes())), None
    except Exception as exc:
        return {}, f"Could not read files.zip: {type(exc).__name__}: {exc}"


def resolve_in_scope_paths(index_files: dict[str, str]) -> list[str]:
    """Return every in-scope PDF: the manifest, or else all PDFs on disk or in the index."""
    allowed_paths = bootstrap.get_allowed_paths()
    if allowed_paths:
        return sorted(allowed_paths)
    paths = {bootstrap.normalize_file_location(file_location) for file_location in index_files}
    if bootstrap.PAPER_DIR.exists():
        paths.update(
            bootstrap.normalize_relative_pdf_path(path.relative_to(bootstrap.PAPER_DIR))
            for path in bootstrap.PAPER_DIR.rglob("*")
            if path.suffix.lower() in bootstrap.PDF_EXTENSIONS
        )
    return sorted(paths)


def load_saved_docs(index_dir: Path, filehash: str) -> Docs:
    """Load one paper's parsed chunks from the index, without touching the PDF."""
    saved_docs = index_dir / "docs" / f"{filehash}.zip"
    return pickle.loads(zlib.decompress(saved_docs.read_bytes()))


def inspect_saved_paper(
    index_dir: Path, file_location: str, filehash: str, count_tokens: TokenCounter
) -> PaperScope:
    """Check one indexed paper's saved chunks and count their tokens."""
    try:
        docs = load_saved_docs(index_dir, filehash)
    except Exception as exc:
        return PaperScope(
            file_location,
            STATUS_UNREADABLE,
            filehash=filehash,
            detail=f"Could not load saved chunks: {type(exc).__name__}: {exc}",
        )
    if not docs.texts:
        return PaperScope(file_location, STATUS_UNREADABLE, filehash=filehash, detail="Saved paper has no chunks.")
    return PaperScope(
        file_location,
        STATUS_LOADABLE,
        filehash=filehash,
        chunk_count=len(docs.texts),
        chunk_tokens=sum(count_tokens(text.text) for text in docs.texts),
        citation_tokens=sum(count_tokens(f"{text.name}: {text.doc.formatted_citation}") for text in docs.texts),
    )


def build_scope(index_dir: Path, count_tokens: TokenCounter) -> list[PaperScope]:
    """Return one entry per in-scope paper, including papers that cannot be searched."""
    index_files, error = read_index_files(index_dir)
    filehashes = {
        bootstrap.normalize_file_location(file_location): filehash for file_location, filehash in index_files.items()
    }
    scope = []
    for file_location in resolve_in_scope_paths(index_files):
        filehash = filehashes.get(file_location)
        if error:
            scope.append(PaperScope(file_location, STATUS_UNREADABLE, detail=error))
        elif filehash is None:
            scope.append(PaperScope(file_location, STATUS_NOT_INDEXED, detail="Not in the active index."))
        elif filehash == FAILED_DOCUMENT_ADD_ID:
            scope.append(PaperScope(file_location, STATUS_FAILED, detail="PaperQA could not parse this PDF."))
        else:
            scope.append(inspect_saved_paper(index_dir, file_location, filehash, count_tokens))
    return scope


def make_token_counter(model: str) -> TokenCounter:
    """Return a token counter for the model, or a character-based estimate if none is available."""
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        return lambda text: len(text) // CHARACTERS_PER_TOKEN
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def get_tokens_per_minute(settings: Settings) -> float | None:
    """Return the summary model's configured token rate limit, if it has one."""
    rate_limits = (settings.summary_llm_config or {}).get("rate_limit", {})
    rate_limit = rate_limits.get(settings.summary_llm)
    if rate_limit is None:
        return None
    parsed = limits.parse(rate_limit)
    return parsed.amount * 60 / parsed.get_expiry()


def estimate_run(
    scope: list[PaperScope], question: str, settings: Settings, count_tokens: TokenCounter
) -> dict:
    """Estimate tokens, cost, and runtime for one summary call per loadable chunk."""
    papers = [paper for paper in scope if paper.status == STATUS_LOADABLE]
    chunk_count = sum(paper.chunk_count for paper in papers)
    system_prompt = settings.prompts.summary_json_system.format(
        summary_length=settings.answer.evidence_summary_length
    )
    user_prompt = settings.prompts.summary_json.format(citation="", text="", question=question)
    tokens_per_call = count_tokens(system_prompt) + count_tokens(user_prompt)
    input_tokens = sum(paper.chunk_tokens + paper.citation_tokens for paper in papers) + chunk_count * tokens_per_call

    prices = litellm.model_cost.get(settings.summary_llm)
    cost_usd = None
    if prices:
        cost_usd = {
            label: input_tokens * prices["input_cost_per_token"]
            + chunk_count * output_tokens * prices["output_cost_per_token"]
            for label, output_tokens in OUTPUT_TOKENS_PER_CHUNK.items()
        }

    # A run is held back by whichever is slower: call speed or the token rate limit.
    estimated_minutes = chunk_count * SECONDS_PER_CALL / settings.answer.max_concurrent_requests / 60
    tokens_per_minute = get_tokens_per_minute(settings)
    if tokens_per_minute:
        limiter_tokens = input_tokens * LIMITER_TOKEN_FACTOR + chunk_count * OUTPUT_TOKENS_PER_CHUNK["typical"]
        estimated_minutes = max(estimated_minutes, limiter_tokens / tokens_per_minute)

    return {
        "chunk_count": chunk_count,
        "input_tokens": input_tokens,
        "cost_usd": cost_usd,
        "estimated_minutes": estimated_minutes,
    }


def build_plan(question: str, settings: Settings | None = None, index_dir: Path | None = None) -> dict:
    """Build the pre-run summary: every in-scope paper plus the estimated run size."""
    current_settings = settings or build_exhaustive_settings()
    active_index_dir = index_dir or bootstrap.get_active_index_dir(current_settings)
    count_tokens = make_token_counter(current_settings.summary_llm)
    scope = build_scope(active_index_dir, count_tokens)
    return {
        "question": question,
        "model": current_settings.summary_llm,
        "paper_count": len(scope),
        "status_counts": dict(Counter(paper.status for paper in scope)),
        **estimate_run(scope, question, current_settings, count_tokens),
        "papers": [asdict(paper) for paper in scope],
    }


def extract_pages(chunk_name: str) -> str:
    """Return the page range from a chunk name like 'Smith2020 pages 3-4'."""
    _, separator, pages = chunk_name.rpartition(" pages ")
    return pages if separator else ""


def redact_secrets(message: str) -> str:
    """Remove anything shaped like an API key, because failure reasons are saved to disk."""
    return SECRET_PATTERN.sub("[redacted]", message)


def extract_key_terms(question: str) -> set[str]:
    """Return shortened topic words from a question, so 'engaging' matches 'engagement'."""
    words = re.findall(r"[a-z]{4,}", question.lower())
    return {word[:KEY_TERM_LENGTH] for word in words if word not in GENERIC_QUESTION_WORDS}


def citation_density(text: str) -> float:
    """Return citations per 100 words, which is high in reference lists."""
    word_count = len(text.split())
    return len(CITATION_PATTERN.findall(text)) / word_count * 100 if word_count else 0


def is_possible_miss(text: str, key_terms: set[str]) -> bool:
    """Return whether a rejected chunk keeps mentioning the topic and is not a reference list."""
    lowered = text.lower()
    repeats_topic = any(lowered.count(term) >= POSSIBLE_MISS_MENTIONS for term in key_terms)
    return repeats_topic and citation_density(text) < REFERENCE_LIST_DENSITY


def verify_quote(quote: str, text: str) -> bool:
    """Return whether a quote is in the chunk, ignoring PDF line breaks and hyphenation."""
    cleaned_quote = QUOTE_NOISE_PATTERN.sub("", quote).lower()
    return bool(cleaned_quote) and cleaned_quote in QUOTE_NOISE_PATTERN.sub("", text).lower()


def tier_for_score(score: int) -> str:
    """Group a score for the report; every tier is kept, so nothing is hidden."""
    if score >= 9:
        return TIER_BEST
    return TIER_LIKELY if score >= 7 else TIER_POSSIBLE


def classify_paper(evidence_count: int, failed_count: int, possible_miss_count: int = 0) -> str:
    """Report 'no relevant evidence' only when every chunk was checked and none looks on topic."""
    if evidence_count:
        return RESULT_RELEVANT
    if failed_count:
        return RESULT_INCOMPLETE
    return RESULT_POSSIBLE_MISSES if possible_miss_count else RESULT_NO_EVIDENCE


def failed_chunk(text: object, reason: str, cost: float) -> dict:
    """Describe a chunk that could not be checked."""
    return {
        "chunk": text.name,
        "pages": extract_pages(text.name),
        "failed": True,
        "reason": redact_secrets(reason),
        "cost_usd": cost,
    }


@dataclass
class RunState:
    """Failure streak shared by all chunks, so a run can stop while papers are in flight."""

    streak: int = 0
    stop_reason: str | None = None

    def record(self, failed: bool) -> None:
        """Count consecutive failed chunks across papers; any success resets the count."""
        self.streak = self.streak + 1 if failed else 0
        if self.streak >= MAX_CONSECUTIVE_FAILED_CHUNKS and not self.stop_reason:
            self.stop_reason = f"Stopped early after {self.streak} failed requests in a row."


async def score_chunk(text: object, question: str, settings: Settings, summary_llm_model: object) -> dict:
    """Score one chunk with PaperQA's evidence prompts, turning any failure into a result."""
    try:
        context, llm_results = await map_fxn_summary(
            text=text,
            question=question,
            summary_llm_model=summary_llm_model,
            prompt_templates=(settings.prompts.summary_json, settings.prompts.summary_json_system),
            extra_prompt_data={
                "summary_length": settings.answer.evidence_summary_length,
                "citation": f"{text.name}: {text.doc.formatted_citation}",
            },
            parser=llm_parse_json,
        )
    except Exception as exc:
        return failed_chunk(text, f"{type(exc).__name__}: {exc}", 0.0)
    cost = sum(result.cost for result in llm_results)
    if context is None:
        return failed_chunk(text, ABANDONED_CHUNK_REASON, cost)
    quote = str((context.model_extra or {}).get("quote") or "")
    return {
        "chunk": text.name,
        "pages": extract_pages(text.name),
        "failed": False,
        "score": context.score,
        "tier": tier_for_score(context.score),
        "summary": context.context,
        "quote": quote,
        "quote_verified": verify_quote(quote, text.text),
        "possible_miss": context.score == 0 and is_possible_miss(text.text, extract_key_terms(question)),
        "raw_text": text.text,
        "cost_usd": cost,
    }


async def summarize_chunk(
    text: object,
    question: str,
    settings: Settings,
    summary_llm_model: object,
    semaphore: asyncio.Semaphore,
    state: RunState,
) -> dict:
    """Score one chunk within the request limit, or skip it if the run has stopped."""
    async with semaphore:
        if state.stop_reason:
            return {**failed_chunk(text, state.stop_reason, 0.0), "skipped": True}
        chunk = await score_chunk(text, question, settings, summary_llm_model)
        state.record(chunk["failed"])
    return chunk


def unsearched_paper(file_location: str, status: str, detail: str) -> dict:
    """Build the result for a paper that was not searched, so it still appears in the report."""
    return {
        "file_location": file_location,
        "status": status,
        "detail": detail,
        "chunks_total": 0,
        "chunks_failed": 0,
        "cost_usd": 0.0,
        "evidence": [],
        "possible_misses": [],
        "failures": [],
    }


async def search_paper(
    paper: PaperScope,
    question: str,
    settings: Settings,
    summary_llm_model: object,
    semaphore: asyncio.Semaphore,
    index_dir: Path,
    state: RunState,
) -> dict:
    """Check every chunk of one paper and return its result."""
    try:
        docs = load_saved_docs(index_dir, paper.filehash)
    except Exception as exc:
        detail = f"Could not load saved chunks: {type(exc).__name__}: {exc}"
        return unsearched_paper(paper.file_location, STATUS_UNREADABLE, detail)
    chunks = await asyncio.gather(
        *(summarize_chunk(text, question, settings, summary_llm_model, semaphore, state) for text in docs.texts)
    )
    if all(chunk.get("skipped") for chunk in chunks):
        return unsearched_paper(paper.file_location, RESULT_NOT_PROCESSED, state.stop_reason)
    failures = [chunk for chunk in chunks if chunk["failed"]]
    evidence = sorted(
        (chunk for chunk in chunks if not chunk["failed"] and chunk["score"] > 0),
        key=lambda chunk: -chunk["score"],
    )
    possible_misses = [chunk for chunk in chunks if chunk.get("possible_miss")]
    return {
        "file_location": paper.file_location,
        "status": classify_paper(len(evidence), len(failures), len(possible_misses)),
        "detail": "",
        "chunks_total": len(chunks),
        "chunks_failed": len(failures),
        "cost_usd": sum(chunk["cost_usd"] for chunk in chunks),
        "evidence": evidence,
        "possible_misses": possible_misses,
        "failures": failures,
    }


def load_manifest_titles() -> dict[str, str]:
    """Return manifest titles by relative path, which read better than file names."""
    if not bootstrap.manifest_exists():
        return {}
    with open(bootstrap.MANIFEST_PATH, newline="", encoding="utf-8-sig") as handle:
        return {
            bootstrap.normalize_relative_pdf_path(row["file_location"]): row["title"]
            for row in csv.DictReader(handle)
            if row.get("file_location") and row.get("title")
        }


def paper_title(file_location: str, titles: dict[str, str]) -> str:
    """Return a paper's manifest title, or its file name if it has none."""
    return one_line(titles.get(file_location) or Path(file_location).stem)


def one_line(text: str) -> str:
    """Make text safe for one Markdown line; it comes from PDFs and the model, not from us."""
    return html.escape(" ".join(text.split()), quote=False)


def pages_label(item: dict) -> str:
    """Describe where an item came from."""
    return f"Pages {item['pages']}" if item["pages"] else "Page unknown"


def format_evidence(item: dict) -> list[str]:
    """Render one piece of evidence with its pages, score, and quote."""
    lines = [f"- **{pages_label(item)} (score {item['score']})**: {one_line(item['summary']) or 'No summary given.'}"]
    if item["quote"]:
        status = "verified" if item["quote_verified"] else "not verified, check the PDF"
        lines.append(f'  - Quote ({status}): "{one_line(item["quote"])}"')
    return lines


def format_collapsed(heading: str, lines: list[str]) -> list[str]:
    """Wrap lower-priority items in a section the reader can open."""
    return ["<details>", f"<summary>{heading}</summary>", "", *lines, "", "</details>", ""]


def format_failures(paper: dict) -> list[str]:
    """Render the chunks of a paper that could not be checked."""
    if not paper["failures"]:
        return []
    lines = [f"**{paper['chunks_failed']} of {paper['chunks_total']} chunks could not be checked:**", ""]
    lines.extend(f"- {pages_label(item)}: {one_line(item['reason'])}" for item in paper["failures"])
    return [*lines, ""]


def format_possible_misses(paper: dict) -> list[str]:
    """Render rejected chunks that keep mentioning the topic."""
    if not paper["possible_misses"]:
        return []
    lines = [f"- {pages_label(item)}: scored 0, but it keeps mentioning the topic." for item in paper["possible_misses"]]
    return format_collapsed(f"Possible misses ({len(lines)})", lines)


def format_paper(paper: dict, titles: dict[str, str]) -> list[str]:
    """Render one paper's evidence, strongest tier first."""
    lines = [
        f"### {paper_title(paper['file_location'], titles)}",
        "",
        f"`{paper['file_location']}`, {paper['chunks_total']} chunks checked",
        "",
    ]
    lines.extend(format_failures(paper))
    for tier in (TIER_BEST, TIER_LIKELY):
        items = [item for item in paper["evidence"] if item["tier"] == tier]
        if items:
            lines.extend([f"**{TIER_LABELS[tier]}**", ""])
            lines.extend(line for item in items for line in format_evidence(item))
            lines.append("")
    possible = [item for item in paper["evidence"] if item["tier"] == TIER_POSSIBLE]
    if possible:
        heading = f"{TIER_LABELS[TIER_POSSIBLE]} ({len(possible)})"
        lines.extend(format_collapsed(heading, [line for item in possible for line in format_evidence(item)]))
    lines.extend(format_possible_misses(paper))
    return lines


def format_problem_paper(paper: dict, titles: dict[str, str]) -> list[str]:
    """Render a paper that was not fully searched, and why."""
    label = STATUS_LABELS[paper["status"]]
    title = paper_title(paper["file_location"], titles)
    detail = f" {one_line(paper['detail'])}" if paper["detail"] else ""
    return [f"- **{title}** (`{paper['file_location']}`): {label}.{detail}", *format_failures(paper)]


def format_warnings(summary: dict) -> list[str]:
    """Render anything that means the results are not complete."""
    lines = []
    if summary["stopped_early"]:
        lines.append(f"**This run did not finish.** {summary['stop_reason']}")
    if summary["failed_chunks"]:
        lines.append(f"**{summary['failed_chunks']} chunks could not be checked.** See the papers listed below.")
    return [line for warning in lines for line in (warning, "")]


def format_totals(summary: dict) -> list[str]:
    """Render the run's totals."""
    cost_note = " The real figure may be slightly higher, because failed requests can be billed." if (
        summary["cost_may_be_understated"]
    ) else ""
    lines = [
        "## Summary",
        "",
        f"- Papers in scope: {summary['paper_count']}",
        *(f"- {label}: {summary['status_counts'][status]}" for status, label in STATUS_LABELS.items() if status in summary["status_counts"]),
        *(f"- {label}: {summary['tier_counts'][tier]}" for tier, label in TIER_LABELS.items() if tier in summary["tier_counts"]),
        f"- Possible misses: {summary['possible_miss_count']}",
        f"- Cost: ${summary['cost_usd']:.4f}.{cost_note}",
        f"- Runtime: {summary['runtime_seconds'] / 60:.1f} minutes",
        "",
    ]
    return lines


def render_report(summary: dict, titles: dict[str, str]) -> str:
    """Render the run as Markdown, with every in-scope paper listed exactly once."""
    papers = summary["papers"]

    def best_first(paper: dict) -> tuple:
        tiers = Counter(item["tier"] for item in paper["evidence"])
        return -tiers[TIER_BEST], -tiers[TIER_LIKELY], -tiers[TIER_POSSIBLE], paper["file_location"].lower()

    lines = [
        "# Exhaustive search report",
        "",
        f"**Question:** {one_line(summary['question'])}",
        "",
        f"Scored by `{summary['model']}`. {REPORT_NOTICE}",
        "",
        *format_warnings(summary),
        *format_totals(summary),
    ]
    problems = [paper for paper in papers if paper["status"] in PROBLEM_STATUSES]
    if problems:
        lines.extend(["## Papers not fully searched", ""])
        lines.extend(line for paper in problems for line in format_problem_paper(paper, titles))
        lines.append("")
    relevant = sorted((paper for paper in papers if paper["status"] == RESULT_RELEVANT), key=best_first)
    if relevant:
        lines.extend(["## Papers with evidence", ""])
        lines.extend(line for paper in relevant for line in format_paper(paper, titles))
    possible = [paper for paper in papers if paper["status"] == RESULT_POSSIBLE_MISSES]
    if possible:
        lines.extend(["## Papers with possible misses only", ""])
        lines.extend(line for paper in possible for line in format_paper(paper, titles))
    empty = [paper for paper in papers if paper["status"] == RESULT_NO_EVIDENCE]
    if empty:
        lines.extend(["## Papers with no relevant evidence", ""])
        lines.extend(
            f"- {paper_title(paper['file_location'], titles)} (`{paper['file_location']}`),"
            f" {paper['chunks_total']} chunks checked"
            for paper in empty
        )
        lines.append("")
    return "\n".join(lines)


async def run_exhaustive(
    question: str,
    output_dir: Path,
    settings: Settings | None = None,
    index_dir: Path | None = None,
    summary_llm_model: object | None = None,
) -> dict:
    """Search every in-scope paper side by side, saving each one as it finishes.

    This makes paid API calls, so callers must get confirmation first.
    """
    current_settings = settings or build_exhaustive_settings()
    active_index_dir = index_dir or bootstrap.get_active_index_dir(current_settings)
    model = summary_llm_model or current_settings.get_summary_llm()
    semaphore = asyncio.Semaphore(current_settings.answer.max_concurrent_requests)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = RunState()

    async def process_paper(paper: PaperScope) -> dict:
        if paper.status == STATUS_LOADABLE:
            result = await search_paper(
                paper, question, current_settings, model, semaphore, active_index_dir, state
            )
        else:
            result = unsearched_paper(paper.file_location, paper.status, paper.detail)
        with open(output_dir / "results.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    started = time.monotonic()
    # Papers run together so short papers still fill the request limit; the semaphore caps calls.
    papers = await asyncio.gather(
        *(process_paper(paper) for paper in build_scope(active_index_dir, lambda _text: 0))
    )

    failed_chunks = sum(paper["chunks_failed"] for paper in papers)
    summary = {
        "question": question,
        "model": current_settings.summary_llm,
        "paper_count": len(papers),
        "status_counts": dict(Counter(paper["status"] for paper in papers)),
        "tier_counts": dict(Counter(item["tier"] for paper in papers for item in paper["evidence"])),
        "possible_miss_count": sum(len(paper["possible_misses"]) for paper in papers),
        "failed_chunks": failed_chunks,
        "cost_usd": sum(paper["cost_usd"] for paper in papers),
        # A failed request can still be billed without its cost reaching us.
        "cost_may_be_understated": failed_chunks > 0,
        "runtime_seconds": time.monotonic() - started,
        "stopped_early": state.stop_reason is not None,
        "stop_reason": state.stop_reason,
        "papers": papers,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    report = render_report(summary, load_manifest_titles())
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the helper contract used by the app and CLI: plan is free, run needs --confirmed."""
    parser = argparse.ArgumentParser(description="Internal exhaustive search helper.")
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("question")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirmed", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # PaperQA logs every abandoned chunk; failures are reported in the results instead.
    logging.disable(logging.CRITICAL)
    try:
        if args.command == "plan":
            payload = {"ok": True, "plan": build_plan(args.question)}
        elif not (args.confirmed and args.output_dir):
            payload = {"ok": False, "error": "A run needs --confirmed and --output-dir."}
        else:
            asyncio.run(run_exhaustive(args.question, args.output_dir))
            payload = {"ok": True}
    except Exception as exc:
        payload = {"ok": False, "error_type": type(exc).__name__, "error": redact_secrets(str(exc))}
    # Plain ASCII, because a Windows pipe can fail on characters from paper titles.
    print(json.dumps(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
