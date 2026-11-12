import asyncio
import importlib
import json
import os
import pickle
import shutil
import sys
import uuid
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import litellm
import pytest
from lmi import LLMResult
from paperqa import Docs
from paperqa.types import Doc, Text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARD_NAME = "pqa_index_test"
RELEVANT_REPLY = '{"summary": "a relevant idea", "relevance_score": 7}'
IRRELEVANT_REPLY = '{"summary": "", "relevance_score": 0}'
BAD_JSON_REPLY = '{"summary": 12 "relevance_score" nonsense'
NULL_SCORE_REPLY = '{"summary": "x", "relevance_score": null}'


class FakeSummaryLLM:
    """Stands in for the summary model; replies are keyed by chunk name, so no API call is possible."""

    def __init__(self, replies=None, default=RELEVANT_REPLY, cost=0.01):
        self.replies = replies or {}
        self.default = default
        self.cost = cost
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def call_single(self, messages, callbacks=None, name=None, **_):
        chunk_name = name.removeprefix("evidence:")
        self.calls.append(chunk_name)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        reply = self.replies.get(chunk_name, self.default)
        if isinstance(reply, BaseException):
            raise reply
        return LLMResult(model="fake", date="", text=reply, cost=self.cost)


def _litellm_error(error_type):
    return error_type("failed", model="m", llm_provider="openai")


def _make_docs(name: str, chunks: list[str]) -> Docs:
    docs = Docs()
    doc = Doc(docname=name, citation=f"{name} citation", dockey=name, content_hash=name)
    docs.docs[name] = doc
    docs.docnames.add(name)
    docs.texts = [Text(text=chunk, name=f"{name} pages {i}", doc=doc) for i, chunk in enumerate(chunks, start=1)]
    return docs


def _write_files_zip(shard_dir: Path, index_files: dict[str, str]) -> None:
    (shard_dir / "docs").mkdir(parents=True, exist_ok=True)
    (shard_dir / "files.zip").write_bytes(zlib.compress(pickle.dumps(index_files)))


def _write_saved_docs(shard_dir: Path, filehash: str, docs: Docs) -> None:
    (shard_dir / "docs").mkdir(parents=True, exist_ok=True)
    (shard_dir / "docs" / f"{filehash}.zip").write_bytes(zlib.compress(pickle.dumps(docs)))


def _write_manifest(path: Path, locations: list[str]) -> None:
    path.write_text("file_location\n" + "\n".join(locations) + "\n", encoding="utf-8")


def _count_chars(text: str) -> int:
    return len(text)


def _paper(module, **overrides):
    values = {"file_location": "a.pdf", "status": module.STATUS_LOADABLE, "chunk_count": 2, "chunk_tokens": 100, "citation_tokens": 10}
    return module.PaperScope(**(values | overrides))


def _settings(rate_limit: dict | None = None, model: str = "gpt-4o-mini"):
    return SimpleNamespace(
        summary_llm=model,
        summary_llm_config={"rate_limit": rate_limit} if rate_limit is not None else None,
        answer=SimpleNamespace(evidence_summary_length="about 100 words", max_concurrent_requests=4),
        prompts=SimpleNamespace(
            summary_json_system="system {summary_length}",
            summary_json="{citation}|{text}|{question}",
        ),
    )


@pytest.fixture
def workspace_tmp_path():
    # Create a unique per-test directory under tests/.tmp/ so filesystem tests stay isolated.
    root = PROJECT_ROOT / "tests" / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(uuid.uuid4())
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def exhaustive_module(workspace_tmp_path):
    paper_dir = workspace_tmp_path / "papers"
    paper_dir.mkdir()
    env = {
        "PAPER_DIR": str(paper_dir),
        "INDEX_DIR": str(workspace_tmp_path / "index"),
        "MANIFEST_PATH": str(workspace_tmp_path / "manifest.csv"),
    }
    module_names = ("pdf_research_assistant.exhaustive", "pdf_research_assistant.bootstrap")

    def forget_modules():
        # Drop the package attribute too, or `from pdf_research_assistant import bootstrap` reuses it.
        package = importlib.import_module("pdf_research_assistant")
        for name in module_names:
            sys.modules.pop(name, None)
            vars(package).pop(name.rsplit(".", maxsplit=1)[-1], None)

    with mock.patch.dict(os.environ, env, clear=False), mock.patch("dotenv.load_dotenv"):
        forget_modules()
        module = importlib.import_module("pdf_research_assistant.exhaustive")
        yield module
    forget_modules()


@pytest.fixture
def shard_dir(workspace_tmp_path):
    return workspace_tmp_path / "index" / SHARD_NAME


@pytest.fixture
def manifest_path(workspace_tmp_path):
    return workspace_tmp_path / "manifest.csv"


def test_import_exposes_scope_api_without_touching_the_index(exhaustive_module, workspace_tmp_path):
    assert callable(exhaustive_module.build_plan)
    assert not (workspace_tmp_path / "index").exists()


def test_read_index_files_returns_entries_when_archive_is_valid(exhaustive_module, shard_dir):
    _write_files_zip(shard_dir, {"a.pdf": "hash-a"})

    assert exhaustive_module.read_index_files(shard_dir) == ({"a.pdf": "hash-a"}, None)


def test_read_index_files_reports_no_error_when_archive_is_missing(exhaustive_module, shard_dir):
    assert exhaustive_module.read_index_files(shard_dir) == ({}, None)


def test_read_index_files_reports_error_when_archive_is_corrupt(exhaustive_module, shard_dir):
    shard_dir.mkdir(parents=True)
    (shard_dir / "files.zip").write_bytes(b"not a zip")

    index_files, error = exhaustive_module.read_index_files(shard_dir)

    assert index_files == {}
    assert "files.zip" in error


def test_resolve_in_scope_paths_uses_only_manifest_paths_when_manifest_is_present(exhaustive_module, manifest_path):
    _write_manifest(manifest_path, ["b/two.pdf", "a/one.pdf"])

    paths = exhaustive_module.resolve_in_scope_paths({"other.pdf": "hash"})

    assert paths == ["a/one.pdf", "b/two.pdf"]


def test_resolve_in_scope_paths_combines_disk_and_index_when_no_manifest_exists(exhaustive_module, workspace_tmp_path):
    papers = workspace_tmp_path / "papers"
    (papers / "sub").mkdir()
    (papers / "sub" / "disk.pdf").write_bytes(b"x")
    (papers / "notes.txt").write_bytes(b"x")

    paths = exhaustive_module.resolve_in_scope_paths({"indexed\\only.pdf": "hash"})

    assert paths == ["indexed/only.pdf", "sub/disk.pdf"]


def test_resolve_in_scope_paths_uses_index_alone_when_paper_dir_is_missing(exhaustive_module, workspace_tmp_path):
    shutil.rmtree(workspace_tmp_path / "papers")

    assert exhaustive_module.resolve_in_scope_paths({"indexed.pdf": "hash"}) == ["indexed.pdf"]


def test_load_saved_docs_returns_stored_chunks(exhaustive_module, shard_dir):
    _write_saved_docs(shard_dir, "hash-a", _make_docs("a", ["one", "two"]))

    docs = exhaustive_module.load_saved_docs(shard_dir, "hash-a")

    assert [text.text for text in docs.texts] == ["one", "two"]


def test_build_scope_reports_every_manifest_paper_with_its_status(exhaustive_module, shard_dir, manifest_path):
    locations = ["good.pdf", "failed.pdf", "absent.pdf", "no-pickle.pdf", "corrupt.pdf", "empty.pdf"]
    _write_manifest(manifest_path, locations)
    _write_files_zip(
        shard_dir,
        {
            "good.pdf": "hash-good",
            "failed.pdf": exhaustive_module.FAILED_DOCUMENT_ADD_ID,
            "no-pickle.pdf": "hash-missing",
            "corrupt.pdf": "hash-corrupt",
            "empty.pdf": "hash-empty",
            "out-of-scope.pdf": "hash-other",
        },
    )
    _write_saved_docs(shard_dir, "hash-good", _make_docs("good", ["alpha", "beta"]))
    _write_saved_docs(shard_dir, "hash-empty", _make_docs("empty", []))
    (shard_dir / "docs" / "hash-corrupt.zip").write_bytes(b"not a zip")

    scope = exhaustive_module.build_scope(shard_dir, _count_chars)

    statuses = {paper.file_location: paper.status for paper in scope}
    assert sorted(statuses) == sorted(locations)
    assert statuses["good.pdf"] == exhaustive_module.STATUS_LOADABLE
    assert statuses["failed.pdf"] == exhaustive_module.STATUS_FAILED
    assert statuses["absent.pdf"] == exhaustive_module.STATUS_NOT_INDEXED
    assert statuses["no-pickle.pdf"] == exhaustive_module.STATUS_UNREADABLE
    assert statuses["corrupt.pdf"] == exhaustive_module.STATUS_UNREADABLE
    assert statuses["empty.pdf"] == exhaustive_module.STATUS_UNREADABLE
    assert all(paper.detail for paper in scope if paper.status != exhaustive_module.STATUS_LOADABLE)


def test_build_scope_counts_chunks_and_tokens_for_loadable_papers(exhaustive_module, shard_dir, manifest_path):
    _write_manifest(manifest_path, ["good.pdf"])
    _write_files_zip(shard_dir, {"good.pdf": "hash-good"})
    _write_saved_docs(shard_dir, "hash-good", _make_docs("good", ["alpha", "beta"]))

    (paper,) = exhaustive_module.build_scope(shard_dir, _count_chars)

    expected_citations = len("good pages 1: good citation") + len("good pages 2: good citation")
    assert (paper.chunk_count, paper.chunk_tokens, paper.citation_tokens) == (2, 9, expected_citations)
    assert paper.filehash == "hash-good"


def test_build_scope_matches_windows_index_keys_to_forward_slash_manifest_paths(exhaustive_module, shard_dir, manifest_path):
    _write_manifest(manifest_path, ["sub/good.pdf"])
    _write_files_zip(shard_dir, {"sub\\good.pdf": "hash-good"})
    _write_saved_docs(shard_dir, "hash-good", _make_docs("good", ["alpha"]))

    (paper,) = exhaustive_module.build_scope(shard_dir, _count_chars)

    assert paper.status == exhaustive_module.STATUS_LOADABLE


def test_build_scope_keeps_both_files_when_their_content_is_identical(exhaustive_module, shard_dir, manifest_path):
    _write_manifest(manifest_path, ["copy-1.pdf", "copy-2.pdf"])
    _write_files_zip(shard_dir, {"copy-1.pdf": "same-hash", "copy-2.pdf": "same-hash"})
    _write_saved_docs(shard_dir, "same-hash", _make_docs("same", ["alpha"]))

    scope = exhaustive_module.build_scope(shard_dir, _count_chars)

    assert [paper.status for paper in scope] == [exhaustive_module.STATUS_LOADABLE] * 2


def test_build_scope_marks_all_papers_not_indexed_when_archive_is_missing(exhaustive_module, shard_dir, manifest_path):
    _write_manifest(manifest_path, ["a.pdf", "b.pdf"])

    scope = exhaustive_module.build_scope(shard_dir, _count_chars)

    assert [paper.status for paper in scope] == [exhaustive_module.STATUS_NOT_INDEXED] * 2


def test_build_scope_marks_all_papers_unreadable_when_archive_is_corrupt(exhaustive_module, shard_dir, manifest_path):
    _write_manifest(manifest_path, ["a.pdf", "b.pdf"])
    shard_dir.mkdir(parents=True)
    (shard_dir / "files.zip").write_bytes(b"not a zip")

    scope = exhaustive_module.build_scope(shard_dir, _count_chars)

    assert [paper.status for paper in scope] == [exhaustive_module.STATUS_UNREADABLE] * 2
    assert all("files.zip" in paper.detail for paper in scope)


def test_make_token_counter_uses_the_model_encoding(exhaustive_module):
    tiktoken_module = ModuleType("tiktoken")
    encoding = mock.Mock()
    encoding.encode.return_value = [1, 2, 3]
    tiktoken_module.encoding_for_model = mock.Mock(return_value=encoding)

    with mock.patch.dict(sys.modules, {"tiktoken": tiktoken_module}):
        count_tokens = exhaustive_module.make_token_counter("gpt-4o-mini")

    assert count_tokens("<|endoftext|>") == 3
    tiktoken_module.encoding_for_model.assert_called_once_with("gpt-4o-mini")
    encoding.encode.assert_called_once_with("<|endoftext|>", disallowed_special=())


def test_make_token_counter_falls_back_to_characters_when_encoding_is_unavailable(exhaustive_module):
    tiktoken_module = ModuleType("tiktoken")
    tiktoken_module.encoding_for_model = mock.Mock(side_effect=KeyError("unknown model"))

    with mock.patch.dict(sys.modules, {"tiktoken": tiktoken_module}):
        count_tokens = exhaustive_module.make_token_counter("unknown-model")

    assert count_tokens("x" * 40) == 10


def test_get_tokens_per_minute_parses_the_configured_limit(exhaustive_module):
    settings = _settings(rate_limit={"gpt-4o-mini": "30000 per 1 minute"})

    assert exhaustive_module.get_tokens_per_minute(settings) == 30000


def test_get_tokens_per_minute_converts_other_periods_to_minutes(exhaustive_module):
    settings = _settings(rate_limit={"gpt-4o-mini": "100 per 1 second"})

    assert exhaustive_module.get_tokens_per_minute(settings) == 6000


def test_get_tokens_per_minute_returns_none_when_no_config_exists(exhaustive_module):
    assert exhaustive_module.get_tokens_per_minute(_settings()) is None


def test_get_tokens_per_minute_returns_none_when_model_has_no_limit(exhaustive_module):
    assert exhaustive_module.get_tokens_per_minute(_settings(rate_limit={"other": "1 per 1 minute"})) is None


def test_estimate_run_counts_only_loadable_papers(exhaustive_module):
    scope = [
        _paper(exhaustive_module),
        _paper(exhaustive_module, status=exhaustive_module.STATUS_FAILED, chunk_count=0, chunk_tokens=0, citation_tokens=0),
    ]

    estimate = exhaustive_module.estimate_run(scope, "q", _settings(), _count_chars)

    fixed_tokens = len("system about 100 words") + len("||q")
    assert estimate["chunk_count"] == 2
    assert estimate["input_tokens"] == 110 + 2 * fixed_tokens


def test_estimate_run_prices_low_typical_and_high_output(exhaustive_module):
    prices = {"gpt-4o-mini": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.01}}

    with mock.patch.object(exhaustive_module.litellm, "model_cost", prices):
        estimate = exhaustive_module.estimate_run([_paper(exhaustive_module)], "q", _settings(), _count_chars)

    input_cost = estimate["input_tokens"] * 0.001
    assert estimate["cost_usd"] == pytest.approx(
        {
            label: input_cost + 2 * output_tokens * 0.01
            for label, output_tokens in exhaustive_module.OUTPUT_TOKENS_PER_CHUNK.items()
        }
    )


def test_estimate_run_leaves_cost_unknown_when_model_has_no_price(exhaustive_module):
    with mock.patch.object(exhaustive_module.litellm, "model_cost", {}):
        estimate = exhaustive_module.estimate_run([_paper(exhaustive_module)], "q", _settings(), _count_chars)

    assert estimate["cost_usd"] is None


def test_estimate_run_derives_minutes_from_the_rate_limit(exhaustive_module):
    settings = _settings(rate_limit={"gpt-4o-mini": "100 per 1 minute"})

    estimate = exhaustive_module.estimate_run([_paper(exhaustive_module)], "q", settings, _count_chars)

    typical_output = exhaustive_module.OUTPUT_TOKENS_PER_CHUNK["typical"]
    limiter_tokens = estimate["input_tokens"] * exhaustive_module.LIMITER_TOKEN_FACTOR + 2 * typical_output
    assert estimate["estimated_minutes"] == pytest.approx(limiter_tokens / 100)


def test_estimate_run_uses_call_speed_when_there_is_no_rate_limit(exhaustive_module):
    estimate = exhaustive_module.estimate_run([_paper(exhaustive_module)], "q", _settings(), _count_chars)

    assert estimate["estimated_minutes"] == pytest.approx(2 * exhaustive_module.SECONDS_PER_CALL / 4 / 60)


def test_estimate_run_uses_call_speed_when_it_is_slower_than_the_rate_limit(exhaustive_module):
    settings = _settings(rate_limit={"gpt-4o-mini": "1500000 per 1 minute"})
    paper = _paper(exhaustive_module, chunk_count=2000, chunk_tokens=2_000_000, citation_tokens=0)

    estimate = exhaustive_module.estimate_run([paper], "q", settings, _count_chars)

    assert estimate["estimated_minutes"] == pytest.approx(2000 * exhaustive_module.SECONDS_PER_CALL / 4 / 60)


def test_build_plan_lists_every_paper_and_counts_statuses(exhaustive_module, shard_dir, manifest_path):
    _write_manifest(manifest_path, ["good.pdf", "failed.pdf", "absent.pdf"])
    _write_files_zip(shard_dir, {"good.pdf": "hash-good", "failed.pdf": exhaustive_module.FAILED_DOCUMENT_ADD_ID})
    _write_saved_docs(shard_dir, "hash-good", _make_docs("good", ["alpha"]))

    with mock.patch.object(exhaustive_module, "make_token_counter", return_value=_count_chars):
        plan = exhaustive_module.build_plan("q", settings=_settings(), index_dir=shard_dir)

    assert plan["question"] == "q"
    assert plan["model"] == "gpt-4o-mini"
    assert plan["paper_count"] == 3
    assert plan["status_counts"] == {
        exhaustive_module.STATUS_LOADABLE: 1,
        exhaustive_module.STATUS_FAILED: 1,
        exhaustive_module.STATUS_NOT_INDEXED: 1,
    }
    assert [paper["file_location"] for paper in plan["papers"]] == ["absent.pdf", "failed.pdf", "good.pdf"]
    assert plan["chunk_count"] == 1


def test_build_plan_defaults_to_the_active_index_without_creating_it(exhaustive_module, workspace_tmp_path, manifest_path):
    _write_manifest(manifest_path, ["a.pdf"])

    with mock.patch.object(exhaustive_module, "make_token_counter", return_value=_count_chars):
        plan = exhaustive_module.build_plan("q")

    assert plan["status_counts"] == {exhaustive_module.STATUS_NOT_INDEXED: 1}
    assert not (workspace_tmp_path / "index").exists()


def _seed_papers(module, shard_dir, manifest_path, papers: dict[str, list[str]], extra_locations=()):
    _write_manifest(manifest_path, [*papers, *extra_locations])
    _write_files_zip(shard_dir, {location: f"hash-{location}" for location in papers})
    for location, chunks in papers.items():
        _write_saved_docs(shard_dir, f"hash-{location}", _make_docs(location.removesuffix(".pdf"), chunks))


def _run(module, shard_dir, output_dir, model, settings=None, question="q"):
    return asyncio.run(
        module.run_exhaustive(
            question,
            output_dir,
            settings=settings or module.build_exhaustive_settings(),
            index_dir=shard_dir,
            summary_llm_model=model,
        )
    )


def _papers_by_location(summary):
    return {paper["file_location"]: paper for paper in summary["papers"]}


@pytest.fixture
def output_dir(workspace_tmp_path):
    return workspace_tmp_path / "run"


def test_build_exhaustive_settings_scores_at_temperature_zero_without_changing_defaults(exhaustive_module):
    settings = exhaustive_module.build_exhaustive_settings()

    params = settings.get_summary_llm().config["model_list"][0]["litellm_params"]
    default_params = exhaustive_module.bootstrap.build_settings().get_summary_llm().config["model_list"][0]["litellm_params"]
    assert params["temperature"] == 0
    # The default differs between lmi versions, so only check that it was left alone.
    assert default_params.get("temperature") != 0
    assert "temperature" not in exhaustive_module.bootstrap.build_settings().summary_llm_config


def test_build_exhaustive_settings_raises_speed_limits_without_changing_defaults(exhaustive_module):
    settings = exhaustive_module.build_exhaustive_settings()
    defaults = exhaustive_module.bootstrap.build_settings()

    assert settings.answer.max_concurrent_requests == 40
    assert settings.summary_llm_config["rate_limit"] == {"gpt-4o-mini": "1500000 per 1 minute"}
    assert defaults.answer.max_concurrent_requests == 4
    assert defaults.summary_llm_config["rate_limit"] == {"gpt-4o-mini": "30000 per 1 minute"}


def test_build_exhaustive_settings_processes_all_chunks_with_longer_summaries(exhaustive_module):
    settings = exhaustive_module.build_exhaustive_settings()

    assert settings.answer.evidence_retrieval is False
    assert settings.answer.evidence_summary_length == "about 200 words"


def test_build_exhaustive_settings_keeps_multimodal_off_and_the_same_index(exhaustive_module):
    settings = exhaustive_module.build_exhaustive_settings()

    assert settings.parsing.multimodal is False
    assert settings.parsing.should_parse_and_enrich_media == (False, False)
    assert settings.get_index_name() == exhaustive_module.bootstrap.build_settings().get_index_name()


def test_extract_pages_reads_the_page_range_from_a_chunk_name(exhaustive_module):
    assert exhaustive_module.extract_pages("Smith2020 pages 3-4") == "3-4"


def test_extract_pages_returns_empty_text_when_chunk_has_no_pages(exhaustive_module):
    assert exhaustive_module.extract_pages("Smith2020 chunk 1") == ""


def test_redact_secrets_removes_key_like_text(exhaustive_module):
    assert exhaustive_module.redact_secrets("Incorrect API key: sk-proj-abc_123*** given") == "Incorrect API key: [redacted] given"


@pytest.mark.parametrize(
    ("evidence_count", "failed_count", "expected"),
    [(2, 0, "relevant"), (2, 1, "relevant"), (0, 0, "no_relevant_evidence"), (0, 1, "incomplete")],
)
def test_classify_paper_only_reports_no_evidence_when_every_chunk_succeeded(
    exhaustive_module, evidence_count, failed_count, expected
):
    assert exhaustive_module.classify_paper(evidence_count, failed_count) == expected


@pytest.mark.parametrize(
    ("evidence_count", "failed_count", "possible_count", "expected"),
    [(1, 0, 1, "relevant"), (0, 1, 1, "incomplete"), (0, 0, 1, "possible_misses_only")],
)
def test_classify_paper_never_calls_a_paper_with_possible_misses_empty(
    exhaustive_module, evidence_count, failed_count, possible_count, expected
):
    assert exhaustive_module.classify_paper(evidence_count, failed_count, possible_count) == expected


def test_build_exhaustive_settings_uses_the_strict_prompt_without_changing_defaults(exhaustive_module):
    settings = exhaustive_module.build_exhaustive_settings()
    default_prompt = exhaustive_module.bootstrap.build_settings().prompts.summary_json_system

    prompt = settings.prompts.summary_json_system.format(summary_length="about 200 words")

    assert settings.prompts.summary_json_system == exhaustive_module.STRICT_SUMMARY_SYSTEM_PROMPT
    assert default_prompt != exhaustive_module.STRICT_SUMMARY_SYSTEM_PROMPT
    assert "about 200 words" in prompt
    assert '"quote"' in prompt
    assert "barrier" in prompt
    assert "Reference lists" in prompt


def test_extract_key_terms_keeps_topic_words_and_drops_generic_ones(exhaustive_module):
    terms = exhaustive_module.extract_key_terms("What ideas improve student engagement with feedback?")

    assert terms == {"engage", "feedba"}


def test_extract_key_terms_returns_nothing_for_a_generic_question(exhaustive_module):
    assert exhaustive_module.extract_key_terms("What ideas are proposed?") == set()


def test_citation_density_is_high_for_a_reference_list(exhaustive_module):
    text = "Smith, J. (2010). Feedback. Jones, K. (2011). Marking. Lee, A. (2012). Grades. doi:10.1/x"

    assert exhaustive_module.citation_density(text) > exhaustive_module.REFERENCE_LIST_DENSITY


def test_citation_density_is_low_for_prose(exhaustive_module):
    assert exhaustive_module.citation_density("Students read their comments and then revise the essay.") == 0


def test_citation_density_handles_empty_text(exhaustive_module):
    assert exhaustive_module.citation_density("") == 0


def test_is_possible_miss_flags_repeated_key_terms_in_prose(exhaustive_module):
    assert exhaustive_module.is_possible_miss("students ignore feedback " * 5, {"feedba"}) is True


def test_is_possible_miss_ignores_a_few_mentions(exhaustive_module):
    assert exhaustive_module.is_possible_miss("students ignore feedback " * 4, {"feedba"}) is False


def test_is_possible_miss_ignores_reference_lists(exhaustive_module):
    text = "Smith (2010). Feedback matters. " * 6

    assert exhaustive_module.is_possible_miss(text, {"feedba"}) is False


def test_is_possible_miss_is_false_without_key_terms(exhaustive_module):
    assert exhaustive_module.is_possible_miss("feedback " * 20, set()) is False


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("Students revise their work.", True),
        ("students revise   their work.", True),
        ("Students re-vise their work.", True),
        ("Students rewrite their work.", False),
        ("", False),
    ],
    ids=["exact", "spacing", "hyphen", "altered", "empty"],
)
def test_verify_quote_ignores_pdf_line_breaks_but_not_changed_words(exhaustive_module, quote, expected):
    assert exhaustive_module.verify_quote(quote, "Intro.\nStudents re-\nvise their\nwork. End.") is expected


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("The enthusiasm of the educators predicted value in the subject matter.", True),
        ("“The enthusiasm of the educators predicted value in the subject matter”", True),
        ('"inspiring students to learn."', True),
        ("The enthusiasm, of the educators predicted value", False),
        ("“.”", False),
    ],
    ids=["added-full-stop", "curly-quote-marks", "straight-quotes-and-full-stop", "inner-punctuation", "only-punctuation"],
)
def test_verify_quote_ignores_punctuation_only_at_the_ends(exhaustive_module, quote, expected):
    text = (
        "classroom:\n“The enthusiasm of the educators predicted value in the subject matter” "
        "(Cavanagh, p. 64). Teacher enthusiasm plays a role in inspiring students to learn, and more."
    )
    assert exhaustive_module.verify_quote(quote, text) is expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [(10, "best"), (9, "best"), (8, "likely"), (7, "likely"), (6, "possible"), (1, "possible")],
)
def test_tier_for_score_groups_scores_for_the_report(exhaustive_module, score, expected):
    assert exhaustive_module.tier_for_score(score) == expected


def test_run_records_tier_and_a_verified_quote(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["Students revise their work."]})
    reply = '{"quote": "Students revise", "summary": "idea", "relevance_score": 9}'

    summary = _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM(default=reply))

    (evidence,) = summary["papers"][0]["evidence"]
    assert (evidence["tier"], evidence["quote"], evidence["quote_verified"]) == ("best", "Students revise", True)
    assert summary["tier_counts"] == {"best": 1}


def test_run_marks_an_invented_quote_as_unverified(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["Students revise their work."]})
    reply = '{"quote": "Teachers mark essays", "summary": "idea", "relevance_score": 8}'

    summary = _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM(default=reply))

    assert summary["papers"][0]["evidence"][0]["quote_verified"] is False


def test_run_handles_a_quote_that_is_not_text(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["Students revise their work."]})
    reply = '{"quote": 12, "summary": "idea", "relevance_score": 8}'

    summary = _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM(default=reply))

    assert summary["papers"][0]["evidence"][0]["quote"] == "12"
    assert summary["papers"][0]["evidence"][0]["quote_verified"] is False


def test_run_keeps_rejected_chunks_that_keep_mentioning_the_topic(exhaustive_module, shard_dir, manifest_path, output_dir):
    chunk = "Students reject feedback when feedback is unclear. " * 3
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": [chunk, "Nothing on topic here."]})
    model = FakeSummaryLLM(default=IRRELEVANT_REPLY)

    summary = _run(exhaustive_module, shard_dir, output_dir, model, question="How can feedback be improved?")

    paper = summary["papers"][0]
    assert paper["status"] == exhaustive_module.RESULT_POSSIBLE_MISSES
    assert paper["evidence"] == []
    assert [item["raw_text"] for item in paper["possible_misses"]] == [chunk]
    assert paper["possible_misses"][0]["pages"] == "1"
    assert summary["possible_miss_count"] == 1


def test_run_reports_no_possible_misses_for_papers_that_were_not_searched(
    exhaustive_module, shard_dir, manifest_path, output_dir
):
    _write_manifest(manifest_path, ["absent.pdf"])

    summary = _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM())

    assert summary["papers"][0]["possible_misses"] == []
    assert summary["possible_miss_count"] == 0


def _evidence(score=9, tier="best", pages="3-4", summary="An idea.", quote="A quote.", verified=True):
    return {
        "chunk": f"Doc pages {pages}",
        "pages": pages,
        "score": score,
        "tier": tier,
        "summary": summary,
        "quote": quote,
        "quote_verified": verified,
        "raw_text": "RAW CHUNK TEXT",
    }


def _result(file_location="a.pdf", status="relevant", **overrides):
    values = {
        "file_location": file_location,
        "status": status,
        "detail": "",
        "chunks_total": 2,
        "chunks_failed": 0,
        "cost_usd": 0.01,
        "evidence": [],
        "possible_misses": [],
        "failures": [],
    }
    return values | overrides


def _summary(papers, **overrides):
    values = {
        "question": "How can feedback improve?",
        "model": "gpt-4o-mini",
        "paper_count": len(papers),
        "status_counts": {},
        "tier_counts": {},
        "possible_miss_count": 0,
        "failed_chunks": 0,
        "cost_usd": 0.1234,
        "cost_may_be_understated": False,
        "runtime_seconds": 180,
        "stopped_early": False,
        "stop_reason": None,
        "papers": papers,
    }
    return values | overrides


def test_load_manifest_titles_maps_paths_to_titles(exhaustive_module, manifest_path):
    manifest_path.write_text("file_location,title\nsub\\a.pdf,Paper A\nb.pdf,\n,No path\n", encoding="utf-8")

    assert exhaustive_module.load_manifest_titles() == {"sub/a.pdf": "Paper A"}


def test_load_manifest_titles_is_empty_without_a_manifest(exhaustive_module):
    assert exhaustive_module.load_manifest_titles() == {}


def test_load_manifest_titles_is_empty_when_manifest_has_no_title_column(exhaustive_module, manifest_path):
    _write_manifest(manifest_path, ["a.pdf"])

    assert exhaustive_module.load_manifest_titles() == {}


def test_paper_title_prefers_the_manifest_title(exhaustive_module):
    assert exhaustive_module.paper_title("sub/a.pdf", {"sub/a.pdf": "Paper A"}) == "Paper A"


def test_paper_title_falls_back_to_the_file_name(exhaustive_module):
    assert exhaustive_module.paper_title("sub/a paper.pdf", {}) == "a paper"


@pytest.mark.parametrize(
    "status",
    [
        "relevant",
        "possible_misses_only",
        "no_relevant_evidence",
        "incomplete",
        "not_processed",
        "failed_to_index",
        "not_indexed",
        "unreadable",
    ],
)
def test_render_report_lists_a_paper_exactly_once_whatever_its_status(exhaustive_module, status):
    evidence = [_evidence()] if status == "relevant" else []
    misses = [_evidence(score=0)] if status == "possible_misses_only" else []
    summary = _summary([_result("unique-name.pdf", status, evidence=evidence, possible_misses=misses)])

    report = exhaustive_module.render_report(summary, {})

    assert report.count("unique-name.pdf") == 1


def test_render_report_says_it_is_an_evidence_listing_that_can_misjudge(exhaustive_module):
    report = exhaustive_module.render_report(_summary([]), {})

    assert "not a conclusion" in report
    assert "can still misjudge relevance" in report


def test_render_report_shows_question_cost_and_runtime(exhaustive_module):
    report = exhaustive_module.render_report(_summary([]), {})

    assert "How can feedback improve?" in report
    assert "$0.1234" in report
    assert "3.0 minutes" in report


def test_render_report_warns_when_cost_may_be_low(exhaustive_module):
    report = exhaustive_module.render_report(_summary([], cost_may_be_understated=True), {})

    assert "may be slightly higher" in report


def test_render_report_warns_at_the_top_when_a_run_stopped_early(exhaustive_module):
    summary = _summary([_result("a.pdf", "not_processed", detail="Stopped.")], stopped_early=True, stop_reason="Stopped.")

    report = exhaustive_module.render_report(summary, {})

    assert report.index("**This run did not finish.**") < report.index("## Summary")


def test_render_report_warns_at_the_top_when_chunks_failed(exhaustive_module):
    report = exhaustive_module.render_report(_summary([], failed_chunks=3), {})

    assert report.index("3 chunks could not be checked") < report.index("## Summary")


def test_render_report_puts_papers_with_problems_before_evidence(exhaustive_module):
    papers = [
        _result("good.pdf", evidence=[_evidence()]),
        _result("broken.pdf", "unreadable", detail="Could not load."),
    ]

    report = exhaustive_module.render_report(_summary(papers), {})

    assert report.index("broken.pdf") < report.index("good.pdf")
    assert "Could not load." in report


def test_render_report_shows_failed_chunks_on_a_paper_that_also_has_evidence(exhaustive_module):
    failure = {"chunk": "Doc pages 7", "pages": "7", "failed": True, "reason": "RateLimitError", "cost_usd": 0}
    paper = _result(evidence=[_evidence()], failures=[failure], chunks_failed=1)

    report = exhaustive_module.render_report(_summary([paper]), {})

    assert "1 of 2 chunks could not be checked" in report
    assert "RateLimitError" in report


def test_render_report_shows_why_an_incomplete_paper_is_incomplete(exhaustive_module):
    failure = {"chunk": "Doc pages 7", "pages": "7", "failed": True, "reason": "RateLimitError", "cost_usd": 0}
    paper = _result(status="incomplete", failures=[failure], chunks_failed=1)

    report = exhaustive_module.render_report(_summary([paper]), {})

    assert "1 of 2 chunks could not be checked" in report


def test_render_report_orders_tiers_and_collapses_only_the_lowest(exhaustive_module):
    paper = _result(
        evidence=[
            _evidence(score=9, tier="best", summary="BEST ONE"),
            _evidence(score=7, tier="likely", summary="LIKELY ONE"),
            _evidence(score=5, tier="possible", summary="POSSIBLE ONE"),
        ]
    )

    report = exhaustive_module.render_report(_summary([paper]), {})

    assert report.index("BEST ONE") < report.index("LIKELY ONE") < report.index("<details>") < report.index("POSSIBLE ONE")


def test_render_report_shows_pages_score_and_quote_status(exhaustive_module):
    paper = _result(
        evidence=[
            _evidence(pages="3-4", quote="Checked quote.", verified=True),
            _evidence(score=8, tier="likely", quote="Doubtful quote.", verified=False),
        ]
    )

    report = exhaustive_module.render_report(_summary([paper]), {})

    assert "Pages 3-4 (score 9)" in report
    assert 'Quote (verified): "Checked quote."' in report
    assert 'Quote (not verified, check the PDF): "Doubtful quote."' in report


def test_render_report_handles_missing_pages_quote_and_summary(exhaustive_module):
    paper = _result(evidence=[_evidence(pages="", summary="", quote="")])

    report = exhaustive_module.render_report(_summary([paper]), {})

    assert "Page unknown (score 9)" in report
    assert "No summary given." in report
    assert "Quote (" not in report


def test_render_report_keeps_each_evidence_item_on_one_line(exhaustive_module):
    paper = _result(evidence=[_evidence(summary="first\nsecond", quote="third\nfourth")])

    report = exhaustive_module.render_report(_summary([paper]), {})

    assert "first second" in report
    assert "third fourth" in report


def test_render_report_leaves_out_raw_chunk_text(exhaustive_module):
    report = exhaustive_module.render_report(_summary([_result(evidence=[_evidence()])]), {})

    assert "RAW CHUNK TEXT" not in report


def test_render_report_puts_papers_with_the_most_best_matches_first(exhaustive_module):
    papers = [
        _result("one.pdf", evidence=[_evidence()]),
        _result("three.pdf", evidence=[_evidence(), _evidence(), _evidence()]),
    ]

    report = exhaustive_module.render_report(_summary(papers), {})

    assert report.index("three.pdf") < report.index("one.pdf")


def test_render_report_lists_possible_misses_for_checking(exhaustive_module):
    paper = _result(evidence=[_evidence()], possible_misses=[_evidence(score=0, pages="8")])

    report = exhaustive_module.render_report(_summary([paper]), {})

    assert "Possible misses" in report
    assert "Page 8" not in report
    assert "Pages 8: scored 0" in report


def test_render_report_uses_manifest_titles(exhaustive_module):
    summary = _summary([_result("a.pdf", evidence=[_evidence()])])

    report = exhaustive_module.render_report(summary, {"a.pdf": "A Readable Title"})

    assert "A Readable Title" in report


def test_render_report_counts_each_status_and_tier(exhaustive_module):
    summary = _summary(
        [], status_counts={"relevant": 2, "unreadable": 1}, tier_counts={"best": 4}, possible_miss_count=5, paper_count=3
    )

    report = exhaustive_module.render_report(summary, {})

    assert "Papers in scope: 3" in report
    assert "Relevant evidence found: 2" in report
    assert "Unreadable: 1" in report
    assert "Best matches: 4" in report
    assert "Possible misses: 5" in report


def test_run_writes_a_report_with_manifest_titles(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["x"]})
    manifest_path.write_text("file_location,title\na.pdf,Saved Title\n", encoding="utf-8")

    _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM())

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Saved Title" in report
    assert "a relevant idea" in report


def test_render_report_escapes_html_from_titles_summaries_and_quotes(exhaustive_module):
    paper = _result(
        "a.pdf", evidence=[_evidence(summary="<script>bad()</script>", quote="<img src=x>")]
    )

    report = exhaustive_module.render_report(_summary([paper]), {"a.pdf": "<b>Title</b>"})

    assert "<script>" not in report
    assert "<img" not in report
    assert "<b>" not in report
    assert "&lt;script&gt;" in report


def test_main_plan_prints_the_plan_as_json(exhaustive_module, capsys):
    with mock.patch.object(exhaustive_module, "build_plan", return_value={"paper_count": 1}) as build_plan:
        exit_code = exhaustive_module.main(["plan", "q"])

    assert exit_code == 0
    build_plan.assert_called_once_with("q")
    assert json.loads(capsys.readouterr().out) == {"ok": True, "plan": {"paper_count": 1}}


def test_main_prints_plain_ascii_so_windows_pipes_cannot_fail(exhaustive_module, capsys):
    with mock.patch.object(exhaustive_module, "build_plan", return_value={"title": "café"}):
        exhaustive_module.main(["plan", "q"])

    assert capsys.readouterr().out.isascii()


def test_main_refuses_to_run_without_confirmation(exhaustive_module, capsys, output_dir):
    with mock.patch.object(exhaustive_module, "run_exhaustive") as run:
        exit_code = exhaustive_module.main(["run", "q", "--output-dir", str(output_dir)])

    assert exit_code == 1
    run.assert_not_called()
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_main_refuses_to_run_without_an_output_folder(exhaustive_module, capsys):
    with mock.patch.object(exhaustive_module, "run_exhaustive") as run:
        exit_code = exhaustive_module.main(["run", "q", "--confirmed"])

    assert exit_code == 1
    run.assert_not_called()


def test_main_runs_when_confirmed(exhaustive_module, capsys, output_dir):
    run = mock.AsyncMock(return_value={})

    with mock.patch.object(exhaustive_module, "run_exhaustive", run):
        exit_code = exhaustive_module.main(["run", "q", "--output-dir", str(output_dir), "--confirmed"])

    assert exit_code == 0
    run.assert_awaited_once_with("q", output_dir)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_main_reports_errors_without_leaking_keys(exhaustive_module, capsys):
    with mock.patch.object(exhaustive_module, "build_plan", side_effect=ValueError("bad key sk-secret123")):
        exit_code = exhaustive_module.main(["plan", "q"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["error_type"] == "ValueError"
    assert "sk-secret123" not in payload["error"]


def test_run_reports_every_in_scope_paper_whatever_happened_to_it(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(
        exhaustive_module,
        shard_dir,
        manifest_path,
        {"relevant.pdf": ["a"], "irrelevant.pdf": ["b"], "incomplete.pdf": ["c"]},
        extra_locations=["absent.pdf", "failed.pdf"],
    )
    index_files = {location: f"hash-{location}" for location in ("relevant.pdf", "irrelevant.pdf", "incomplete.pdf")}
    _write_files_zip(shard_dir, index_files | {"failed.pdf": exhaustive_module.FAILED_DOCUMENT_ADD_ID})
    model = FakeSummaryLLM(
        {"irrelevant pages 1": IRRELEVANT_REPLY, "incomplete pages 1": _litellm_error(litellm.RateLimitError)}
    )

    summary = _run(exhaustive_module, shard_dir, output_dir, model)

    assert {location: paper["status"] for location, paper in _papers_by_location(summary).items()} == {
        "absent.pdf": exhaustive_module.STATUS_NOT_INDEXED,
        "failed.pdf": exhaustive_module.STATUS_FAILED,
        "incomplete.pdf": exhaustive_module.RESULT_INCOMPLETE,
        "irrelevant.pdf": exhaustive_module.RESULT_NO_EVIDENCE,
        "relevant.pdf": exhaustive_module.RESULT_RELEVANT,
    }
    assert summary["paper_count"] == 5


def test_run_processes_every_chunk_regardless_of_evidence_k(exhaustive_module, shard_dir, manifest_path, output_dir):
    chunks = [f"chunk {i}" for i in range(12)]
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"big.pdf": chunks})
    settings = exhaustive_module.build_exhaustive_settings()
    settings.answer.evidence_k = 1
    model = FakeSummaryLLM()

    summary = _run(exhaustive_module, shard_dir, output_dir, model, settings=settings)

    assert len(set(model.calls)) == 12
    assert summary["papers"][0]["chunks_total"] == 12
    assert len(summary["papers"][0]["evidence"]) == 12


def test_run_keeps_summary_pages_and_raw_text_sorted_by_score(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["low text", "high text"]})
    model = FakeSummaryLLM(
        {
            "p pages 1": '{"summary": "minor", "relevance_score": 2}',
            "p pages 2": '{"summary": "major", "relevance_score": 9}',
        }
    )

    summary = _run(exhaustive_module, shard_dir, output_dir, model)

    first, second = summary["papers"][0]["evidence"]
    assert (first["score"], first["summary"], first["pages"], first["raw_text"]) == (9, "major", "2", "high text")
    assert second["score"] == 2


@pytest.mark.parametrize(
    "reply",
    [
        _litellm_error(litellm.RateLimitError),
        _litellm_error(litellm.Timeout),
        _litellm_error(litellm.BadRequestError),
        _litellm_error(litellm.APIConnectionError),
        _litellm_error(litellm.InternalServerError),
        TimeoutError("rate limiter wait"),
        BAD_JSON_REPLY,
        NULL_SCORE_REPLY,
    ],
    ids=["rate-limit", "timeout", "bad-request", "connection", "server", "limiter", "bad-json", "null-score"],
)
def test_run_never_reports_a_failed_chunk_as_no_relevant_evidence(
    exhaustive_module, shard_dir, manifest_path, output_dir, reply
):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"bad.pdf": ["a", "b"], "good.pdf": ["c"]})
    model = FakeSummaryLLM({"bad pages 1": reply}, default=IRRELEVANT_REPLY)
    model.replies["good pages 1"] = RELEVANT_REPLY

    summary = _run(exhaustive_module, shard_dir, output_dir, model)

    papers = _papers_by_location(summary)
    assert papers["bad.pdf"]["status"] == exhaustive_module.RESULT_INCOMPLETE
    assert papers["bad.pdf"]["chunks_failed"] == 1
    assert papers["bad.pdf"]["failures"][0]["pages"] == "1"
    assert papers["bad.pdf"]["failures"][0]["reason"]
    assert papers["good.pdf"]["status"] == exhaustive_module.RESULT_RELEVANT
    assert summary["failed_chunks"] == 1


def test_run_keeps_evidence_from_a_paper_with_one_failed_chunk(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["a", "b"]})
    model = FakeSummaryLLM({"p pages 2": _litellm_error(litellm.RateLimitError)})

    summary = _run(exhaustive_module, shard_dir, output_dir, model)

    paper = summary["papers"][0]
    assert paper["status"] == exhaustive_module.RESULT_RELEVANT
    assert (len(paper["evidence"]), paper["chunks_failed"]) == (1, 1)
    assert "RateLimitError" in paper["failures"][0]["reason"]


def test_run_redacts_key_like_text_from_failure_reasons(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["a"]})
    model = FakeSummaryLLM({"p pages 1": RuntimeError("bad key sk-secret123")})

    summary = _run(exhaustive_module, shard_dir, output_dir, model)

    assert "sk-secret123" not in summary["papers"][0]["failures"][0]["reason"]
    assert "sk-secret123" not in (output_dir / "results.jsonl").read_text(encoding="utf-8")


def test_run_adds_up_cost_including_abandoned_attempts(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["a", "b"]})
    model = FakeSummaryLLM({"p pages 2": BAD_JSON_REPLY}, cost=0.25)

    summary = _run(exhaustive_module, shard_dir, output_dir, model)

    assert summary["papers"][0]["cost_usd"] == pytest.approx(0.75)
    assert summary["cost_usd"] == pytest.approx(0.75)


def test_run_flags_cost_as_possibly_low_only_when_a_chunk_failed(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"p.pdf": ["a"]})

    clean = _run(exhaustive_module, shard_dir, output_dir / "clean", FakeSummaryLLM())
    failed = _run(
        exhaustive_module,
        shard_dir,
        output_dir / "failed",
        FakeSummaryLLM({"p pages 1": _litellm_error(litellm.Timeout)}),
    )

    assert clean["cost_may_be_understated"] is False
    assert failed["cost_may_be_understated"] is True


def test_run_saves_each_paper_before_the_next_so_a_crash_keeps_finished_work(
    exhaustive_module, shard_dir, manifest_path, output_dir
):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["x"], "b.pdf": ["1", "2", "3", "4", "5"]})
    settings = exhaustive_module.build_exhaustive_settings()
    settings.answer.max_concurrent_requests = 1
    model = FakeSummaryLLM({"b pages 5": KeyboardInterrupt()})

    with pytest.raises(KeyboardInterrupt):
        _run(exhaustive_module, shard_dir, output_dir, model, settings=settings)

    lines = (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["file_location"] for line in lines] == ["a.pdf"]


def test_run_writes_a_summary_file_matching_the_result(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["x"]})

    summary = _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM())

    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == summary
    assert summary["question"] == "q"
    assert summary["model"] == "gpt-4o-mini"
    assert summary["status_counts"] == {exhaustive_module.RESULT_RELEVANT: 1}
    assert summary["runtime_seconds"] >= 0


def test_run_marks_a_paper_unreadable_when_its_chunks_vanish_after_scoping(
    exhaustive_module, shard_dir, manifest_path, output_dir
):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["x"]})
    real_load = exhaustive_module.load_saved_docs
    outcomes = iter([real_load(shard_dir, "hash-a.pdf"), OSError("gone")])

    with mock.patch.object(exhaustive_module, "load_saved_docs", side_effect=outcomes):
        summary = _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM())

    assert summary["papers"][0]["status"] == exhaustive_module.STATUS_UNREADABLE
    assert "OSError" in summary["papers"][0]["detail"]


def test_run_stops_after_repeated_failures_and_marks_the_rest_not_processed(
    exhaustive_module, shard_dir, manifest_path, output_dir
):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["1", "2"], "b.pdf": ["3"], "c.pdf": ["4"]})
    settings = exhaustive_module.build_exhaustive_settings()
    settings.answer.max_concurrent_requests = 1
    model = FakeSummaryLLM(default=_litellm_error(litellm.RateLimitError))

    with mock.patch.object(exhaustive_module, "MAX_CONSECUTIVE_FAILED_CHUNKS", 2):
        summary = _run(exhaustive_module, shard_dir, output_dir, model, settings=settings)

    papers = _papers_by_location(summary)
    assert papers["a.pdf"]["status"] == exhaustive_module.RESULT_INCOMPLETE
    assert papers["b.pdf"]["status"] == exhaustive_module.RESULT_NOT_PROCESSED
    assert papers["c.pdf"]["status"] == exhaustive_module.RESULT_NOT_PROCESSED
    assert papers["c.pdf"]["detail"] == summary["stop_reason"]
    assert summary["stopped_early"] is True
    assert len(model.calls) == 2


def test_run_stops_inside_a_paper_and_reports_it_incomplete(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["1", "2", "3"]})
    settings = exhaustive_module.build_exhaustive_settings()
    settings.answer.max_concurrent_requests = 1
    model = FakeSummaryLLM(default=_litellm_error(litellm.RateLimitError))

    with mock.patch.object(exhaustive_module, "MAX_CONSECUTIVE_FAILED_CHUNKS", 2):
        summary = _run(exhaustive_module, shard_dir, output_dir, model, settings=settings)

    paper = summary["papers"][0]
    assert paper["status"] == exhaustive_module.RESULT_INCOMPLETE
    assert paper["chunks_failed"] == 3
    assert paper["failures"][-1]["reason"] == summary["stop_reason"]
    assert len(model.calls) == 2


def test_run_searches_papers_side_by_side_up_to_the_request_limit(exhaustive_module, shard_dir, manifest_path, output_dir):
    papers = {f"{i}.pdf": ["x"] for i in range(12)}
    _seed_papers(exhaustive_module, shard_dir, manifest_path, papers)
    settings = exhaustive_module.build_exhaustive_settings()
    settings.answer.max_concurrent_requests = 6
    model = FakeSummaryLLM()

    summary = _run(exhaustive_module, shard_dir, output_dir, model, settings=settings)

    assert model.max_active == 6
    assert len(summary["papers"]) == 12


def test_run_lists_papers_in_scope_order_whatever_order_they_finish(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["1", "2", "3"], "b.pdf": ["x"]})

    summary = _run(exhaustive_module, shard_dir, output_dir, FakeSummaryLLM())

    assert [paper["file_location"] for paper in summary["papers"]] == ["a.pdf", "b.pdf"]


def test_run_keeps_going_when_a_success_breaks_the_failure_streak(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["1", "2"], "b.pdf": ["3", "4"]})
    error = _litellm_error(litellm.RateLimitError)
    model = FakeSummaryLLM({"a pages 1": error, "b pages 1": error})

    with mock.patch.object(exhaustive_module, "MAX_CONSECUTIVE_FAILED_CHUNKS", 2):
        summary = _run(exhaustive_module, shard_dir, output_dir, model)

    assert summary["stopped_early"] is False
    assert summary["stop_reason"] is None


def test_run_limits_concurrent_requests_to_the_configured_value(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": [str(i) for i in range(10)]})
    settings = exhaustive_module.build_exhaustive_settings()
    settings.answer.max_concurrent_requests = 2
    model = FakeSummaryLLM()

    _run(exhaustive_module, shard_dir, output_dir, model, settings=settings)

    assert model.max_active == 2


def test_run_uses_exhaustive_settings_and_active_index_by_default(exhaustive_module, manifest_path, output_dir):
    _write_manifest(manifest_path, ["a.pdf"])
    settings = exhaustive_module.build_exhaustive_settings()

    with mock.patch.object(exhaustive_module, "build_exhaustive_settings", return_value=settings) as build:
        summary = asyncio.run(exhaustive_module.run_exhaustive("q", output_dir, summary_llm_model=FakeSummaryLLM()))

    build.assert_called_once_with()
    assert summary["status_counts"] == {exhaustive_module.STATUS_NOT_INDEXED: 1}


def test_run_builds_the_summary_model_from_settings_when_none_is_given(exhaustive_module, shard_dir, manifest_path, output_dir):
    _seed_papers(exhaustive_module, shard_dir, manifest_path, {"a.pdf": ["x"]})
    model = FakeSummaryLLM()
    real_settings = exhaustive_module.build_exhaustive_settings()
    settings = SimpleNamespace(
        summary_llm=real_settings.summary_llm,
        answer=real_settings.answer,
        prompts=real_settings.prompts,
        get_summary_llm=lambda: model,
    )

    asyncio.run(exhaustive_module.run_exhaustive("q", output_dir, settings=settings, index_dir=shard_dir))

    assert model.calls == ["a pages 1"]
