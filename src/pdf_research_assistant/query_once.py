"""Internal one-shot query helper.

Runs a single PaperQA query in a fresh process so the Streamlit app and CLI
can avoid cross-query asyncio/LiteLLM state issues.
"""

import base64
import json
import os
import sys

os.environ.setdefault("PQA_INDEX_DONT_CACHE_INDEXES", "true")

from lmi.cost_tracker import GLOBAL_COST_TRACKER, enable_cost_tracking
from paperqa import Docs
from paperqa.agents.main import run_agent
from paperqa.utils import run_or_ensure

from pdf_research_assistant import rate_profiles, saved_answers, settings_groups
from pdf_research_assistant.bootstrap import build_settings


def serialize_media(media) -> dict:
    """Convert one image or table stored with a chunk into a JSON-safe dictionary for the UI."""
    info = getattr(media, "info", None) or {}
    data = getattr(media, "data", None) or b""
    return {
        "index": getattr(media, "index", 0),
        "kind": info.get("type"),
        "page": info.get("page_num"),
        "text": getattr(media, "text", None) or "",
        "description": info.get("enriched_description") or "",
        "image": base64.b64encode(data).decode("ascii"),
        "url": getattr(media, "url", None) or "",
    }


def passage_key(text) -> tuple:
    """Identify a chunk by its name and the document it came from."""
    return getattr(text, "name", None), getattr(getattr(text, "doc", None), "dockey", None)


def serialize_contexts(contexts, cited_ids=(), passages=()):
    """Convert PaperQA context objects into JSON-safe dictionaries for the UI.

    cited_ids holds the ids of the contexts the answer cites. passages holds the full chunks the
    question read (the Docs object's texts). PaperQA's answer empties each context's chunk text and
    drops its images and tables, so each chunk is looked up again there by name and document.
    """
    full_chunks = {passage_key(passage): passage for passage in passages}
    items = []
    for ctx in list(contexts or []):
        text = getattr(ctx, "text", None)
        if text:
            text = full_chunks.get(passage_key(text), text)
        context_id = getattr(ctx, "id", "")
        items.append(
            {
                "id": context_id,
                "name": getattr(text, "name", "Source") if text else "Source",
                "cited": context_id in cited_ids,
                "context": getattr(ctx, "context", "") or "",
                "raw_text": getattr(text, "text", "") if text else "",
                "media": [serialize_media(media) for media in (getattr(text, "media", None) or [])],
                # PaperQA's 0-10 relevance rating from the summary step; -1 means it was never set.
                "score": getattr(ctx, "score", -1),
            }
        )
    return items


USAGE_ERROR = "Expected a question, optionally followed by one-off settings as a JSON object."
# Shown in place of the answer when PaperQA stops without writing one, so the reply is never blank.
NO_ANSWER_NOTE = (
    "No answer was written: PaperQA stopped before writing one (it said \"{reason}\"). "
    "Try the question again, ask it another way, or gather more passages."
)


def answer_text(session) -> str:
    """Return the answer to show, or a note saying why there is none when PaperQA wrote no answer."""
    formatted = (getattr(session, "formatted_answer", "") or "").strip()
    if formatted:
        return formatted
    reason = (getattr(session, "answer", "") or "").strip() or "No answer generated."
    return NO_ANSWER_NOTE.format(reason=reason)


def question_cost(session_cost: float | None, tracked_cost: float | None) -> float:
    """Return what the question cost, embeddings included.

    PaperQA's session cost adds up only the model's calls, so it leaves out the embeddings that find
    passages; with a local model it is always zero. PaperQA's model client tracks the cost of every call
    it makes, embeddings included. The larger of the two is used, so a tracker that saw nothing never
    hides the session's own figure.
    """
    return max(session_cost or 0.0, tracked_cost or 0.0)


def run_query_payload(question: str, one_off: dict | None = None) -> dict:
    """Run one PaperQA query, keep its answer for searching later, and return a serializable payload.

    one_off holds settings changed for this question only, by setting key. They go on top of the saved
    settings for this run and are never saved; any that cannot be used are skipped.
    """
    try:
        settings = build_settings()
        applied = settings_groups.apply_usable(settings, one_off or {})
        # Pictures make each request weigh a lot against the rate limit, so let requests wait their turn.
        rate_profiles.allow_limiter_wait(settings)
        # PaperQA adds every chunk the question reads to this Docs object, text, images and tables included.
        docs = Docs()
        # Off by default; switched on so the embedding calls' cost is counted too.
        enable_cost_tracking()
        tracked_before = GLOBAL_COST_TRACKER.lifetime_cost_usd
        response = run_or_ensure(
            coro=run_agent(
                docs,
                question,
                settings,
                agent_type=settings.agent.agent_type,
            )
        )
        cost = question_cost(response.session.cost, GLOBAL_COST_TRACKER.lifetime_cost_usd - tracked_before)
        # The saved answer takes its cost from the session, so it shows the same figure as the chat.
        response.session.cost = cost
        contexts = serialize_contexts(
            getattr(response.session, "contexts", []),
            cited_ids=getattr(response.session, "used_contexts", ()),
            passages=docs.texts,
        )
        recorded = settings_groups.record_answer_settings(settings, one_off=applied)
        payload = {
            "ok": True,
            "answer": answer_text(response.session),
            "cost": cost,
            "contexts": contexts,
            "settings": recorded,
        }
        # Kept where PaperQA's own `pqa ask` keeps its answers, so it can be searched later.
        save_error = saved_answers.save_answer(
            settings.agent.index.index_directory, response, saved_answers.build_record(contexts, recorded)
        )
        if save_error:
            payload["save_error"] = save_error
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def use_utf8_output(stream) -> None:
    """Make a stream write UTF-8, which is what the app and the CLI read.

    A piped stdout on Windows uses the console code page instead, which garbles curly quotes and cannot
    write some characters at all, so an answer that had already been paid for would be lost.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")


def main() -> int:
    """Execute the one-shot query CLI contract expected by the app and shell wrapper."""
    use_utf8_output(sys.stdout)
    if len(sys.argv) not in (2, 3):
        print(json.dumps({"ok": False, "error": USAGE_ERROR}))
        return 1
    one_off = None
    if len(sys.argv) == 3:
        try:
            one_off = json.loads(sys.argv[2])
        except ValueError:
            one_off = None
        if not isinstance(one_off, dict):
            print(json.dumps({"ok": False, "error": USAGE_ERROR}))
            return 1

    question = sys.argv[1]
    payload = run_query_payload(question, one_off)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

