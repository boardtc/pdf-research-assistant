"""Internal one-shot query helper.

Runs a single PaperQA query in a fresh process so the Streamlit app and CLI
can avoid cross-query asyncio/LiteLLM state issues.
"""

import base64
import json
import os
import sys

os.environ.setdefault("PQA_INDEX_DONT_CACHE_INDEXES", "true")

from paperqa import Docs
from paperqa.agents.main import run_agent
from paperqa.utils import run_or_ensure

from pdf_research_assistant import settings_groups
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


def serialize_contexts(contexts, cited_ids=()):
    """Convert PaperQA context objects into JSON-safe dictionaries for the UI.

    cited_ids holds the ids of the contexts the answer cites.
    """
    items = []
    for ctx in list(contexts or []):
        text = getattr(ctx, "text", None)
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


def run_query_payload(question: str, one_off: dict | None = None) -> dict:
    """Run one PaperQA query and return a serializable success or error payload.

    one_off holds settings changed for this question only, by setting key. They go on top of the saved
    settings for this run and are never saved; any that cannot be used are skipped.
    """
    try:
        settings = build_settings()
        applied = settings_groups.apply_usable(settings, one_off or {})
        response = run_or_ensure(
            coro=run_agent(
                Docs(),
                question,
                settings,
                agent_type=settings.agent.agent_type,
            )
        )
        return {
            "ok": True,
            "answer": response.session.formatted_answer,
            "cost": response.session.cost,
            "contexts": serialize_contexts(
                getattr(response.session, "contexts", []),
                cited_ids=getattr(response.session, "used_contexts", ()),
            ),
            "settings": settings_groups.record_answer_settings(settings, one_off=applied),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> int:
    """Execute the one-shot query CLI contract expected by the app and shell wrapper."""
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

