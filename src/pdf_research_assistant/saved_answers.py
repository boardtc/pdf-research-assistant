"""Keep each answer in PaperQA's own index of past answers, and search it.

PaperQA's `agent_query` (behind `pqa ask`) adds every answer to a full-text index named "answers" in the
index folder, and `pqa search -i answers` searches it. Questions here run through `run_agent`, which skips
that step, so the query process adds each answer itself: same index, same fields, same stored form. The
app's own record of the answer (its passages, the settings used and when it was asked) is stored beside
PaperQA's answer, where PaperQA ignores it, so `pqa search -i answers` still reads these answers.

PaperQA is imported inside the functions, so modules that import this one load without it.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# PaperQA's names for its index of past answers and the fields it indexes (paperqa.agents.main.agent_query).
INDEX_NAME = "answers"
INDEX_FIELDS = ["file_location", "body", "question"]
APP_FIELD = "app"
MAX_RESULTS = 10


class StoredAnswer(BaseModel):
    """A PaperQA answer as PaperQA stores it, plus the app's record under APP_FIELD.

    PaperQA reads a stored answer back with `AnswerResponse(**data)`, which ignores the extra field.
    """

    model_config = ConfigDict(extra="allow")


def answers_folder(index_directory) -> Path:
    return Path(index_directory) / INDEX_NAME


def open_index(index_directory):
    """Open PaperQA's index of past answers in the index folder, creating it if it is not there yet."""
    from paperqa.agents.search import SearchDocumentStorage, SearchIndex

    return SearchIndex(
        fields=INDEX_FIELDS,
        index_name=INDEX_NAME,
        index_directory=index_directory,
        storage=SearchDocumentStorage.JSON_MODEL_DUMP,
    )


def build_record(contexts: list[dict], recorded_settings: dict | None, asked_at: str | None = None) -> dict:
    """Return what the app keeps beside PaperQA's answer: what it showed under the answer, and when."""
    return {
        "asked_at": asked_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "contexts": contexts,
        "settings": recorded_settings,
    }


async def add_answer(index_directory, response, record: dict) -> None:
    """Add one answer to the index the way PaperQA's agent_query does, with the app's record beside it."""
    index = open_index(index_directory)
    session = response.session
    await index.add_document(
        {"file_location": str(session.id), "body": session.answer, "question": session.question},
        document=StoredAnswer(**response.model_dump(mode="json"), **{APP_FIELD: record}),
    )
    await index.save_index()


def save_answer(index_directory, response, record: dict) -> str | None:
    """Keep one answer for searching later; return None, or why it could not be kept.

    It never raises: an answer that could not be kept is still a good answer to show.
    """
    try:
        asyncio.run(add_answer(index_directory, response, record))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def describe_stored(stored: object) -> dict | None:
    """Turn one stored answer into what the app shows, or None if it is not a PaperQA answer.

    Answers kept by PaperQA itself (`pqa ask`) have no app record, so they come without passages,
    settings or a date.
    """
    if not isinstance(stored, dict):
        return None
    session = stored.get("session")
    if not isinstance(session, dict):
        return None
    record = stored.get(APP_FIELD)
    if not isinstance(record, dict):
        record = {}
    return {
        "id": str(session.get("id", "")),
        "question": session.get("question", ""),
        "content": session.get("formatted_answer") or session.get("answer", ""),
        "cost": session.get("cost", 0.0),
        "asked_at": record.get("asked_at"),
        "contexts": record.get("contexts") or [],
        "settings": record.get("settings"),
    }


async def find_answers(index_directory, text: str, top_n: int) -> tuple[list, int]:
    """Return the stored answers that best match the text, best first, and how many answers are kept."""
    index = open_index(index_directory)
    return await index.query(query=text, top_n=top_n), await index.count


def search(index_directory, text: str, top_n: int = MAX_RESULTS) -> dict:
    """Search the questions and answers kept so far; return the matches and how many answers were searched.

    PaperQA names each stored file after the answer text, so two answers with the same text share one
    file and come back as the same answer; it is shown once.
    """
    found, searched = asyncio.run(find_answers(index_directory, text, top_n))
    answers = []
    seen = set()
    for stored in found:
        answer = describe_stored(stored)
        if answer is None or answer["id"] in seen:
            continue
        seen.add(answer["id"])
        answers.append(answer)
    return {"text": text, "answers": answers, "searched": searched}
