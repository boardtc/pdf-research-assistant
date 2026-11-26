"""Keep each answer in PaperQA's own index of past answers, and search it.

PaperQA's `agent_query` (behind `pqa ask`) adds every answer to a full-text index named "answers" in the
index folder, and `pqa search -i answers` searches it. Questions here run through `run_agent`, which skips
that step, so the query process adds each answer itself: same index, same fields, same stored form. The
app's own record of the answer (its passages, the settings used and when it was asked) is stored beside
PaperQA's answer, where PaperQA ignores it, so `pqa search -i answers` still reads these answers.

The pictures shown with the passages are kept apart, each once, in a "pictures" folder inside the answers
folder, named after their contents; the record names them. Kept inline, a whole-page picture would add
about a megabyte to every answer that showed it, each time it showed it.

PaperQA is imported inside the functions, so modules that import this one load without it.
"""

import asyncio
import base64
import copy
import hashlib
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# PaperQA's names for its index of past answers and the fields it indexes (paperqa.agents.main.agent_query).
INDEX_NAME = "answers"
INDEX_FIELDS = ["file_location", "body", "question"]
APP_FIELD = "app"
PICTURES_FOLDER = "pictures"
# A kept picture's name: the SHA-256 of its bytes.
PICTURE_NAME = re.compile(r"[0-9a-f]{64}")
MAX_RESULTS = 10
# How often to try a save that fails, and how long to pause before the second try; the pause doubles after
# each try, so five tries pause three seconds in all. Failures on Windows tend to come a few in a row.
SAVE_ATTEMPTS = 5
RETRY_PAUSE_SECONDS = 0.2
# How often PaperQA tries for the writer lock when another writer holds it. Its own default of 1000 tries,
# each waiting up to a minute, would leave a question waiting for hours on a lock that never comes free;
# 8 tries wait a minute at most, about half a minute on average.
LOCK_ATTEMPTS = 8


class StoredAnswer(BaseModel):
    """A PaperQA answer as PaperQA stores it, plus the app's record under APP_FIELD.

    PaperQA reads a stored answer back with `AnswerResponse(**data)`, which ignores the extra field.
    """

    model_config = ConfigDict(extra="allow")


def answers_folder(index_directory) -> Path:
    return Path(index_directory) / INDEX_NAME


def pictures_folder(index_directory) -> Path:
    return answers_folder(index_directory) / PICTURES_FOLDER


def write_picture(folder: Path, data: bytes) -> str:
    """Keep one picture in the folder, unless it is already there, and return its name.

    It is written to a part file first and then renamed, so a picture that is there is always whole. Two
    saves keeping the same picture at once both end with it kept, whichever renames first.
    """
    name = hashlib.sha256(data).hexdigest()
    target = folder / name
    if target.exists():
        return name
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / f"{name}.{uuid.uuid4().hex}.part"
    try:
        part.write_bytes(data)
        os.replace(part, target)
    except OSError:
        if not target.exists():
            raise
    finally:
        part.unlink(missing_ok=True)
    return name


def keep_media_picture(folder: Path, media: dict) -> dict:
    """Return one image or table as the record keeps it: its picture kept in the folder and named.

    Media with no picture lose the empty field; a picture that is not base64 stays inline, since it has
    no bytes to keep.
    """
    kept = dict(media)
    image = kept.pop("image", None)
    if not image:
        return kept
    try:
        data = base64.b64decode(image, validate=True)
    except ValueError:
        kept["image"] = image
        return kept
    kept["picture"] = write_picture(folder, data)
    return kept


def keep_pictures(index_directory, record: dict) -> dict:
    """Return a copy of the app's record with each picture kept in the pictures folder instead of inline.

    The record given is left as it is, since the chat shows it too.
    """
    folder = pictures_folder(index_directory)
    kept = copy.deepcopy(record)
    for context in kept.get("contexts") or []:
        if "media" in context:
            context["media"] = [keep_media_picture(folder, media) for media in context["media"] or []]
    return kept


def load_picture(index_directory, name) -> bytes:
    """Return the bytes of a picture kept with the saved answers; raise if the name is not a picture's."""
    if not isinstance(name, str) or not PICTURE_NAME.fullmatch(name):
        raise ValueError(f"Not the name of a kept picture: {name!r}")
    return (pictures_folder(index_directory) / name).read_bytes()


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


def release_writer(index) -> None:
    """Let go of the index's writer, and with it the folder's writer lock, after a failed save.

    A writer whose save failed keeps the lock for as long as the process runs, even once nothing refers to
    it, so every later save in the folder waits for a lock that never comes free. Waiting for its merging
    threads ends the writer and frees the lock; anything it had not committed is dropped.
    """
    writer, index._writer = index._writer, None
    if writer is not None:
        try:
            writer.wait_merging_threads()
        except Exception:
            pass


async def add_answer(index_directory, response, record: dict) -> None:
    """Add one answer to the index the way PaperQA's agent_query does, with the app's record beside it.

    Its pictures are kept first, in the pictures folder. On Windows a save now and then fails with "Access
    is denied" on one of the index's own files, so a failed save is tried again, after its writer has been
    let go. A save can fail after its answer was committed; the next try then only records the answer as
    kept, so it is not added twice.
    """
    record = keep_pictures(index_directory, record)
    session = response.session
    fields = {"file_location": str(session.id), "body": session.answer, "question": session.question}
    kept_before = await open_index(index_directory).count
    for attempt in range(1, SAVE_ATTEMPTS + 1):
        index = open_index(index_directory)
        try:
            if await index.count > kept_before:
                (await index.index_files)[fields["file_location"]] = index.filehash(fields["body"])
            else:
                await index.add_document(
                    fields,
                    document=StoredAnswer(**response.model_dump(mode="json"), **{APP_FIELD: record}),
                    lock_acquisition_max_retries=LOCK_ATTEMPTS,
                )
            await index.save_index()
            return
        except Exception:
            release_writer(index)
            if attempt == SAVE_ATTEMPTS:
                raise
        await asyncio.sleep(RETRY_PAUSE_SECONDS * 2 ** (attempt - 1))


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
