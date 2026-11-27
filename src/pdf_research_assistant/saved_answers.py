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
import functools
import hashlib
import os
import pickle
import re
import uuid
import zlib
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


async def delete_from_search(index, file_location: str) -> None:
    """Take one answer out of the index's search and count, as PaperQA's `delete_document` means to.

    PaperQA's own deletes the whole file location as a single term, but the field is split into words when
    it is indexed, so an answer's id, words joined by hyphens, matches nothing: the answer stays counted,
    and is only hidden from searches once its stored file is gone. A phrase of the id's words matches it.
    """
    from paperqa.agents.search import AsyncRetryError

    try:
        words = index.CLEAN_QUERY_REGEX.sub("", file_location)
        query = (await index.index).parse_query(f'"{words}"', ["file_location"])
        async with index.writer() as writer:
            writer.delete_documents_by_query(query)
        await index.save_index()
    except ValueError as exc:
        # PaperQA's own words for a writer lock another writer holds, which its delete tries again for.
        if "Failed to acquire Lockfile: LockBusy." in str(exc):
            raise AsyncRetryError("Failed to acquire lock") from exc
        raise


def limit_lock_waits(index) -> None:
    """Give the index a delete that works, and make it and the save try for the writer lock LOCK_ATTEMPTS
    times, not PaperQA's 1000, each waiting up to a minute.

    PaperQA's `remove_from_index` calls both through the index itself, so it uses them too.
    """
    from paperqa.agents.search import AsyncRetryError, SearchIndex
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

    @retry(
        stop=stop_after_attempt(LOCK_ATTEMPTS),
        wait=wait_random_exponential(multiplier=0.25, max=60),
        retry=retry_if_exception_type(AsyncRetryError),
        reraise=True,
    )
    async def delete_document(file_location: str) -> None:
        await delete_from_search(index, file_location)

    index.delete_document = delete_document
    retry_save = getattr(SearchIndex.save_index, "retry_with", None)
    if retry_save is not None:
        index.save_index = functools.partial(retry_save(stop=stop_after_attempt(LOCK_ATTEMPTS)), index)


def sharing_locations(index_files: dict, answer_id: str) -> list[str]:
    """Return the kept answers stored in the same file as this one, itself first; none if it is not kept."""
    filehash = index_files.get(answer_id)
    if not filehash:
        return []
    return [answer_id] + [location for location, kept in index_files.items() if kept == filehash and location != answer_id]


async def find_sharing(index_directory, answer_id: str) -> int:
    index = open_index(index_directory)
    return max(len(sharing_locations(await index.index_files, answer_id)) - 1, 0)


def count_sharing(index_directory, answer_id: str) -> int:
    """Return how many other kept answers share this one's stored file, because their text is the same."""
    return asyncio.run(find_sharing(index_directory, answer_id))


def picture_names(stored: object) -> set[str]:
    """Return the names of the kept pictures a stored answer's record shows."""
    record = stored.get(APP_FIELD) if isinstance(stored, dict) else None
    contexts = record.get("contexts") if isinstance(record, dict) else None
    names = set()
    for context in contexts if isinstance(contexts, list) else []:
        media_list = context.get("media") if isinstance(context, dict) else None
        for media in media_list if isinstance(media_list, list) else []:
            name = media.get("picture") if isinstance(media, dict) else None
            if isinstance(name, str) and PICTURE_NAME.fullmatch(name):
                names.add(name)
    return names


async def read_pictures(index, location: str) -> set[str]:
    """Return the pictures one kept answer shows; raise if its stored file cannot be read."""
    return picture_names(await index.get_saved_object(location))


async def pictures_in_use(index, locations) -> set[str] | None:
    """Return the pictures the kept answers at these locations show, or None if one cannot be read."""
    in_use = set()
    for location in locations:
        try:
            in_use |= await read_pictures(index, location)
        except Exception:
            return None
    return in_use


def remove_picture(index_directory, name: str) -> None:
    (pictures_folder(index_directory) / name).unlink(missing_ok=True)


async def save_empty_list(index) -> None:
    """Save PaperQA's list of kept answers once the last one has gone, as `save_index` would.

    `save_index` cannot save an empty list: it empties the list's file for writing and then, finding its
    list empty, reads it back from that file, which fails. Each delete has already committed its change.
    """
    path = await index.file_index_filename
    await path.write_bytes(zlib.compress(pickle.dumps({})))


async def remove_answer(index_directory, answer_id: str) -> None:
    """Take one kept answer out of PaperQA's index of answers, with any sharing its stored file.

    Every copy is taken out of the search first, and only then are their stored file and PaperQA's list of
    kept files changed, so a delete that fails partway still leaves a copy that can be shown. Its pictures
    go afterwards, unless another answer shows them or another answer cannot be read to tell.
    """
    index = open_index(index_directory)
    limit_lock_waits(index)
    index_files = await index.index_files
    locations = sharing_locations(index_files, answer_id)
    if not locations:
        return
    remaining = [location for location in index_files if location not in locations]
    try:
        pictures = await read_pictures(index, answer_id)
    except Exception:
        pictures = set()
    try:
        for location in locations:
            await index.delete_document(location)
        for location in locations:
            await index.remove_from_index(location)
        if remaining:
            await index.save_index()
        else:
            await save_empty_list(index)
    except Exception:
        release_writer(index)
        raise
    in_use = await pictures_in_use(index, remaining)
    if in_use is None:
        return
    for name in pictures - in_use:
        try:
            remove_picture(index_directory, name)
        except OSError:
            pass


def delete_answer(index_directory, answer_id: str) -> str | None:
    """Delete one kept answer for good; return None, or why it could not be deleted. It never raises."""
    try:
        asyncio.run(remove_answer(index_directory, answer_id))
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
