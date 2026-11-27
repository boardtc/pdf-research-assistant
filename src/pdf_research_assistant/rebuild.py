import asyncio
import contextvars
import importlib
import inspect
import json
import os
import pickle
import zlib
from pathlib import Path

import litellm
from lmi.cost_tracker import GLOBAL_COST_TRACKER, enable_cost_tracking
from paperqa.agents.search import get_directory_index
from paperqa.settings import Settings
from tantivy import Document
from tenacity import stop_after_attempt

from pdf_research_assistant import index_record, local_llm, rebuild_client, rebuild_estimate
from pdf_research_assistant.bootstrap import (
    FAILED_DOCUMENT_ADD_ID,
    PAPER_DIR,
    build_settings,
    get_allowed_paths,
    get_failed_files,
    get_files_to_index,
    get_indexed_doc_count,
)
from pdf_research_assistant.saved_answers import LOCK_ATTEMPTS, release_writer

settings = build_settings()

# How many PDFs were left unstarted because the app asked this rebuild to stop.
stop_state = {"skipped": 0}
# Whether PaperQA's build of the index ended with an error before every PDF was done.
run_state = {"failed": False}
# On Windows tantivy now and then fails to create one of the index's new files with "Access is denied" (#157).
# A failed write is tried this many times in all, each time with a new writer and a slightly longer pause.
WRITE_ATTEMPTS = 5
WRITE_RETRY_PAUSE_SECONDS = 0.5
# tantivy's words for a file it could not write, and for a writer that stopped after one of its threads did.
WRITE_ERROR_MARKERS = ("IoError", "IO error", "writer was killed", "worker thread encountered an error")
# Set while PaperQA describes a PDF's images and tables, so their calls' cost is counted as enrichment.
enriching = contextvars.ContextVar("enriching", default=False)
ESTIMATE_HEADING = "Rough estimate before starting:"


class _StopRequested(Exception):
    """Raised inside a PDF's turn when a stop was asked for before it started."""


def stop_requested() -> bool:
    """Say whether the app has asked this rebuild to stop, by creating the stop file it named."""
    stop_file = os.environ.get(rebuild_client.STOP_ENV)
    return bool(stop_file) and Path(stop_file).exists()


class _StopCheckingSemaphore:
    """Stands in for PaperQA's concurrency limit, and turns a PDF away once a stop was asked for.

    PaperQA queues every PDF at once and each takes its turn at this limit, so the turn is the moment a PDF
    starts. A PDF past this point is finished and saved as usual.
    """

    def __init__(self, inner):
        self._inner = inner

    async def __aenter__(self):
        await self._inner.__aenter__()
        if stop_requested():
            await self._inner.__aexit__(None, None, None)
            raise _StopRequested
        return self

    async def __aexit__(self, *exc_info):
        return await self._inner.__aexit__(*exc_info)


def stop_between_pdfs(process_file):
    """Wrap PaperQA's per-PDF step so that no new PDF starts after a stop was asked for.

    Nothing is cancelled or killed: PDFs under way finish, and PaperQA's own closing save of the index runs.
    """
    signature = inspect.signature(process_file)

    async def stoppable_process_file(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.arguments["semaphore"] = _StopCheckingSemaphore(bound.arguments["semaphore"])
        try:
            return await process_file(*bound.args, **bound.kwargs)
        except _StopRequested:
            stop_state["skipped"] += 1

    return stoppable_process_file


def is_write_error(error: BaseException) -> bool:
    """Say whether an error is tantivy failing to write one of the index's files, which a new writer can retry."""
    return isinstance(error, ValueError) and any(marker in str(error) for marker in WRITE_ERROR_MARKERS)


def commit_landed(meta_before: str | None, meta_after: str | None) -> bool:
    """Say whether a commit that failed had already written the index's record of its contents, meta.json.

    Its PDFs are then in the index, and adding them again would put them there twice. When either reading
    failed, the answer is no: twice in the index is better than not at all.
    """
    return meta_before is not None and meta_after is not None and meta_before != meta_after


def lock_capped_index(search_index_class):
    """Return PaperQA's index class, changed so that a rebuild cannot wait hours for the writer lock, and so
    that a write that fails with "Access is denied" is tried again.

    PaperQA tries for the index folder's writer lock up to 1000 times, waiting up to a minute between tries.
    On Windows a write that fails with "Access is denied" leaves its writer holding that lock, so the next
    build in the same process waited for it for hours. The class tries LOCK_ATTEMPTS times instead, and
    remembers each index it opens in `opened`, so the rebuild can let go of their writers afterwards.

    A writer whose write failed has dropped every PDF added since the last save, although PaperQA's file list
    already counts them as indexed. So the class keeps those changes in `unsaved`, and after a failed write lets
    go of the writer, gives them to a new one, and tries the write again, WRITE_ATTEMPTS times in all. A commit
    can also fail after it was written, so they are given again only when meta.json has not changed.
    """
    save_index = search_index_class.save_index.retry_with(stop=stop_after_attempt(LOCK_ATTEMPTS))
    delete_document = search_index_class.delete_document.retry_with(stop=stop_after_attempt(LOCK_ATTEMPTS))

    class LockCappedSearchIndex(search_index_class):
        opened = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.opened.append(self)
            # ("add", PaperQA's fields for a PDF) or ("delete", file location), in order, since the last save.
            self.unsaved = []

        async def read_meta(self) -> str | None:
            """Return the text of the index's meta.json, or None when there is none yet or it cannot be read."""
            try:
                return Path(await self.index_filename, "meta.json").read_text(encoding="utf-8")
            except OSError:
                return None

        async def write_unsaved_again(self, attempt: int, meta_before: str | None) -> None:
            """Let go of the failed writer, then give a new writer every change since the last save, unless the
            failed write was a commit that had already been written."""
            release_writer(self)
            await asyncio.sleep(WRITE_RETRY_PAUSE_SECONDS * attempt)
            if commit_landed(meta_before, await self.read_meta()):
                self.unsaved.clear()
            async with self.writer() as writer:
                for kind, item in self.unsaved:
                    if kind == "add":
                        writer.add_document(Document.from_dict(item))
                    else:
                        writer.delete_documents("file_location", item)

        async def try_write(self, write) -> None:
            """Run a write to the index, and try it again with a new writer when tantivy cannot write a file."""
            meta_before = await self.read_meta()
            for attempt in range(1, WRITE_ATTEMPTS + 1):
                try:
                    if attempt > 1:
                        await self.write_unsaved_again(attempt - 1, meta_before)
                    await write()
                    return
                except Exception as error:
                    if attempt == WRITE_ATTEMPTS or not is_write_error(error):
                        raise
                    print(f"Writing the index failed; trying again (try {attempt + 1} of {WRITE_ATTEMPTS}): {error}")

        async def add_document(self, index_doc, document=None, lock_acquisition_max_retries=LOCK_ATTEMPTS):
            if await self.filecheck(index_doc["file_location"], self.filehash(index_doc["body"])):
                # PaperQA adds nothing for a PDF the index already holds unchanged.
                return
            tries = min(lock_acquisition_max_retries, LOCK_ATTEMPTS)
            add = super().add_document
            await self.try_write(lambda: add(index_doc, document, lock_acquisition_max_retries=tries))
            self.unsaved.append(("add", index_doc))

        async def save_index(self):
            await self.try_write(lambda: save_index(self))
            self.unsaved.clear()

        async def delete_document(self, file_location):
            # PaperQA's removal saves the index straight away, through save_index above.
            self.unsaved.append(("delete", file_location))
            await delete_document(self, file_location)

    return LockCappedSearchIndex


def summarize_exception_messages(error: BaseException) -> list[str]:
    """Flatten nested exception groups into a stable list of leaf messages."""
    if isinstance(error, BaseExceptionGroup):
        messages = []
        for child in error.exceptions:
            messages.extend(summarize_exception_messages(child))
        return messages
    return [str(error)]


def get_active_index_dir() -> Path:
    """Return the active PaperQA index directory for the current settings hash."""
    return Path(settings.agent.index.index_directory) / settings.get_index_name()


def clear_failed_documents_from_active_index(index_dir: Path) -> int:
    """Remove stale failed-document markers so rebuilds can retry those PDFs."""
    file_index = index_dir / "files.zip"
    if not file_index.exists():
        return 0
    try:
        index_files = pickle.loads(zlib.decompress(file_index.read_bytes()))
    except Exception:
        return 0

    remaining_index_files = {
        file_location: status
        for file_location, status in index_files.items()
        if status != FAILED_DOCUMENT_ADD_ID
    }
    removed_count = len(index_files) - len(remaining_index_files)
    if removed_count:
        file_index.write_bytes(zlib.compress(pickle.dumps(remaining_index_files)))
    return removed_count


async def async_main():
    """Rebuild the PaperQA directory index and print a short before/after status summary."""
    active_index_dir = get_active_index_dir()
    allowed_paths = get_allowed_paths()
    if allowed_paths:
        print(f"Manifest PDFs: {len(allowed_paths)}")
    else:
        print("Manifest PDFs: no paper list in use; indexing all PDFs under PAPER_DIR")
    print(f"Indexed before run: {get_indexed_doc_count(index_dir=active_index_dir)}")
    cleared_failed_documents = clear_failed_documents_from_active_index(active_index_dir)
    if cleared_failed_documents:
        print(f"Retrying {cleared_failed_documents} previously failed PDFs in the active index.")
    rebuild_error = None
    stop_state["skipped"] = 0
    run_state["failed"] = False
    paperqa_search = importlib.import_module("paperqa.agents.search")
    original_process_file = paperqa_search.process_file
    original_search_index = paperqa_search.SearchIndex
    paperqa_search.process_file = stop_between_pdfs(original_process_file)
    paperqa_search.SearchIndex = capped_search_index = lock_capped_index(original_search_index)
    try:
        await get_directory_index(settings=settings, build=True)
    except* Exception as exc_group:
        rebuild_error = exc_group
    finally:
        paperqa_search.process_file = original_process_file
        paperqa_search.SearchIndex = original_search_index
        # A writer left by a failed write keeps the folder's writer lock until the process ends.
        for search_index in capped_search_index.opened:
            release_writer(search_index)
    if stop_state["skipped"]:
        print(
            f"Stopped at your request: {stop_state['skipped']} PDFs were not started. "
            "PDFs already started were finished and saved."
        )
    failed_files = get_failed_files(index_dir=active_index_dir)
    if rebuild_error and failed_files:
        print("Rebuild stopped with parser/indexing errors; recorded failed PDFs will be reported below.")
    elif rebuild_error:
        print("Rebuild stopped with parser/indexing errors before any failed PDFs were recorded.")
        for message in summarize_exception_messages(rebuild_error):
            print(f"ERROR: {message}")
    indexed_after = get_indexed_doc_count(index_dir=active_index_dir)
    if indexed_after:
        # The app compares later settings changes with this index.
        index_record.remember(settings, active_index_dir)
    print(f"Indexed after run: {indexed_after}")
    print(f"Failed PDFs: {len(failed_files)}")
    for file_location in failed_files:
        print(f"FAILED: {file_location}")
    if rebuild_error:
        # PaperQA's build stops every PDF still to do once one of them raises, so those are not indexed.
        run_state["failed"] = True
        left = len(get_files_to_index(active_index_dir))
        print(f"The rebuild did not finish: {left} PDFs are not indexed yet. Run it again to carry on.")


def cost_kind(response) -> str:
    """Say which kind of call a response's cost belongs to: embeddings, enrichment, or reading a citation.

    Apart from embeddings and enrichment, the only calls a rebuild makes are the two that read the citation of
    a PDF without a row in the paper list.
    """
    if isinstance(response, litellm.EmbeddingResponse):
        return "embeddings"
    return "enrichment" if enriching.get() else "citations"


class CostCounter:
    """Add up the cost PaperQA's model client tracks for each call, by kind of call.

    The tracker adds each call's cost to its running total, then calls back with the response, so the cost of
    the call is what the total has grown by since the last call back.
    """

    def __init__(self, tracker=GLOBAL_COST_TRACKER):
        self.tracker = tracker
        self.seen = tracker.lifetime_cost_usd
        self.costs = rebuild_client.empty_costs()

    async def __call__(self, response) -> None:
        cost = self.tracker.lifetime_cost_usd - self.seen
        self.seen = self.tracker.lifetime_cost_usd
        self.costs[cost_kind(response)] += cost


def mark_enrichment(make_media_enricher):
    """Wrap PaperQA's maker of picture describers, so the calls each describer makes are marked as enrichment."""

    def make(settings):
        describe = make_media_enricher(settings)

        async def describe_and_mark(parsed_text):
            token = enriching.set(True)
            try:
                return await describe(parsed_text)
            finally:
                enriching.reset(token)

        return describe_and_mark

    return make


def report_costs(costs: dict[str, float]) -> None:
    """Print what the rebuild cost, and write it where the app asked for it, when it did."""
    print(rebuild_client.describe_cost(costs))
    cost_file = os.environ.get(rebuild_client.COST_ENV)
    if cost_file:
        Path(cost_file).write_text(json.dumps(costs), encoding="utf-8")


async def rebuild_counting_costs() -> None:
    """Rebuild the index and report what it really cost, through LiteLLM's prices, even if it fails part way."""
    counter = CostCounter()
    GLOBAL_COST_TRACKER.add_callback(counter)
    # Off by default; the calls PaperQA makes from here on are counted.
    enable_cost_tracking()
    original_enricher = Settings.make_media_enricher
    Settings.make_media_enricher = mark_enrichment(original_enricher)
    try:
        await async_main()
    finally:
        Settings.make_media_enricher = original_enricher
        GLOBAL_COST_TRACKER._callbacks.remove(counter)
        report_costs(counter.costs)


def print_estimate() -> None:
    """Print the rough cost and time of indexing the PDFs that are not in the index yet. Makes no paid calls."""
    files = get_files_to_index(get_active_index_dir())
    result = rebuild_estimate.estimate_files(settings, PAPER_DIR, files, local=bool(local_llm.load_config()))
    print(ESTIMATE_HEADING)
    for line in rebuild_estimate.describe(result):
        print(f"- {line}")


def main() -> int:
    """Run the async rebuild workflow from console-script entry points, with the estimate before it and the cost after."""
    print_estimate()
    asyncio.run(rebuild_counting_costs())
    if run_state["failed"]:
        return rebuild_client.FAILED_EXIT_CODE
    return rebuild_client.STOPPED_EXIT_CODE if stop_state["skipped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
