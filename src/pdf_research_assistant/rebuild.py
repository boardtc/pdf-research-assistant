import asyncio
import importlib
import inspect
import os
import pickle
import zlib
from pathlib import Path

from paperqa.agents.search import get_directory_index

from pdf_research_assistant import index_record, rebuild_client
from pdf_research_assistant.bootstrap import (
    ALLOWED_PATHS,
    FAILED_DOCUMENT_ADD_ID,
    USE_MANIFEST,
    build_settings,
    get_failed_files,
    get_indexed_doc_count,
)

settings = build_settings()

# How many PDFs were left unstarted because the app asked this rebuild to stop.
stop_state = {"skipped": 0}


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
    if USE_MANIFEST:
        print(f"Manifest PDFs: {len(ALLOWED_PATHS)}")
    else:
        print("Manifest PDFs: manifest.csv not found; indexing all PDFs under PAPER_DIR")
    print(f"Indexed before run: {get_indexed_doc_count(index_dir=active_index_dir)}")
    cleared_failed_documents = clear_failed_documents_from_active_index(active_index_dir)
    if cleared_failed_documents:
        print(f"Retrying {cleared_failed_documents} previously failed PDFs in the active index.")
    rebuild_error = None
    stop_state["skipped"] = 0
    paperqa_search = importlib.import_module("paperqa.agents.search")
    original_process_file = paperqa_search.process_file
    paperqa_search.process_file = stop_between_pdfs(original_process_file)
    try:
        await get_directory_index(settings=settings, build=True)
    except* Exception as exc_group:
        rebuild_error = exc_group
    finally:
        paperqa_search.process_file = original_process_file
    if stop_state["skipped"]:
        print(
            f"Stopped at your request: {stop_state['skipped']} PDFs were not started. "
            "PDFs already started were finished and saved."
        )
    failed_files = get_failed_files(index_dir=active_index_dir)
    if rebuild_error and failed_files:
        print("Rebuild completed with parser/indexing errors; recorded failed PDFs will be reported below.")
    elif rebuild_error:
        print("Rebuild completed with parser/indexing errors before any failed PDFs were recorded.")
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


def main() -> int:
    """Run the async rebuild workflow from console-script entry points."""
    asyncio.run(async_main())
    return rebuild_client.STOPPED_EXIT_CODE if stop_state["skipped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
