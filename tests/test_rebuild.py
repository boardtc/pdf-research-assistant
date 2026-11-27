import asyncio
import contextlib
import importlib
import json
import shutil
import pickle
import sys
import uuid
import zlib
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _write_files_zip(index_dir: Path, payload: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "files.zip").write_bytes(zlib.compress(pickle.dumps(payload)))


@pytest.fixture
def workspace_tmp_path():
    root = PROJECT_ROOT / "tests" / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(uuid.uuid4())
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def rebuild_index_module():
    from paperqa.agents.search import SearchIndex

    bootstrap_module = ModuleType("pdf_research_assistant.bootstrap")
    search_module = ModuleType("paperqa.agents.search")
    settings_object = mock.Mock()
    settings_object.agent.index.index_directory = Path("index-root")
    settings_object.get_index_name.return_value = "active-index"
    settings_object.parsing.multimodal = False

    bootstrap_module.get_allowed_paths = mock.Mock(return_value={"paper_a.pdf", "paper_b.pdf"})
    bootstrap_module.FAILED_DOCUMENT_ADD_ID = "ERROR"
    bootstrap_module.build_settings = mock.Mock(return_value=settings_object)
    bootstrap_module.get_failed_files = mock.Mock(return_value=[])
    bootstrap_module.get_indexed_doc_count = mock.Mock(side_effect=[1, 2])
    bootstrap_module.get_files_to_index = mock.Mock(return_value=[])
    bootstrap_module.load_allowed_manifest_paths = mock.Mock(return_value=set())
    bootstrap_module.PAPER_DIR = Path("papers")

    search_module.get_directory_index = mock.AsyncMock(name="get_directory_index")
    search_module.process_file = mock.AsyncMock(name="process_file")
    search_module.SearchIndex = SearchIndex

    with mock.patch.dict(
        sys.modules,
        {
            "pdf_research_assistant.bootstrap": bootstrap_module,
            "paperqa.agents.search": search_module,
        },
    ):
        sys.modules.pop("pdf_research_assistant.rebuild", None)
        module = importlib.import_module("pdf_research_assistant.rebuild")
        # Never write a record of the index during tests.
        with mock.patch.object(module, "index_record"):
            yield module
        sys.modules.pop("pdf_research_assistant.rebuild", None)


def test_main_prints_manifest_summary_and_rebuild_counts_when_manifest_is_enabled(rebuild_index_module):
    rebuild_index_module.get_allowed_paths = mock.Mock(return_value={"paper_a.pdf", "paper_b.pdf"})
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[1, 2])
    rebuild_index_module.get_failed_files = mock.Mock(return_value=[])

    # Mock print so the test can assert on the CLI status output.
    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    rebuild_index_module.get_directory_index.assert_awaited_once_with(
        settings=rebuild_index_module.settings,
        build=True,
    )
    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 2"),
        mock.call("Indexed before run: 1"),
        mock.call("Indexed after run: 2"),
        mock.call("Failed PDFs: 0"),
    ]


def test_main_prints_non_manifest_summary_when_no_paper_list_is_in_use(rebuild_index_module):
    rebuild_index_module.get_allowed_paths = mock.Mock(return_value=set())
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[0, 3])
    rebuild_index_module.get_failed_files = mock.Mock(return_value=[])

    # Mock print so the test can assert on the non-manifest branch output.
    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: no paper list in use; indexing all PDFs under PAPER_DIR"),
        mock.call("Indexed before run: 0"),
        mock.call("Indexed after run: 3"),
        mock.call("Failed PDFs: 0"),
    ]


def test_main_prints_each_failed_file_after_rebuild(rebuild_index_module):
    rebuild_index_module.get_allowed_paths = mock.Mock(return_value={"paper_a.pdf"})
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[1, 1])
    rebuild_index_module.get_failed_files = mock.Mock(return_value=["bad_a.pdf", "bad_b.pdf"])

    # Mock print so the test can assert that each failed file is reported individually.
    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 1"),
        mock.call("Indexed before run: 1"),
        mock.call("Indexed after run: 1"),
        mock.call("Failed PDFs: 2"),
        mock.call("FAILED: bad_a.pdf"),
        mock.call("FAILED: bad_b.pdf"),
    ]


def test_main_retries_previously_failed_documents_from_the_active_index_before_rebuild(rebuild_index_module):
    active_index_dir = Path("index-root") / "active-index"
    rebuild_index_module.settings = mock.Mock()
    rebuild_index_module.settings.agent.index.index_directory = Path("index-root")
    rebuild_index_module.settings.get_index_name.return_value = "active-index"
    rebuild_index_module.get_allowed_paths = mock.Mock(return_value={"paper_a.pdf"})
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[5, 6])
    rebuild_index_module.get_failed_files = mock.Mock(return_value=[])

    with mock.patch.object(
        rebuild_index_module,
        "clear_failed_documents_from_active_index",
        return_value=1,
    ) as clear_failed_mock:
        with mock.patch("builtins.print") as print_mock:
            asyncio.run(rebuild_index_module.async_main())

    clear_failed_mock.assert_called_once_with(active_index_dir)
    rebuild_index_module.get_indexed_doc_count.assert_has_calls(
        [mock.call(index_dir=active_index_dir), mock.call(index_dir=active_index_dir)]
    )
    rebuild_index_module.get_failed_files.assert_called_once_with(index_dir=active_index_dir)
    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 1"),
        mock.call("Indexed before run: 5"),
        mock.call("Retrying 1 previously failed PDFs in the active index."),
        mock.call("Indexed after run: 6"),
        mock.call("Failed PDFs: 0"),
    ]


def test_main_continues_when_directory_rebuild_raises_exception_group_and_reports_recorded_failed_files(
    rebuild_index_module,
):
    rebuild_index_module.get_allowed_paths = mock.Mock(return_value={"paper_a.pdf"})
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[105, 105])
    rebuild_index_module.get_failed_files = mock.Mock(
        return_value=["Session 7/Sadler 2013 Opening-up-feedback-Teaching-learners-to-see Chapter.pdf"]
    )
    rebuild_index_module.get_directory_index = mock.AsyncMock(
        side_effect=ExceptionGroup("unhandled errors in a TaskGroup", [LookupError("unknown encoding: /SymbolSetEncoding")])
    )
    rebuild_index_module.get_files_to_index = mock.Mock(
        return_value=["Session 7/Sadler 2013 Opening-up-feedback-Teaching-learners-to-see Chapter.pdf"]
    )

    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 1"),
        mock.call("Indexed before run: 105"),
        mock.call("Rebuild stopped with parser/indexing errors; recorded failed PDFs will be reported below."),
        mock.call("Indexed after run: 105"),
        mock.call("Failed PDFs: 1"),
        mock.call("FAILED: Session 7/Sadler 2013 Opening-up-feedback-Teaching-learners-to-see Chapter.pdf"),
        mock.call("The rebuild did not finish: 1 PDFs are not indexed yet. Run it again to carry on."),
    ]


def test_main_prints_exception_messages_when_directory_rebuild_raises_exception_group_without_recorded_failed_files(
    rebuild_index_module,
):
    rebuild_index_module.get_allowed_paths = mock.Mock(return_value={"paper_a.pdf"})
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[3, 3])
    rebuild_index_module.get_failed_files = mock.Mock(return_value=[])
    rebuild_index_module.get_directory_index = mock.AsyncMock(
        side_effect=ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [
                LookupError("unknown encoding: /SymbolSetEncoding"),
                ValueError("bad metadata"),
            ],
        )
    )
    rebuild_index_module.get_files_to_index = mock.Mock(return_value=["paper_a.pdf"])

    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 1"),
        mock.call("Indexed before run: 3"),
        mock.call("Rebuild stopped with parser/indexing errors before any failed PDFs were recorded."),
        mock.call("ERROR: unknown encoding: /SymbolSetEncoding"),
        mock.call("ERROR: bad metadata"),
        mock.call("Indexed after run: 3"),
        mock.call("Failed PDFs: 0"),
        mock.call("The rebuild did not finish: 1 PDFs are not indexed yet. Run it again to carry on."),
    ]


def test_main_records_the_rebuilt_index_so_the_app_stops_warning_about_changed_settings(rebuild_index_module):
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[0, 2])

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    rebuild_index_module.index_record.remember.assert_called_once_with(
        rebuild_index_module.settings, Path("index-root") / "active-index"
    )


def test_main_records_nothing_when_the_rebuild_left_the_index_empty(rebuild_index_module):
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[0, 0])

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    rebuild_index_module.index_record.remember.assert_not_called()


def test_summarize_exception_messages_returns_single_message_for_non_group_exception(rebuild_index_module):
    assert rebuild_index_module.summarize_exception_messages(ValueError("bad metadata")) == ["bad metadata"]


def test_main_runs_async_main_and_returns_zero(rebuild_index_module):
    with mock.patch.object(rebuild_index_module.asyncio, "run") as asyncio_run_mock:
        assert rebuild_index_module.main() == 0

    asyncio_run_mock.assert_called_once()
    asyncio_run_mock.call_args.args[0].close()


def test_summarize_exception_messages_flattens_nested_exception_groups_in_order(rebuild_index_module):
    nested_group = ExceptionGroup(
        "outer",
        [
            LookupError("unknown encoding: /SymbolSetEncoding"),
            ExceptionGroup("inner", [ValueError("bad metadata")]),
        ],
    )

    assert rebuild_index_module.summarize_exception_messages(nested_group) == [
        "unknown encoding: /SymbolSetEncoding",
        "bad metadata",
    ]


def test_get_active_index_dir_returns_current_settings_index_directory(rebuild_index_module):
    rebuild_index_module.settings = mock.Mock()
    rebuild_index_module.settings.agent.index.index_directory = Path("index-root")
    rebuild_index_module.settings.get_index_name.return_value = "active-index"

    assert rebuild_index_module.get_active_index_dir() == Path("index-root") / "active-index"


def test_clear_failed_documents_from_active_index_returns_zero_when_files_zip_is_missing(
    rebuild_index_module, workspace_tmp_path
):
    assert rebuild_index_module.clear_failed_documents_from_active_index(workspace_tmp_path / "missing") == 0


def test_clear_failed_documents_from_active_index_returns_zero_when_files_zip_is_unreadable(
    rebuild_index_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "files.zip").write_bytes(b"not-a-valid-archive")

    assert rebuild_index_module.clear_failed_documents_from_active_index(index_dir) == 0


def test_clear_failed_documents_from_active_index_removes_only_failed_entries_and_rewrites_files_zip(
    rebuild_index_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(
        index_dir,
        {
            "failed.pdf": rebuild_index_module.FAILED_DOCUMENT_ADD_ID,
            "kept.pdf": "abc123",
        },
    )

    removed_count = rebuild_index_module.clear_failed_documents_from_active_index(index_dir)
    rewritten_payload = pickle.loads(zlib.decompress((index_dir / "files.zip").read_bytes()))

    assert removed_count == 1
    assert rewritten_payload == {"kept.pdf": "abc123"}


# Stopping between PDFs


def _stop_file(workspace_tmp_path, monkeypatch, exists):
    stop = workspace_tmp_path / "rebuild.stop"
    if exists:
        stop.write_text("stop", encoding="utf-8")
    monkeypatch.setenv("PDF_RESEARCH_STOP_FILE", str(stop))
    return stop


def test_stop_requested_is_false_when_the_app_gave_no_stop_file(rebuild_index_module, monkeypatch):
    monkeypatch.delenv("PDF_RESEARCH_STOP_FILE", raising=False)

    assert rebuild_index_module.stop_requested() is False


def test_stop_requested_is_false_until_the_stop_file_appears(rebuild_index_module, workspace_tmp_path, monkeypatch):
    stop = _stop_file(workspace_tmp_path, monkeypatch, exists=False)

    assert rebuild_index_module.stop_requested() is False
    stop.write_text("stop", encoding="utf-8")
    assert rebuild_index_module.stop_requested() is True


def test_the_stop_check_names_the_same_variable_and_exit_code_as_the_app_side(rebuild_index_module):
    from pdf_research_assistant import rebuild_client

    assert rebuild_index_module.rebuild_client.STOP_ENV == rebuild_client.STOP_ENV == "PDF_RESEARCH_STOP_FILE"
    assert rebuild_index_module.rebuild_client.STOPPED_EXIT_CODE == 3


async def _index_one(semaphore, ran):
    async with semaphore:
        ran.append(1)
        return "indexed"


def test_a_pdf_is_indexed_as_usual_when_no_stop_was_asked_for(rebuild_index_module, workspace_tmp_path, monkeypatch):
    import anyio

    _stop_file(workspace_tmp_path, monkeypatch, exists=False)
    ran = []

    async def original(path, semaphore):
        return await _index_one(semaphore, ran)

    wrapped = rebuild_index_module.stop_between_pdfs(original)

    assert asyncio.run(wrapped("a.pdf", anyio.Semaphore(1))) == "indexed"
    assert ran == [1]
    assert rebuild_index_module.stop_state["skipped"] == 0


def test_a_pdf_that_has_not_started_is_skipped_once_a_stop_was_asked_for(
    rebuild_index_module, workspace_tmp_path, monkeypatch
):
    import anyio

    _stop_file(workspace_tmp_path, monkeypatch, exists=True)
    ran = []
    semaphore = anyio.Semaphore(1)

    async def original(path, semaphore):
        return await _index_one(semaphore, ran)

    wrapped = rebuild_index_module.stop_between_pdfs(original)

    async def run_twice():
        # The second call would wait for ever if the first kept its place in the semaphore.
        return [await wrapped("a.pdf", semaphore), await wrapped("b.pdf", semaphore)]

    assert asyncio.run(asyncio.wait_for(run_twice(), 5)) == [None, None]
    assert ran == []
    assert rebuild_index_module.stop_state["skipped"] == 2


def test_a_pdf_already_being_indexed_is_finished_when_a_stop_is_asked_for(
    rebuild_index_module, workspace_tmp_path, monkeypatch
):
    import anyio

    stop = _stop_file(workspace_tmp_path, monkeypatch, exists=False)
    finished = []

    async def original(path, semaphore):
        async with semaphore:
            stop.write_text("stop", encoding="utf-8")
            finished.append(path)
            return "indexed"

    wrapped = rebuild_index_module.stop_between_pdfs(original)

    assert asyncio.run(wrapped("a.pdf", anyio.Semaphore(1))) == "indexed"
    assert finished == ["a.pdf"]
    assert rebuild_index_module.stop_state["skipped"] == 0


def test_the_wrapper_finds_the_semaphore_by_name_even_when_it_is_passed_as_a_keyword(
    rebuild_index_module, workspace_tmp_path, monkeypatch
):
    import anyio

    _stop_file(workspace_tmp_path, monkeypatch, exists=True)

    async def original(path, semaphore):
        return await _index_one(semaphore, [])

    wrapped = rebuild_index_module.stop_between_pdfs(original)

    assert asyncio.run(wrapped("a.pdf", semaphore=anyio.Semaphore(1))) is None


def test_paperqas_process_file_still_takes_the_semaphore_the_stop_check_relies_on():
    import inspect

    from paperqa.agents import search

    assert "semaphore" in inspect.signature(search.process_file).parameters
    assert "process_file" in inspect.getsource(search.get_directory_index)


def test_main_swaps_in_the_stop_check_only_while_the_directory_index_is_built(rebuild_index_module):
    search_module = sys.modules["paperqa.agents.search"]
    original = search_module.process_file
    seen = []

    async def build(settings, build):
        seen.append(search_module.process_file)

    rebuild_index_module.get_directory_index = mock.AsyncMock(side_effect=build)

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    assert seen[0] is not original
    assert search_module.process_file is original


def test_main_puts_the_original_process_file_back_when_the_build_fails(rebuild_index_module):
    search_module = sys.modules["paperqa.agents.search"]
    original = search_module.process_file
    rebuild_index_module.get_directory_index = mock.AsyncMock(side_effect=ExceptionGroup("x", [ValueError("bad")]))

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    assert search_module.process_file is original


def test_main_says_how_many_pdfs_were_not_started_after_a_stop(rebuild_index_module):
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[1, 2])

    async def build(settings, build):
        rebuild_index_module.stop_state["skipped"] = 4

    rebuild_index_module.get_directory_index = mock.AsyncMock(side_effect=build)

    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    assert mock.call(
        "Stopped at your request: 4 PDFs were not started. PDFs already started were finished and saved."
    ) in print_mock.call_args_list


def test_main_exits_with_the_stopped_code_when_pdfs_were_skipped(rebuild_index_module):
    async def stopped_run():
        rebuild_index_module.stop_state["skipped"] = 2

    with mock.patch.object(rebuild_index_module, "async_main", side_effect=stopped_run):
        assert rebuild_index_module.main() == 3


# A rebuild must not wait for hours on the index folder's writer lock (#134).


def test_main_swaps_in_the_lock_capped_index_only_while_the_directory_index_is_built(rebuild_index_module):
    search_module = sys.modules["paperqa.agents.search"]
    original = search_module.SearchIndex
    seen = []

    async def build(settings, build):
        seen.append(search_module.SearchIndex)

    rebuild_index_module.get_directory_index = mock.AsyncMock(side_effect=build)

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    assert seen[0] is not original
    assert issubclass(seen[0], original)
    assert search_module.SearchIndex is original


def test_main_puts_the_original_index_class_back_when_the_build_fails(rebuild_index_module):
    search_module = sys.modules["paperqa.agents.search"]
    original = search_module.SearchIndex
    rebuild_index_module.get_directory_index = mock.AsyncMock(side_effect=ExceptionGroup("x", [ValueError("bad")]))

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    assert search_module.SearchIndex is original


def _build_that_leaves_a_writer(rebuild_index_module, writers, error=None):
    """Return a stand-in for PaperQA's build that opens an index, gives it a writer, and maybe fails."""
    search_module = sys.modules["paperqa.agents.search"]

    async def build(settings, build):
        index = search_module.SearchIndex(index_directory="unused")
        index._writer = mock.Mock(name="writer")
        writers.append((index, index._writer))
        if error:
            raise error

    return build


def test_main_lets_go_of_the_writer_a_failed_build_left_holding_the_lock(rebuild_index_module):
    writers = []
    rebuild_index_module.get_directory_index = mock.AsyncMock(
        side_effect=_build_that_leaves_a_writer(
            rebuild_index_module, writers, ExceptionGroup("x", [PermissionError(13, "Access is denied")])
        )
    )

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    index, writer = writers[0]
    writer.wait_merging_threads.assert_called_once_with()
    assert index._writer is None


def test_main_lets_go_of_any_writer_left_after_a_build_that_finished(rebuild_index_module):
    writers = []
    rebuild_index_module.get_directory_index = mock.AsyncMock(
        side_effect=_build_that_leaves_a_writer(rebuild_index_module, writers)
    )

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    index, writer = writers[0]
    writer.wait_merging_threads.assert_called_once_with()
    assert index._writer is None


def test_main_forgets_the_indexes_of_earlier_rebuilds(rebuild_index_module):
    rebuild_index_module.get_indexed_doc_count = mock.Mock(return_value=1)
    writers = []
    rebuild_index_module.get_directory_index = mock.AsyncMock(
        side_effect=_build_that_leaves_a_writer(rebuild_index_module, writers)
    )

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())
        first_index, _writer = writers[0]
        first_index._writer = later_writer = mock.Mock(name="later writer")
        asyncio.run(rebuild_index_module.async_main())

    later_writer.wait_merging_threads.assert_not_called()


@pytest.fixture
def locked_index_folder(workspace_tmp_path):
    """Yield a folder whose index writer lock another index holds, as a failed write leaves it."""
    from paperqa.agents.search import SearchIndex

    holder = SearchIndex(index_name="locked", index_directory=workspace_tmp_path)

    async def take_lock():
        async with holder.writer():
            pass

    asyncio.run(take_lock())
    yield workspace_tmp_path
    holder._writer.wait_merging_threads()


@pytest.fixture
def counted_lock_waits(monkeypatch):
    """Count the pauses between tries for the lock, and skip them."""
    real_sleep = asyncio.sleep
    pauses = []

    async def no_pause(seconds, *args, **kwargs):
        pauses.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_pause)
    return pauses


def _capped_index(rebuild_index_module, folder):
    from paperqa.agents.search import SearchIndex

    return rebuild_index_module.lock_capped_index(SearchIndex)(index_name="locked", index_directory=folder)


def test_adding_a_pdf_gives_up_on_a_held_lock_after_the_capped_tries(
    rebuild_index_module, locked_index_folder, counted_lock_waits
):
    from tenacity import RetryError

    index = _capped_index(rebuild_index_module, locked_index_folder)
    document = {"file_location": "a.pdf", "body": "text", "title": "A", "year": "2026"}

    with pytest.raises(RetryError):
        asyncio.run(index.add_document(document))

    assert len(counted_lock_waits) == rebuild_index_module.LOCK_ATTEMPTS - 1


def test_adding_a_pdf_never_tries_for_the_lock_more_than_the_cap(
    rebuild_index_module, locked_index_folder, counted_lock_waits
):
    from tenacity import RetryError

    index = _capped_index(rebuild_index_module, locked_index_folder)
    document = {"file_location": "a.pdf", "body": "text", "title": "A", "year": "2026"}

    with pytest.raises(RetryError):
        asyncio.run(index.add_document(document, lock_acquisition_max_retries=1000))

    assert len(counted_lock_waits) == rebuild_index_module.LOCK_ATTEMPTS - 1


def test_saving_the_index_gives_up_on_a_held_lock_after_the_capped_tries(
    rebuild_index_module, locked_index_folder, counted_lock_waits
):
    index = _capped_index(rebuild_index_module, locked_index_folder)

    with pytest.raises(Exception, match="Failed to acquire lock"):
        asyncio.run(index.save_index())

    assert len(counted_lock_waits) == rebuild_index_module.LOCK_ATTEMPTS - 1


def test_removing_a_pdf_gives_up_on_a_held_lock_after_the_capped_tries(
    rebuild_index_module, locked_index_folder, counted_lock_waits
):
    index = _capped_index(rebuild_index_module, locked_index_folder)

    with pytest.raises(Exception, match="Failed to acquire lock"):
        asyncio.run(index.delete_document("a.pdf"))

    assert len(counted_lock_waits) == rebuild_index_module.LOCK_ATTEMPTS - 1


def test_the_lock_cap_is_the_one_saved_answers_uses(rebuild_index_module):
    from pdf_research_assistant import saved_answers

    assert rebuild_index_module.LOCK_ATTEMPTS == saved_answers.LOCK_ATTEMPTS
    assert rebuild_index_module.release_writer is saved_answers.release_writer


# On Windows, tantivy now and then fails to create a new file of the index with "Access is denied" (#157).

WRITE_ERROR = (
    "Failed to open file for write: 'IoError { io_error: Os { code: 5, kind: PermissionDenied, message: "
    '"Access is denied." }, filepath: "89fae2c035d04722bf5746835c1243dd.fast" }\''
)


@pytest.mark.parametrize(
    "message",
    [
        WRITE_ERROR,
        "An index writer was killed.",
        "The index writer was killed. It can happen if an indexing worker encountered an Io error for instance.",
        "A worker thread encountered an error (io::Error most likely) or panicked.",
        "An IO error occurred: 'Access is denied. (os error 5)'",
    ],
)
def test_a_failed_write_of_an_index_file_counts_as_a_write_error(rebuild_index_module, message):
    assert rebuild_index_module.is_write_error(ValueError(message)) is True


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Failed to acquire Lockfile: LockBusy."),
        ValueError("bad metadata"),
        PermissionError(13, "Access is denied"),
    ],
)
def test_other_errors_do_not_count_as_write_errors(rebuild_index_module, error):
    assert rebuild_index_module.is_write_error(error) is False


class _FailingWriter:
    """Stands in for a tantivy writer: fails the named step `times` times, otherwise passes it to the real one."""

    def __init__(self, real, step, times=1, message=WRITE_ERROR, after=False):
        self.real, self.step, self.left, self.message, self.after = real, step, times, message, after

    def __getattr__(self, name):
        attribute = getattr(self.real, name)
        if name != self.step:
            return attribute

        def call(*args, **kwargs):
            if self.left:
                self.left -= 1
                if self.after:
                    # The step's work is done, and only then does it fail.
                    attribute(*args, **kwargs)
                raise ValueError(self.message)
            return attribute(*args, **kwargs)

        return call


@pytest.fixture
def no_write_pauses(monkeypatch):
    """Count the pauses before each new try at a write, and skip them."""
    real_sleep = asyncio.sleep
    pauses = []

    async def no_pause(seconds, *args, **kwargs):
        pauses.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_pause)
    return pauses


def _doc(name, body=None):
    return {"file_location": name, "body": body or f"text of {name}", "title": name, "year": "2026"}


def _real_capped_index(rebuild_index_module, folder):
    from paperqa.agents.search import SearchIndex

    capped = rebuild_index_module.lock_capped_index(SearchIndex)
    capped.opened.clear()
    return capped(index_name="written", index_directory=folder, fields=["file_location", "body", "title", "year"])


def _indexed_names(index) -> list[str]:
    async def read():
        tantivy_index = await index.index
        tantivy_index.reload()
        searcher = tantivy_index.searcher()
        return sorted(
            searcher.doc(address)["file_location"][0]
            for _score, address in searcher.search(tantivy_index.parse_query("*"), 100).hits
        )

    return asyncio.run(read())


def _close(index):
    """Let go of the index's writer, so the test folder can be deleted."""
    from pdf_research_assistant.saved_answers import release_writer

    # A writer whose thread really failed raises here too, so this goes through the forgiving release.
    release_writer(index)


def test_a_save_that_fails_to_write_is_tried_again_with_every_unsaved_pdf(
    rebuild_index_module, workspace_tmp_path, no_write_pauses
):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    async def run():
        await index.add_document(_doc("a.pdf"))
        await index.add_document(_doc("b.pdf"))
        index._writer = _FailingWriter(index._writer, "commit")
        with mock.patch("builtins.print") as print_mock:
            await index.save_index()
        return print_mock

    print_mock = asyncio.run(run())

    assert _indexed_names(index) == ["a.pdf", "b.pdf"]
    assert set(asyncio.run(index.index_files)) == {"a.pdf", "b.pdf"}
    assert index.unsaved == []
    # On Windows a real "Access is denied" can add a try of its own.
    assert no_write_pauses[0] == rebuild_index_module.WRITE_RETRY_PAUSE_SECONDS
    assert print_mock.call_args_list[0] == mock.call(
        f"Writing the index failed; trying again (try 2 of {rebuild_index_module.WRITE_ATTEMPTS}): {WRITE_ERROR}"
    )
    _close(index)


def test_a_save_whose_commit_landed_before_it_failed_does_not_add_the_pdfs_twice(
    rebuild_index_module, workspace_tmp_path, no_write_pauses
):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    async def run():
        await index.add_document(_doc("a.pdf"))
        await index.add_document(_doc("b.pdf"))
        index._writer = _FailingWriter(index._writer, "commit", after=True)
        with mock.patch("builtins.print"):
            await index.save_index()

    asyncio.run(run())

    assert _indexed_names(index) == ["a.pdf", "b.pdf"]
    assert set(asyncio.run(index.index_files)) == {"a.pdf", "b.pdf"}
    assert index.unsaved == []
    _close(index)


@pytest.mark.parametrize(
    ("before", "after", "landed"),
    [
        ('{"opstamp": 1}', '{"opstamp": 1}', False),
        ('{"opstamp": 1}', '{"opstamp": 3}', True),
        # When either could not be read, the PDFs are added again: twice in the index is better than not at all.
        (None, '{"opstamp": 3}', False),
        ('{"opstamp": 1}', None, False),
    ],
)
def test_a_commit_landed_only_when_the_index_record_changed(rebuild_index_module, before, after, landed):
    assert rebuild_index_module.commit_landed(before, after) is landed


def test_reading_the_index_record_gives_its_text(rebuild_index_module, workspace_tmp_path):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    async def run():
        await index.add_document(_doc("a.pdf"))
        await index.save_index()
        return await index.read_meta()

    meta = asyncio.run(run())

    assert meta == (workspace_tmp_path / "written" / "index" / "meta.json").read_text(encoding="utf-8")
    _close(index)


def test_reading_the_index_record_gives_none_before_there_is_one(rebuild_index_module, workspace_tmp_path):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    assert asyncio.run(index.read_meta()) is None


def test_reading_the_index_record_gives_none_when_it_cannot_be_read(rebuild_index_module, workspace_tmp_path):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    async def run():
        await index.add_document(_doc("a.pdf"))
        await index.save_index()
        with mock.patch.object(Path, "read_text", side_effect=PermissionError(13, "Access is denied")):
            return await index.read_meta()

    assert asyncio.run(run()) is None
    _close(index)


def test_a_save_that_keeps_failing_to_write_gives_up_after_the_capped_tries(
    rebuild_index_module, workspace_tmp_path, no_write_pauses
):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)
    real_writer = type(index).writer

    @contextlib.asynccontextmanager
    async def failing_writer(self, reset=False):
        async with real_writer(self, reset=False) as writer:
            if not isinstance(writer, _FailingWriter):
                self._writer = _FailingWriter(writer, "commit", times=99)
            yield self._writer
        if reset:
            self._writer = None

    async def run():
        await index.add_document(_doc("a.pdf"))
        with mock.patch.object(type(index), "writer", failing_writer), mock.patch("builtins.print"):
            await index.save_index()

    with pytest.raises(ValueError) as raised:
        asyncio.run(run())

    # On Windows the last try can also fail for real, in tantivy's own words.
    assert rebuild_index_module.is_write_error(raised.value)
    assert len(no_write_pauses) == rebuild_index_module.WRITE_ATTEMPTS - 1
    assert no_write_pauses == sorted(no_write_pauses)
    assert index.unsaved == [("add", _doc("a.pdf"))]
    _close(index)


def test_a_save_that_fails_for_another_reason_is_not_tried_again(rebuild_index_module, workspace_tmp_path, no_write_pauses):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    async def run():
        await index.add_document(_doc("a.pdf"))
        index._writer = _FailingWriter(index._writer, "commit", message="bad metadata")
        await index.save_index()

    with pytest.raises(ValueError, match="bad metadata"):
        asyncio.run(run())

    assert no_write_pauses == []
    _close(index)


def test_adding_a_pdf_after_the_writer_was_killed_starts_a_new_writer_with_the_unsaved_pdfs(
    rebuild_index_module, workspace_tmp_path, no_write_pauses
):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    async def run():
        await index.add_document(_doc("a.pdf"))
        index._writer = _FailingWriter(index._writer, "add_document", message="An index writer was killed.")
        with mock.patch("builtins.print"):
            await index.add_document(_doc("b.pdf"))
        await index.save_index()

    asyncio.run(run())

    assert _indexed_names(index) == ["a.pdf", "b.pdf"]
    assert no_write_pauses[0] == rebuild_index_module.WRITE_RETRY_PAUSE_SECONDS
    _close(index)


def test_a_pdf_already_in_the_index_is_not_added_or_remembered_again(rebuild_index_module, workspace_tmp_path):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    async def run():
        await index.add_document(_doc("a.pdf"))
        await index.add_document(_doc("a.pdf"))

    asyncio.run(run())

    assert index.unsaved == [("add", _doc("a.pdf"))]
    _close(index)


def test_a_removal_is_written_again_when_its_save_fails(rebuild_index_module, workspace_tmp_path, no_write_pauses):
    index = _real_capped_index(rebuild_index_module, workspace_tmp_path)

    # PaperQA removes by the whole file location as one word, which only a name of one word matches.
    async def run():
        await index.add_document(_doc("alpha"))
        await index.save_index()
        await index.add_document(_doc("beta"))
        index._writer = _FailingWriter(index._writer, "commit")
        with mock.patch("builtins.print"):
            await index.delete_document("alpha")

    asyncio.run(run())

    assert _indexed_names(index) == ["beta"]
    assert index.unsaved == []
    _close(index)


def test_a_new_index_starts_with_nothing_unsaved(rebuild_index_module, workspace_tmp_path):
    first = _real_capped_index(rebuild_index_module, workspace_tmp_path)
    asyncio.run(first.add_document(_doc("a.pdf")))
    _close(first)

    second = _real_capped_index(rebuild_index_module, workspace_tmp_path / "other")

    assert second.unsaved == []


# A rebuild that did not finish says so and how many PDFs are left, and exits with an error (#157).


def test_main_says_how_many_pdfs_are_left_when_the_build_fails(rebuild_index_module):
    active_index_dir = Path("index-root") / "active-index"
    rebuild_index_module.get_allowed_paths = mock.Mock(return_value={"a.pdf", "b.pdf", "c.pdf"})
    rebuild_index_module.get_indexed_doc_count = mock.Mock(side_effect=[0, 1])
    rebuild_index_module.get_files_to_index = mock.Mock(return_value=["b.pdf", "c.pdf"])
    rebuild_index_module.get_directory_index = mock.AsyncMock(side_effect=ExceptionGroup("x", [ValueError(WRITE_ERROR)]))

    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    rebuild_index_module.get_files_to_index.assert_called_once_with(active_index_dir)
    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 3"),
        mock.call("Indexed before run: 0"),
        mock.call("Rebuild stopped with parser/indexing errors before any failed PDFs were recorded."),
        mock.call(f"ERROR: {WRITE_ERROR}"),
        mock.call("Indexed after run: 1"),
        mock.call("Failed PDFs: 0"),
        mock.call("The rebuild did not finish: 2 PDFs are not indexed yet. Run it again to carry on."),
    ]
    assert rebuild_index_module.run_state["failed"] is True


def test_main_does_not_count_the_pdfs_left_after_a_build_that_finished(rebuild_index_module):
    rebuild_index_module.run_state["failed"] = True

    with mock.patch("builtins.print"):
        asyncio.run(rebuild_index_module.async_main())

    rebuild_index_module.get_files_to_index.assert_not_called()
    assert rebuild_index_module.run_state["failed"] is False


def test_main_exits_with_the_failed_code_when_the_build_did_not_finish(rebuild_index_module):
    async def failed_run():
        rebuild_index_module.run_state["failed"] = True

    with mock.patch.object(rebuild_index_module, "async_main", side_effect=failed_run):
        assert rebuild_index_module.main() == 4


def test_main_exits_with_the_failed_code_even_when_a_stop_was_asked_for_too(rebuild_index_module):
    async def failed_and_stopped_run():
        rebuild_index_module.run_state["failed"] = True
        rebuild_index_module.stop_state["skipped"] = 2

    with mock.patch.object(rebuild_index_module, "async_main", side_effect=failed_and_stopped_run):
        assert rebuild_index_module.main() == 4


def test_the_failed_exit_code_is_the_one_the_app_side_reads(rebuild_index_module):
    from pdf_research_assistant import rebuild_client

    assert rebuild_index_module.rebuild_client.FAILED_EXIT_CODE == rebuild_client.FAILED_EXIT_CODE == 4


# The rough cost before a rebuild and the real cost after it (#153).


def _embedding_response():
    import litellm

    return litellm.EmbeddingResponse()


def _completion_response():
    import litellm

    return litellm.ModelResponse()


def test_cost_kind_tells_embeddings_from_citation_reading_and_enrichment(rebuild_index_module):
    module = rebuild_index_module

    assert module.cost_kind(_embedding_response()) == "embeddings"
    assert module.cost_kind(_completion_response()) == "citations"
    token = module.enriching.set(True)
    try:
        assert module.cost_kind(_completion_response()) == "enrichment"
    finally:
        module.enriching.reset(token)


def test_cost_counter_adds_each_calls_cost_to_its_kind(rebuild_index_module):
    module = rebuild_index_module
    tracker = mock.Mock(lifetime_cost_usd=1.0)
    counter = module.CostCounter(tracker)

    async def calls():
        # The tracker adds a call's cost to its running total before it calls back.
        tracker.lifetime_cost_usd = 1.25
        await counter(_embedding_response())
        tracker.lifetime_cost_usd = 1.75
        await counter(_completion_response())
        tracker.lifetime_cost_usd = 1.80
        await counter(_embedding_response())

    asyncio.run(calls())

    assert counter.costs == pytest.approx({"embeddings": 0.30, "enrichment": 0.0, "citations": 0.50})


def test_mark_enrichment_marks_the_calls_made_while_describing_pictures(rebuild_index_module):
    module = rebuild_index_module
    seen = []

    async def describe_pictures(parsed_text):
        async def one_picture():
            seen.append(module.enriching.get())

        # PaperQA describes the pictures side by side, in tasks of their own.
        await asyncio.gather(one_picture(), one_picture())
        return f"described {parsed_text}"

    make = module.mark_enrichment(lambda settings: describe_pictures)

    async def run():
        result = await make("settings")("text")
        seen.append(module.enriching.get())
        return result

    assert asyncio.run(run()) == "described text"
    assert seen == [True, True, False]


def test_rebuild_counting_costs_reports_what_each_kind_of_call_cost(rebuild_index_module, workspace_tmp_path, monkeypatch):
    from lmi.cost_tracker import GLOBAL_COST_TRACKER
    from paperqa.settings import Settings

    module = rebuild_index_module
    cost_file = workspace_tmp_path / "rebuild.cost.json"
    monkeypatch.setenv(module.rebuild_client.COST_ENV, str(cost_file))
    original_enricher = Settings.make_media_enricher
    callbacks_before = list(GLOBAL_COST_TRACKER._callbacks)

    async def rebuild():
        assert GLOBAL_COST_TRACKER.enabled.get() is True
        assert Settings.make_media_enricher is not original_enricher
        GLOBAL_COST_TRACKER.lifetime_cost_usd += 0.002
        await GLOBAL_COST_TRACKER._callbacks[-1](_embedding_response())

    with mock.patch.object(module, "async_main", side_effect=rebuild):
        with mock.patch("builtins.print") as print_mock:
            asyncio.run(module.rebuild_counting_costs())

    costs = {"embeddings": pytest.approx(0.002), "enrichment": 0.0, "citations": 0.0}
    assert json.loads(cost_file.read_text(encoding="utf-8")) == costs
    print_mock.assert_called_with(module.rebuild_client.describe_cost(json.loads(cost_file.read_text(encoding="utf-8"))))
    assert Settings.make_media_enricher is original_enricher
    assert GLOBAL_COST_TRACKER._callbacks == callbacks_before


def test_rebuild_counting_costs_still_reports_the_cost_when_the_rebuild_fails(rebuild_index_module, monkeypatch):
    module = rebuild_index_module
    monkeypatch.delenv(module.rebuild_client.COST_ENV, raising=False)

    with mock.patch.object(module, "async_main", side_effect=RuntimeError("crash")):
        with mock.patch("builtins.print") as print_mock:
            with pytest.raises(RuntimeError):
                asyncio.run(module.rebuild_counting_costs())

    print_mock.assert_called_once_with(module.rebuild_client.describe_cost(module.rebuild_client.empty_costs()))


def test_print_estimate_prints_the_rough_cost_of_the_pdfs_still_to_index(rebuild_index_module):
    module = rebuild_index_module
    module.get_files_to_index = mock.Mock(return_value=["new.pdf"])
    result = mock.Mock()

    with mock.patch.object(module.rebuild_estimate, "estimate_files", return_value=result) as estimate_files:
        with mock.patch.object(module.rebuild_estimate, "describe", return_value=["line one", "line two"]):
            with mock.patch("builtins.print") as print_mock:
                module.print_estimate()

    module.get_files_to_index.assert_called_once_with(module.get_active_index_dir())
    # Which PDFs PaperQA must read a citation for comes from the paper list PaperQA itself is given.
    estimate_files.assert_called_once_with(module.settings, Path("papers"), ["new.pdf"], local=False)
    assert print_mock.call_args_list == [
        mock.call(module.ESTIMATE_HEADING),
        mock.call("- line one"),
        mock.call("- line two"),
    ]


def test_print_estimate_counts_a_saved_local_model_as_free(rebuild_index_module):
    module = rebuild_index_module

    with mock.patch.object(module.local_llm, "load_config", return_value={"model": "llama3"}):
        with mock.patch.object(module.rebuild_estimate, "estimate_files") as estimate_files:
            with mock.patch.object(module.rebuild_estimate, "describe", return_value=[]):
                with mock.patch("builtins.print"):
                    module.print_estimate()

    assert estimate_files.call_args.kwargs["local"] is True


def test_main_prints_the_estimate_before_the_rebuild_starts(rebuild_index_module):
    module = rebuild_index_module
    order = []

    async def rebuild():
        order.append("rebuild")

    with mock.patch.object(module, "print_estimate", side_effect=lambda: order.append("estimate")):
        with mock.patch.object(module, "async_main", side_effect=rebuild):
            with mock.patch("builtins.print"):
                assert module.main() == 0

    assert order == ["estimate", "rebuild"]
