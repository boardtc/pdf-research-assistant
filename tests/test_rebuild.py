import asyncio
import importlib
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
    bootstrap_module = ModuleType("pdf_research_assistant.bootstrap")
    search_module = ModuleType("paperqa.agents.search")
    settings_object = mock.Mock()
    settings_object.agent.index.index_directory = Path("index-root")
    settings_object.get_index_name.return_value = "active-index"

    bootstrap_module.get_allowed_paths = mock.Mock(return_value={"paper_a.pdf", "paper_b.pdf"})
    bootstrap_module.FAILED_DOCUMENT_ADD_ID = "ERROR"
    bootstrap_module.build_settings = mock.Mock(return_value=settings_object)
    bootstrap_module.get_failed_files = mock.Mock(return_value=[])
    bootstrap_module.get_indexed_doc_count = mock.Mock(side_effect=[1, 2])

    search_module.get_directory_index = mock.AsyncMock(name="get_directory_index")
    search_module.process_file = mock.AsyncMock(name="process_file")

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

    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 1"),
        mock.call("Indexed before run: 105"),
        mock.call("Rebuild completed with parser/indexing errors; recorded failed PDFs will be reported below."),
        mock.call("Indexed after run: 105"),
        mock.call("Failed PDFs: 1"),
        mock.call("FAILED: Session 7/Sadler 2013 Opening-up-feedback-Teaching-learners-to-see Chapter.pdf"),
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

    with mock.patch("builtins.print") as print_mock:
        asyncio.run(rebuild_index_module.async_main())

    assert print_mock.call_args_list == [
        mock.call("Manifest PDFs: 1"),
        mock.call("Indexed before run: 3"),
        mock.call("Rebuild completed with parser/indexing errors before any failed PDFs were recorded."),
        mock.call("ERROR: unknown encoding: /SymbolSetEncoding"),
        mock.call("ERROR: bad metadata"),
        mock.call("Indexed after run: 3"),
        mock.call("Failed PDFs: 0"),
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
