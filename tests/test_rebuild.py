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
