import importlib
import os
import pickle
import shutil
import sys
import uuid
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _CaptureConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _CaptureSettings(_CaptureConfig):
    pass


def _write_files_zip(index_dir: Path, shard_name: str, payload: dict) -> None:
    shard_dir = index_dir / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    data = zlib.compress(pickle.dumps(payload))
    (shard_dir / "files.zip").write_bytes(data)


@pytest.fixture
def workspace_tmp_path():
    # Create a unique per-test directory under tests/.tmp/ so filesystem tests stay isolated.
    root = PROJECT_ROOT / "tests" / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(uuid.uuid4())
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def bootstrap_module(workspace_tmp_path):
    paperqa_module = ModuleType("paperqa")
    paperqa_settings_module = ModuleType("paperqa.settings")
    dotenv_module = ModuleType("dotenv")

    paperqa_module.Settings = _CaptureSettings
    paperqa_settings_module.AgentSettings = _CaptureConfig
    paperqa_settings_module.IndexSettings = _CaptureConfig
    paperqa_settings_module.ParsingSettings = _CaptureConfig

    # Mock dotenv loading so importing bootstrap does not depend on the local shell state.
    dotenv_module.load_dotenv = mock.Mock(name="load_dotenv")

    paper_dir = workspace_tmp_path / "papers"
    paper_dir.mkdir()
    env = {
        "PAPER_DIR": str(paper_dir),
        "INDEX_DIR": str(workspace_tmp_path / "index"),
        "MANIFEST_PATH": str(workspace_tmp_path / "manifest.csv"),
    }
    with mock.patch.dict(
        os.environ,
        env,
        clear=False,
    ):
        with mock.patch.dict(
            sys.modules,
            {
                "paperqa": paperqa_module,
                "paperqa.settings": paperqa_settings_module,
                "dotenv": dotenv_module,
                "pdf_research_assistant.pdf_parser": ModuleType("pdf_research_assistant.pdf_parser"),
            },
        ):
            sys.modules["pdf_research_assistant.pdf_parser"].parse_pdf_with_fallback = mock.Mock(
                name="parse_pdf_with_fallback"
            )
            sys.modules.pop("pdf_research_assistant.bootstrap", None)
            module = importlib.import_module("pdf_research_assistant.bootstrap")
            yield module
            sys.modules.pop("pdf_research_assistant.bootstrap", None)


def test_import_with_existing_manifest_populates_allowed_paths(workspace_tmp_path):
    paperqa_module = ModuleType("paperqa")
    paperqa_settings_module = ModuleType("paperqa.settings")
    dotenv_module = ModuleType("dotenv")

    paperqa_module.Settings = _CaptureSettings
    paperqa_settings_module.AgentSettings = _CaptureConfig
    paperqa_settings_module.IndexSettings = _CaptureConfig
    paperqa_settings_module.ParsingSettings = _CaptureConfig
    dotenv_module.load_dotenv = mock.Mock(name="load_dotenv")

    paper_dir = workspace_tmp_path / "papers"
    paper_dir.mkdir()
    manifest_path = workspace_tmp_path / "manifest.csv"
    manifest_path.write_text("file_location\nfolder/paper.pdf\n", encoding="utf-8")

    env = {
        "PAPER_DIR": str(paper_dir),
        "INDEX_DIR": str(workspace_tmp_path / "index"),
        "MANIFEST_PATH": str(manifest_path),
    }

    with mock.patch.dict(os.environ, env, clear=False):
        with mock.patch.dict(
            sys.modules,
            {
                "paperqa": paperqa_module,
                "paperqa.settings": paperqa_settings_module,
                "dotenv": dotenv_module,
                "pdf_research_assistant.pdf_parser": ModuleType("pdf_research_assistant.pdf_parser"),
            },
        ):
            sys.modules["pdf_research_assistant.pdf_parser"].parse_pdf_with_fallback = mock.Mock(
                name="parse_pdf_with_fallback"
            )
            sys.modules.pop("pdf_research_assistant.bootstrap", None)
            module = importlib.import_module("pdf_research_assistant.bootstrap")
            try:
                assert module.ALLOWED_PATHS == {"folder/paper.pdf"}
            finally:
                sys.modules.pop("pdf_research_assistant.bootstrap", None)


def test_sanitize_proxy_environment_leaves_environment_unchanged_when_no_proxy_variables_are_set(bootstrap_module):
    with mock.patch.dict(os.environ, {}, clear=True):
        bootstrap_module.sanitize_proxy_environment()

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert "ALL_PROXY" not in os.environ


def test_sanitize_proxy_environment_keeps_proxy_when_value_is_not_a_loopback_blackhole(bootstrap_module):
    with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.example.com:8080"}, clear=True):
        bootstrap_module.sanitize_proxy_environment()
        assert os.environ["HTTP_PROXY"] == "http://proxy.example.com:8080"


def test_sanitize_proxy_environment_removes_http_proxy_when_it_points_at_loopback_port_9(bootstrap_module):
    with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9"}, clear=True):
        bootstrap_module.sanitize_proxy_environment()

    assert "HTTP_PROXY" not in os.environ


def test_sanitize_proxy_environment_removes_https_proxy_when_it_points_at_loopback_port_9(bootstrap_module):
    with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://localhost:9"}, clear=True):
        bootstrap_module.sanitize_proxy_environment()

    assert "HTTPS_PROXY" not in os.environ


def test_sanitize_proxy_environment_removes_all_proxy_when_it_points_at_loopback_port_9(bootstrap_module):
    with mock.patch.dict(os.environ, {"ALL_PROXY": "http://[::1]:9"}, clear=True):
        bootstrap_module.sanitize_proxy_environment()

    assert "ALL_PROXY" not in os.environ


def test_env_path_returns_expanded_env_value_when_variable_is_set(bootstrap_module):
    env = {"TEST_PATH_VAR": "~/docs"}
    expected = Path("/tmp/test-user/docs")
    if sys.platform == "win32":
        env["USERPROFILE"] = r"C:\Users\TestUser"
        expected = Path(r"C:\Users\TestUser\docs")
    else:
        env["HOME"] = "/tmp/test-user"

    with mock.patch.dict(os.environ, env, clear=True):
        result = bootstrap_module.env_path("TEST_PATH_VAR")

    assert result == expected


def test_env_path_returns_default_when_variable_is_missing_and_default_is_provided(bootstrap_module, workspace_tmp_path):
    default_path = workspace_tmp_path / "fallback"

    with mock.patch.dict(os.environ, {}, clear=True):
        result = bootstrap_module.env_path("TEST_PATH_VAR", default=default_path)

    assert result == default_path


def test_env_path_raises_config_error_when_variable_is_missing_and_required_is_true(bootstrap_module):
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="TEST_PATH_VAR is not set. Configure it in .env or your environment"):
            bootstrap_module.env_path("TEST_PATH_VAR", required=True)


def test_env_path_raises_basic_error_when_variable_is_missing_and_no_default_or_required_flag_is_given(
    bootstrap_module,
):
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="TEST_PATH_VAR is not set."):
            bootstrap_module.env_path("TEST_PATH_VAR")


def test_manifest_exists_returns_true_when_manifest_file_exists(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"
    manifest_path.write_text("file_location\npaper.pdf\n", encoding="utf-8")

    assert bootstrap_module.manifest_exists(manifest_path) is True


def test_load_allowed_manifest_paths_returns_empty_set_when_manifest_file_is_missing(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "missing.csv"

    assert bootstrap_module.load_allowed_manifest_paths(manifest_path) == set()


def test_load_allowed_manifest_paths_loads_a_single_manifest_path_in_normalized_form(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"
    manifest_path.write_text("file_location\npaper.pdf\n", encoding="utf-8")

    assert bootstrap_module.load_allowed_manifest_paths(manifest_path) == {"paper.pdf"}


def test_load_allowed_manifest_paths_normalizes_slashes_to_a_stable_relative_form(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"
    manifest_path.write_text("file_location\nfolder/paper.pdf\n", encoding="utf-8")

    assert bootstrap_module.load_allowed_manifest_paths(manifest_path) == {"folder/paper.pdf"}


def test_load_allowed_manifest_paths_skips_rows_without_file_location_values(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"
    manifest_path.write_text("file_location\n\npaper.pdf\n", encoding="utf-8")

    assert bootstrap_module.load_allowed_manifest_paths(manifest_path) == {"paper.pdf"}


def test_get_allowed_paths_returns_manifest_paths_for_current_configuration(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"
    manifest_path.write_text("file_location\nfolder/paper.pdf\n", encoding="utf-8")

    assert bootstrap_module.get_allowed_paths(manifest_path) == {"folder/paper.pdf"}


def test_normalize_relative_pdf_path_converts_backslashes_to_forward_slashes(bootstrap_module):
    result = bootstrap_module.normalize_relative_pdf_path(r"folder\paper.pdf")

    assert result == "folder/paper.pdf"


def test_use_manifest_returns_true_when_manifest_contains_at_least_one_allowed_path(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"
    manifest_path.write_text("file_location\npaper.pdf\n", encoding="utf-8")

    assert bootstrap_module.use_manifest(manifest_path) is True


def test_normalize_file_location_strips_paper_dir_prefix_from_absolute_path(bootstrap_module):
    paper_dir = Path(r"C:\papers")

    result = bootstrap_module.normalize_file_location(Path(r"C:\papers\folder\paper.pdf"), paper_dir=paper_dir)

    assert result == "folder/paper.pdf"


def test_normalize_file_location_preserves_relative_path_when_paper_dir_prefix_is_not_present(bootstrap_module):
    result = bootstrap_module.normalize_file_location("folder/paper.pdf", paper_dir=Path(r"C:\papers"))

    assert result == "folder/paper.pdf"


def test_only_manifest_allows_pdf_when_manifest_is_not_configured(bootstrap_module):
    # Mock manifest lookup so this test exercises the "all PDFs allowed" branch directly.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value=set()):
        assert bootstrap_module.only_manifest("folder/paper.pdf") is True


def test_only_manifest_rejects_non_pdf_when_manifest_is_not_configured(bootstrap_module):
    # Mock manifest lookup so this test exercises the extension filter branch directly.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value=set()):
        assert bootstrap_module.only_manifest("folder/paper.txt") is False


def test_only_manifest_allows_path_present_in_manifest_when_manifest_scope_is_enabled(bootstrap_module):
    # Mock manifest lookup so this test exercises the relative-path membership branch directly.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value={"folder/paper.pdf"}):
        # Mock the configured paper root so normalization strips the shared prefix before matching.
        with mock.patch.object(bootstrap_module, "PAPER_DIR", Path(r"C:\papers")):
            assert bootstrap_module.only_manifest(Path(r"C:\papers\folder\paper.pdf")) is True


def test_only_manifest_rejects_path_absent_from_manifest_when_manifest_scope_is_enabled(bootstrap_module):
    # Mock manifest lookup so this test exercises the negative membership branch directly.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value={"folder/paper.pdf"}):
        # Mock the configured paper root so normalization strips the shared prefix before matching.
        with mock.patch.object(bootstrap_module, "PAPER_DIR", Path(r"C:\papers")):
            assert bootstrap_module.only_manifest(Path(r"C:\papers\folder\other.pdf")) is False


def test_get_indexed_doc_count_returns_zero_when_index_directory_does_not_exist(bootstrap_module, workspace_tmp_path):
    assert bootstrap_module.get_indexed_doc_count(workspace_tmp_path / "missing-index") == 0


def test_get_indexed_doc_count_ignores_unreadable_index_files_and_continues_scanning(bootstrap_module, workspace_tmp_path):
    index_dir = workspace_tmp_path / "index"
    bad_shard = index_dir / "bad"
    bad_shard.mkdir(parents=True)
    (bad_shard / "files.zip").write_bytes(b"not-a-valid-archive")
    _write_files_zip(index_dir, "good", {"paper.pdf": "OK"})

    # Mock manifest lookup so valid documents remain in scope while the unreadable shard is skipped.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value=set()):
        assert bootstrap_module.get_indexed_doc_count(index_dir) == 1


def test_get_indexed_doc_count_skips_entries_marked_as_failed_documents(bootstrap_module, workspace_tmp_path):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(
        index_dir,
        "one",
        {
            "paper.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID,
            "kept.pdf": "OK",
        },
    )

    # Mock manifest lookup so this test focuses on failed-entry filtering rather than manifest scoping.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value=set()):
        assert bootstrap_module.get_indexed_doc_count(index_dir) == 1


def test_get_indexed_doc_count_counts_all_normalized_documents_when_manifest_scope_is_disabled(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(index_dir, "one", {"paper_a.pdf": "OK", "paper_b.pdf": "OK"})

    # Mock manifest lookup with an empty set so every valid PDF is counted.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value=set()):
        assert bootstrap_module.get_indexed_doc_count(index_dir) == 2


def test_get_indexed_doc_count_counts_only_documents_present_in_allowed_manifest_paths(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(index_dir, "one", {"keep.pdf": "OK", "drop.pdf": "OK"})

    # Mock manifest lookup so only the whitelisted document remains in scope.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value={"keep.pdf"}):
        assert bootstrap_module.get_indexed_doc_count(index_dir) == 1


def test_get_indexed_doc_count_deduplicates_same_document_seen_in_multiple_index_files(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(index_dir, "one", {"paper.pdf": "OK"})
    _write_files_zip(index_dir, "two", {"paper.pdf": "OK"})

    # Mock manifest lookup with an empty set so duplicate handling is the only thing under test.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value=set()):
        assert bootstrap_module.get_indexed_doc_count(index_dir) == 1


def test_get_active_index_dir_returns_current_settings_shard_path(bootstrap_module, workspace_tmp_path):
    fake_settings = SimpleNamespace(
        agent=SimpleNamespace(index=SimpleNamespace(index_directory=workspace_tmp_path / "index")),
        get_index_name=mock.Mock(return_value="pqa_index_active"),
    )

    result = bootstrap_module.get_active_index_dir(fake_settings)

    assert result == workspace_tmp_path / "index" / "pqa_index_active"
    fake_settings.get_index_name.assert_called_once_with()


def test_get_indexed_doc_count_normalizes_absolute_indexed_paths_before_manifest_comparison(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    paper_dir = workspace_tmp_path / "papers"
    absolute_path = str(paper_dir / "folder" / "paper.pdf")
    _write_files_zip(index_dir, "one", {absolute_path: "OK"})

    # Mock the configured paper root so normalization strips the shared prefix before matching.
    with mock.patch.object(bootstrap_module, "PAPER_DIR", paper_dir):
        # Mock manifest lookup so the test can assert on normalized matching only.
        with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value={"folder/paper.pdf"}):
            assert bootstrap_module.get_indexed_doc_count(index_dir) == 1


def test_get_indexed_doc_count_returns_zero_when_every_indexed_entry_is_failed_or_out_of_scope(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(
        index_dir,
        "one",
        {
            "failed.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID,
            "out_of_scope.pdf": "OK",
        },
    )

    # Mock manifest lookup so the remaining successful document is explicitly out of scope.
    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value={"keep.pdf"}):
        assert bootstrap_module.get_indexed_doc_count(index_dir) == 0


def test_get_indexed_doc_count_reads_direct_files_zip_when_index_dir_is_a_single_active_shard(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index" / "active"
    _write_files_zip(index_dir, "", {"paper.pdf": "OK"})

    with mock.patch.object(bootstrap_module, "get_allowed_paths", return_value=set()):
        assert bootstrap_module.get_indexed_doc_count(index_dir) == 1


def test_get_failed_files_returns_empty_list_when_index_directory_does_not_exist(
    bootstrap_module, workspace_tmp_path
):
    assert bootstrap_module.get_failed_files(workspace_tmp_path / "missing-index") == []


def test_get_failed_files_ignores_unreadable_index_files_and_continues_scanning(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    bad_shard = index_dir / "bad"
    bad_shard.mkdir(parents=True)
    (bad_shard / "files.zip").write_bytes(b"not-a-valid-archive")
    _write_files_zip(index_dir, "good", {"failed.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID})

    assert bootstrap_module.get_failed_files(index_dir) == ["failed.pdf"]


def test_get_failed_files_collects_failed_file_locations_from_index_payloads(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(
        index_dir,
        "one",
        {"failed.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID, "ok.pdf": "OK"},
    )

    assert bootstrap_module.get_failed_files(index_dir) == ["failed.pdf"]


def test_get_failed_files_reads_direct_files_zip_when_index_dir_is_a_single_active_shard(
    bootstrap_module, workspace_tmp_path
):
    index_dir = workspace_tmp_path / "index" / "active"
    _write_files_zip(index_dir, "", {"failed.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID})

    assert bootstrap_module.get_failed_files(index_dir) == ["failed.pdf"]


def test_get_failed_files_skips_successfully_indexed_documents(bootstrap_module, workspace_tmp_path):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(index_dir, "one", {"ok.pdf": "OK"})

    assert bootstrap_module.get_failed_files(index_dir) == []


def test_get_failed_files_deduplicates_repeated_failed_file_locations(bootstrap_module, workspace_tmp_path):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(index_dir, "one", {"failed.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID})
    _write_files_zip(index_dir, "two", {"failed.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID})

    assert bootstrap_module.get_failed_files(index_dir) == ["failed.pdf"]


def test_get_failed_files_returns_failed_file_locations_in_sorted_order(bootstrap_module, workspace_tmp_path):
    index_dir = workspace_tmp_path / "index"
    _write_files_zip(
        index_dir,
        "one",
        {
            "zeta.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID,
            "alpha.pdf": bootstrap_module.FAILED_DOCUMENT_ADD_ID,
        },
    )

    assert bootstrap_module.get_failed_files(index_dir) == ["alpha.pdf", "zeta.pdf"]


def test_build_settings_uses_shared_model_name_for_llm_summary_and_agent_llm(bootstrap_module, workspace_tmp_path):
    paper_dir = workspace_tmp_path / "papers"
    index_dir = workspace_tmp_path / "index"
    manifest_path = workspace_tmp_path / "manifest.csv"

    # Mock module-level paths so build_settings uses test-owned directories instead of imported defaults.
    with mock.patch.multiple(
        bootstrap_module,
        PAPER_DIR=paper_dir,
        INDEX_DIR=index_dir,
        MANIFEST_PATH=manifest_path,
    ):
        # Mock manifest existence so the settings object includes the manifest path branch.
        with mock.patch.object(bootstrap_module, "manifest_exists", return_value=True):
            settings = bootstrap_module.build_settings()

    assert settings.llm == bootstrap_module.MODEL_NAME
    assert settings.summary_llm == bootstrap_module.MODEL_NAME
    assert settings.agent.agent_llm == bootstrap_module.MODEL_NAME
    assert settings.agent.index.paper_directory == paper_dir
    assert settings.agent.index.index_directory == index_dir
    assert settings.agent.index.files_filter is bootstrap_module.only_manifest


def test_build_settings_uses_custom_pdf_parser_with_fallback(bootstrap_module):
    settings = bootstrap_module.build_settings()

    assert settings.parsing.parse_pdf is bootstrap_module.parse_pdf_with_fallback


def test_build_settings_uses_manifest_path_when_manifest_exists(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"

    # Mock module-level paths so the produced settings object is easy to assert against.
    with mock.patch.multiple(bootstrap_module, MANIFEST_PATH=manifest_path):
        # Mock manifest existence so build_settings follows the explicit manifest branch.
        with mock.patch.object(bootstrap_module, "manifest_exists", return_value=True):
            settings = bootstrap_module.build_settings()

    assert settings.agent.index.manifest_file == manifest_path


def test_build_settings_omits_manifest_path_when_manifest_does_not_exist(bootstrap_module, workspace_tmp_path):
    manifest_path = workspace_tmp_path / "manifest.csv"

    # Mock module-level paths so the produced settings object is easy to assert against.
    with mock.patch.multiple(bootstrap_module, MANIFEST_PATH=manifest_path):
        # Mock manifest existence so build_settings follows the no-manifest branch.
        with mock.patch.object(bootstrap_module, "manifest_exists", return_value=False):
            settings = bootstrap_module.build_settings()

    assert settings.agent.index.manifest_file is None


def test_get_model_settings_uses_the_hosted_model_when_no_local_llm_is_saved(bootstrap_module):
    with mock.patch.object(bootstrap_module.local_llm, "load_config", return_value=None):
        name, config = bootstrap_module.get_model_settings()

    assert name == bootstrap_module.MODEL_NAME
    assert config == {"rate_limit": {bootstrap_module.MODEL_NAME: "30000 per 1 minute"}}


def test_get_model_settings_uses_the_saved_local_llm(bootstrap_module):
    saved = {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3.2"}
    with mock.patch.object(bootstrap_module.local_llm, "load_config", return_value=saved):
        name, config = bootstrap_module.get_model_settings()

    assert name == "ollama/llama3.2"
    assert config["model_list"][0]["litellm_params"]["api_base"] == "http://localhost:11434"
    assert "rate_limit" not in config


def test_build_settings_sends_all_three_model_roles_to_the_saved_local_llm(bootstrap_module):
    saved = {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3.2"}
    with mock.patch.object(bootstrap_module.local_llm, "load_config", return_value=saved):
        settings = bootstrap_module.build_settings()

    assert settings.llm == settings.summary_llm == settings.agent.agent_llm == "ollama/llama3.2"
    configs = [settings.llm_config, settings.summary_llm_config, settings.agent.agent_llm_config]
    assert all(config["model_list"][0]["model_name"] == "ollama/llama3.2" for config in configs)
    # Each role gets its own copy, so changing one, as exhaustive mode does, leaves the others alone.
    assert len({id(config) for config in configs}) == 3


def test_build_settings_keeps_the_hosted_rate_limit_for_each_model_role(bootstrap_module):
    settings = bootstrap_module.build_settings()

    expected = {"rate_limit": {bootstrap_module.MODEL_NAME: "30000 per 1 minute"}}
    assert settings.llm_config == settings.summary_llm_config == settings.agent.agent_llm_config == expected
