"""Shared startup and runtime wiring for the UI, CLI, and rebuild scripts.

Edit .env or environment variables for local configuration; this module is
internal project wiring.
"""

import csv
from copy import deepcopy
import os
import pickle
import zlib
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import urlparse

from dotenv import load_dotenv
from paperqa import Settings
from paperqa.settings import AgentSettings, AnswerSettings, IndexSettings, ParsingSettings

from pdf_research_assistant import local_llm, rate_profiles, saved_answers, saved_settings
from pdf_research_assistant.pdf_parser import parse_pdf_with_fallback

load_dotenv()

# Avoid reusing opened Tantivy indexes across different asyncio event loops.
os.environ.setdefault("PQA_INDEX_DONT_CACHE_INDEXES", "true")

PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
LOOPBACK_PROXY_HOSTS = {"127.0.0.1", "localhost", "::1"}

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "manifest.csv"
FAILED_DOCUMENT_ADD_ID = "ERROR"
MODEL_NAME = "gpt-4o-mini"
PDF_EXTENSIONS = {".pdf"}
# How sharply images and tables are drawn when they are read from PDFs. PaperQA's reader draws them at 72
# dots per inch, where the text in a table is barely readable. PaperQA does not name an index after this,
# so changing it keeps the index in use and only affects PDFs read afterwards.
PICTURE_DPI = 150


def sanitize_proxy_environment() -> None:
    """Drop obviously broken loopback proxy placeholders from the process environment."""
    for name in PROXY_ENV_VARS:
        value = os.getenv(name)
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.hostname in LOOPBACK_PROXY_HOSTS and parsed.port == 9:
            os.environ.pop(name, None)


sanitize_proxy_environment()


def env_path(name: str, default: Path | None = None, required: bool = False) -> Path:
    value = os.getenv(name)
    if value:
        return Path(value).expanduser()
    if default is not None:
        return default
    if required:
        raise RuntimeError(
            f"{name} is not set. Configure it in .env or your environment before running the app."
        )
    raise RuntimeError(f"{name} is not set.")


PAPER_DIR = env_path("PAPER_DIR", required=True)
INDEX_DIR = env_path("INDEX_DIR", DEFAULT_INDEX_DIR)
MANIFEST_PATH = env_path("MANIFEST_PATH", DEFAULT_MANIFEST_PATH)


def manifest_exists(manifest_path: Path = MANIFEST_PATH) -> bool:
    """Return whether a manifest file is present at the configured path."""
    return manifest_path.exists()


def normalize_relative_pdf_path(file_location: str | Path) -> str:
    """Normalize a relative PDF path to a forward-slash form for stable comparisons."""
    value = str(file_location).replace("\\", "/").strip()
    return str(PurePosixPath(value))


def load_allowed_manifest_paths(manifest_path: Path = MANIFEST_PATH) -> set[str]:
    """Load manifest-listed PDF paths in a normalized relative form."""
    if not manifest_exists(manifest_path):
        return set()
    with open(manifest_path, newline="", encoding="utf-8-sig") as handle:
        return {
            normalize_relative_pdf_path(row["file_location"])
            for row in csv.DictReader(handle)
            if row.get("file_location")
        }


ALLOWED_PATHS = load_allowed_manifest_paths()
USE_MANIFEST = bool(ALLOWED_PATHS)


def get_allowed_paths(manifest_path: Path = MANIFEST_PATH) -> set[str]:
    """Return the currently allowed relative PDF paths from the manifest, if any."""
    return load_allowed_manifest_paths(manifest_path)


def use_manifest(manifest_path: Path = MANIFEST_PATH) -> bool:
    """Return whether manifest scoping is active for the current configuration."""
    return bool(get_allowed_paths(manifest_path))


def normalize_file_location(file_location: str | Path, paper_dir: Path = PAPER_DIR) -> str:
    """Convert an indexed file path into the manifest-style relative path form."""
    value = normalize_relative_pdf_path(file_location)
    prefix = normalize_relative_pdf_path(paper_dir).rstrip("/")
    if prefix and value.startswith(prefix + "/"):
        return value[len(prefix) + 1 :]
    return value


def only_manifest(path: str | Path) -> bool:
    """Allow only manifest-listed PDFs, or all PDFs when no manifest is configured."""
    current_allowed_paths = get_allowed_paths()
    if not current_allowed_paths:
        return Path(path).suffix.lower() in PDF_EXTENSIONS
    return normalize_file_location(path, paper_dir=PAPER_DIR) in current_allowed_paths


def iter_index_file_archives(index_dir: Path) -> list[Path]:
    """Return the relevant index archive files for either a shard dir or an index root dir.

    The index of past answers beside the shards lists answers, not PDFs, so it is left out.
    """
    direct_archive = index_dir / "files.zip"
    if direct_archive.exists():
        return [direct_archive]
    return [path for path in index_dir.glob("*/files.zip") if path.parent.name != saved_answers.INDEX_NAME]


def get_active_index_dir(settings: Settings | None = None) -> Path:
    """Return the active PaperQA index directory for the current settings hash."""
    current_settings = settings or build_settings()
    return Path(current_settings.agent.index.index_directory) / current_settings.get_index_name()


def get_indexed_doc_count(index_dir: Path = INDEX_DIR) -> int:
    """Count indexed PDFs that are in scope for the current manifest settings."""
    if not index_dir.exists():
        return 0
    current_allowed_paths = get_allowed_paths()
    current_use_manifest = bool(current_allowed_paths)
    indexed = set()
    for file_index in iter_index_file_archives(index_dir):
        try:
            data = pickle.loads(zlib.decompress(file_index.read_bytes()))
        except Exception:
            continue
        for file_location, status in data.items():
            if status == FAILED_DOCUMENT_ADD_ID:
                continue
            normalized = normalize_file_location(file_location)
            if not current_use_manifest or normalized in current_allowed_paths:
                indexed.add(normalized)
    return len(indexed)


def get_failed_files(index_dir: Path = INDEX_DIR) -> list[str]:
    """Return unique file paths that PaperQA recorded as failed during indexing."""
    failed = []
    if not index_dir.exists():
        return failed
    for file_index in iter_index_file_archives(index_dir):
        try:
            data = pickle.loads(zlib.decompress(file_index.read_bytes()))
        except Exception:
            continue
        for file_location, status in data.items():
            if status == FAILED_DOCUMENT_ADD_ID:
                failed.append(str(file_location))
    return sorted(set(failed))


def get_model_settings() -> tuple[str, dict]:
    """Return the model name and config for answers: a saved local LLM, or else the hosted model.

    The rate limits are added later by build_settings, from the chosen rate profile.
    """
    saved_local_llm = local_llm.load_config()
    if saved_local_llm:
        return local_llm.litellm_model(saved_local_llm)
    return MODEL_NAME, {}


def build_settings() -> Settings:
    """Build the shared PaperQA settings object used across all entry points, with the sidebar choices applied."""
    model_name, model_config = get_model_settings()
    settings = Settings(
        llm=model_name,
        summary_llm=model_name,
        llm_config=deepcopy(model_config),
        summary_llm_config=deepcopy(model_config),
        answer=AnswerSettings(),
        parsing=ParsingSettings(
            multimodal=False,
            use_doc_details=False,
            parse_pdf=parse_pdf_with_fallback,
            # PaperQA's own chunk size and overlap, plus sharper pictures.
            reader_config={"chunk_chars": 5000, "overlap": 250, "dpi": PICTURE_DPI},
        ),
        agent=AgentSettings(
            agent_llm=model_name,
            agent_llm_config=deepcopy(model_config),
            index=IndexSettings(
                paper_directory=PAPER_DIR,
                index_directory=INDEX_DIR,
                manifest_file=MANIFEST_PATH if manifest_exists() else None,
                use_absolute_paper_directory=False,
                recurse_subdirectories=True,
                concurrency=1,
                files_filter=only_manifest,
            ),
        ),
    )
    # A rate profile saved in the sidebar replaces this one when saved_settings.apply runs.
    rate_profiles.apply_profile(settings, rate_profiles.DEFAULT_PROFILE)
    saved_settings.apply(settings)
    return settings

