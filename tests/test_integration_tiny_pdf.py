"""Integration tests: index a real one-page PDF and search it, with no API calls.

Unlike the unit tests, nothing here is stubbed out of the project or PaperQA. Two
things are swapped so the tests run offline and give the same result every time:
PaperQA's local "sparse" embedding replaces OpenAI embeddings, and the scoring model
returns a fixed reply through LiteLLM's mock_response instead of calling OpenAI.

The fixture is tests/fixtures/tiny-enthusiasm.pdf; make_tiny_pdf.py beside it
regenerates it. Indexing runs once for the module and the tests share the index.
"""

import asyncio
import contextlib
import importlib
import io
import json
import pickle
import shutil
import sys
import uuid
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "tiny-enthusiasm.pdf"
PDF_NAME = "tiny-enthusiasm.pdf"
PDF_TITLE = "Tiny enthusiasm fixture"
QUESTION = "Does teacher enthusiasm help students learn?"
QUOTE = "Enthusiasm is contagious in a classroom."
FAKE_API_KEY = "sk-test-not-a-real-key"
REBUILD_ATTEMPTS = 3
PROJECT_MODULES = (
    "pdf_research_assistant.bootstrap",
    "pdf_research_assistant.rebuild",
    "pdf_research_assistant.exhaustive",
)


def forget_project_modules():
    for name in PROJECT_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def tiny_index():
    """Index the fixture PDF through the real rebuild entry point, in a throwaway folder."""
    root = PROJECT_ROOT / "tests" / ".tmp" / str(uuid.uuid4())
    papers = root / "papers"
    papers.mkdir(parents=True)
    shutil.copy(FIXTURE_PDF, papers / PDF_NAME)
    manifest = root / "manifest.csv"
    manifest.write_text(
        f"file_location,title,citation,doi\n{PDF_NAME},{PDF_TITLE},Tests (2026) {PDF_TITLE},\n", encoding="utf-8"
    )

    with pytest.MonkeyPatch.context() as env:
        env.setenv("PAPER_DIR", str(papers))
        env.setenv("INDEX_DIR", str(root / "index"))
        env.setenv("MANIFEST_PATH", str(manifest))
        env.setenv("OPENAI_API_KEY", FAKE_API_KEY)
        env.setenv("PQA_INDEX_DONT_CACHE_INDEXES", "true")
        forget_project_modules()
        try:
            rebuild = importlib.import_module("pdf_research_assistant.rebuild")
            exhaustive = importlib.import_module("pdf_research_assistant.exhaustive")
            rebuild.settings.embedding = "sparse"
            # On Windows another program can briefly lock a new index file ("Access is denied").
            # The rebuild reports that and carries on when run again, as a user would run it again.
            for _attempt in range(REBUILD_ATTEMPTS):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    asyncio.run(rebuild.async_main())
                if "Indexed after run: 1" in output.getvalue():
                    break
            settings = exhaustive.build_exhaustive_settings()
            settings.embedding = "sparse"
            yield SimpleNamespace(
                root=root,
                rebuild_output=output.getvalue(),
                index_dir=rebuild.get_active_index_dir(),
                exhaustive=exhaustive,
                settings=settings,
            )
        finally:
            forget_project_modules()
            shutil.rmtree(root, ignore_errors=True)


def canned_model(reply: dict):
    """Return the real LiteLLM model class, set to answer every call with `reply`."""
    from lmi import LiteLLMModel

    name = "gpt-4o-mini"
    params = {"model": name, "mock_response": json.dumps(reply), "api_key": FAKE_API_KEY}
    return LiteLLMModel(name=name, config={"model_list": [{"model_name": name, "litellm_params": params}]})


def run_search(tiny_index, reply: dict, folder: str) -> dict:
    output_dir = tiny_index.root / folder
    summary = asyncio.run(
        tiny_index.exhaustive.run_exhaustive(
            QUESTION,
            output_dir,
            settings=tiny_index.settings,
            index_dir=tiny_index.index_dir,
            summary_llm_model=canned_model(reply),
        )
    )
    return {"summary": summary, "report": (output_dir / "report.md").read_text(encoding="utf-8")}


def test_rebuild_indexes_the_tiny_pdf_with_its_text(tiny_index):
    assert "Indexed after run: 1" in tiny_index.rebuild_output
    assert "Failed PDFs: 0" in tiny_index.rebuild_output

    index_files = pickle.loads(zlib.decompress((tiny_index.index_dir / "files.zip").read_bytes()))
    assert list(index_files) == [PDF_NAME]
    saved = pickle.loads(zlib.decompress((tiny_index.index_dir / "docs" / f"{index_files[PDF_NAME]}.zip").read_bytes()))
    saved_text = " ".join(" ".join(text.text.split()) for text in saved.texts)
    assert "shows genuine enthusiasm for the subject." in saved_text


def test_exhaustive_plan_finds_the_indexed_pdf_without_api_calls(tiny_index):
    plan = tiny_index.exhaustive.build_plan(QUESTION, settings=tiny_index.settings, index_dir=tiny_index.index_dir)

    assert plan["paper_count"] == 1
    assert plan["status_counts"] == {tiny_index.exhaustive.STATUS_LOADABLE: 1}
    assert plan["chunk_count"] >= 1


def test_exhaustive_run_reports_relevant_evidence_from_the_pdf(tiny_index):
    result = run_search(
        tiny_index,
        {"quote": QUOTE, "summary": "Teacher enthusiasm helps students learn.", "relevance_score": 9},
        "relevant-run",
    )

    paper = result["summary"]["papers"][0]
    assert paper["status"] == tiny_index.exhaustive.RESULT_RELEVANT
    assert paper["chunks_failed"] == 0
    evidence = paper["evidence"][0]
    assert evidence["tier"] == tiny_index.exhaustive.TIER_BEST
    assert evidence["quote_verified"] is True
    assert PDF_TITLE in result["report"]
    assert f'Quote (verified): "{QUOTE}"' in result["report"]


def test_exhaustive_run_reports_no_evidence_when_every_chunk_scores_zero(tiny_index):
    result = run_search(tiny_index, {"quote": "", "summary": "", "relevance_score": 0}, "irrelevant-run")

    paper = result["summary"]["papers"][0]
    assert paper["status"] == tiny_index.exhaustive.RESULT_NO_EVIDENCE
    assert paper["evidence"] == []
    assert "## Papers with no relevant evidence" in result["report"]
