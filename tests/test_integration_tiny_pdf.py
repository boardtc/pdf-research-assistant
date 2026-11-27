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
PDF_CITATION = f"Tests (2026) {PDF_TITLE}"
QUESTION = "Does teacher enthusiasm help students learn?"
QUOTE = "Enthusiasm is contagious in a classroom."
FAKE_API_KEY = "sk-test-not-a-real-key"
REBUILD_ATTEMPTS = 3
# Longest one rebuild of the fixture PDF may take; it takes a few seconds.
REBUILD_SECONDS = 120
PROJECT_MODULES = (
    "pdf_research_assistant.bootstrap",
    "pdf_research_assistant.rebuild",
    "pdf_research_assistant.exhaustive",
)


def forget_project_modules():
    for name in PROJECT_MODULES:
        sys.modules.pop(name, None)


@contextlib.contextmanager
def tiny_workspace(folder: str = ""):
    """Set up a throwaway folder with the fixture PDF (in `folder` under the PDF folder, if given) and its manifest,
    and load the project against it."""
    root = PROJECT_ROOT / "tests" / ".tmp" / str(uuid.uuid4())
    papers = root / "papers"
    (papers / folder).mkdir(parents=True)
    shutil.copy(FIXTURE_PDF, papers / folder / PDF_NAME)
    manifest = root / "manifest.csv"
    listed = f"{folder}/{PDF_NAME}" if folder else PDF_NAME
    manifest.write_text(f"file_location,title,citation,doi\n{listed},{PDF_TITLE},{PDF_CITATION},\n", encoding="utf-8")

    with pytest.MonkeyPatch.context() as env:
        env.setenv("PAPER_DIR", str(papers))
        env.setenv("INDEX_DIR", str(root / "index"))
        env.setenv("MANIFEST_PATH", str(manifest))
        env.setenv("OPENAI_API_KEY", FAKE_API_KEY)
        env.setenv("PQA_INDEX_DONT_CACHE_INDEXES", "true")
        forget_project_modules()
        try:
            rebuild = importlib.import_module("pdf_research_assistant.rebuild")
            rebuild.settings.embedding = "sparse"
            yield root, rebuild
        finally:
            forget_project_modules()
            shutil.rmtree(root, ignore_errors=True)


async def rebuild_once(rebuild, seconds: float = REBUILD_SECONDS) -> str:
    """Run the real rebuild and return what it printed; fail instead of waiting past `seconds`."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        await asyncio.wait_for(rebuild.async_main(), seconds)
    return output.getvalue()


@pytest.fixture(scope="module")
def tiny_index():
    """Index the fixture PDF through the real rebuild entry point, in a throwaway folder."""
    with tiny_workspace() as (root, rebuild):
        exhaustive = importlib.import_module("pdf_research_assistant.exhaustive")
        # On Windows another program can briefly lock a new index file ("Access is denied").
        # The rebuild reports that and carries on when run again, as a user would run it again.
        for _attempt in range(REBUILD_ATTEMPTS):
            output = asyncio.run(rebuild_once(rebuild))
            if "Indexed after run: 1" in output:
                break
        settings = exhaustive.build_exhaustive_settings()
        settings.embedding = "sparse"
        yield SimpleNamespace(
            root=root,
            rebuild_output=output,
            index_dir=rebuild.get_active_index_dir(),
            exhaustive=exhaustive,
            settings=settings,
        )


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


def saved_doc(index_dir: Path, file_location: str):
    """Return the document PaperQA saved in the index for this PDF."""
    index_files = pickle.loads(zlib.decompress((index_dir / "files.zip").read_bytes()))
    saved = pickle.loads(zlib.decompress((index_dir / "docs" / f"{index_files[file_location]}.zip").read_bytes()))
    return next(iter(saved.docs.values()))


def test_the_rebuild_takes_the_citation_from_the_paper_list(tiny_index):
    doc = saved_doc(tiny_index.index_dir, PDF_NAME)

    assert doc.citation == PDF_CITATION
    assert doc.docname == "Tests2026"


def test_a_pdf_in_a_subfolder_takes_its_citation_from_the_paper_list_without_asking_the_model():
    # The fake API key makes any call to the answer model fail, so the PDF only indexes if PaperQA finds its row.
    with tiny_workspace(folder="Session 1") as (_root, rebuild):
        for _attempt in range(REBUILD_ATTEMPTS):
            output = asyncio.run(rebuild_once(rebuild))
            if "Indexed after run: 1" in output:
                break
        index_dir = rebuild.get_active_index_dir()

        assert "Failed PDFs: 0" in output
        doc = saved_doc(index_dir, str(Path("Session 1") / PDF_NAME))

    assert doc.citation == PDF_CITATION
    assert doc.docname == "Tests2026"


def test_a_rebuild_after_one_whose_index_write_failed_finishes(monkeypatch):
    """A failed write leaves PaperQA's index writer holding the folder's writer lock, which the next rebuild
    in the same process waited for, hours on end, until the rebuild let go of the writer (#134)."""
    from paperqa.agents.search import SearchDocumentStorage

    write_to_string = SearchDocumentStorage.write_to_string
    failures = {"left": 1}

    def denied_once(self, data):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError(13, "Access is denied")
        return write_to_string(self, data)

    monkeypatch.setattr(SearchDocumentStorage, "write_to_string", denied_once)
    with tiny_workspace() as (_root, rebuild):
        first = asyncio.run(rebuild_once(rebuild))
        # Windows can deny a write of its own now and then; each try must still finish, not wait on the lock.
        for _attempt in range(REBUILD_ATTEMPTS):
            later = asyncio.run(rebuild_once(rebuild))
            if "Indexed after run: 1" in later:
                break

    assert "ERROR: [Errno 13] Access is denied" in first
    assert "Indexed after run: 0" in first
    assert "Indexed after run: 1" in later
    assert "Failed PDFs: 0" in later


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


def test_pictures_are_drawn_at_the_apps_sharpness_when_paperqa_reads_a_pdf(tiny_index):
    """With images and tables on, PaperQA passes the app's picture sharpness through to its PDF reader."""
    from paperqa import Docs
    from paperqa.settings import MultimodalOptions

    settings = tiny_index.settings.model_copy(deep=True)
    settings.parsing.multimodal = MultimodalOptions.ON_WITHOUT_ENRICHMENT
    settings.parsing.reader_config["full_page"] = True
    docs = Docs()

    asyncio.run(docs.aadd(FIXTURE_PDF, citation=f"Tests (2026) {PDF_TITLE}", docname="tiny", settings=settings))

    pictures = [media.info for text in docs.texts for media in text.media]
    assert pictures
    dpi = settings.parsing.reader_config["dpi"]
    assert dpi == 150
    for picture in pictures:
        assert picture["bitmap_width"] == pytest.approx(picture["page_width"] * dpi / 72, abs=1)


FAST_REPLY = {"summary": QUOTE, "relevance_score": 9}


def fast_question_settings(tiny_index, monkeypatch):
    """Return ordinary question settings with the Fast preset, every model call canned to FAST_REPLY."""
    from aviary.core import ToolCall, ToolRequestMessage
    from lmi import LiteLLMModel

    from pdf_research_assistant import quality_presets

    async def finish(self, *, messages, tools, tool_choice, **_kwargs):
        """The last fixed step asks the model to call PaperQA's complete tool; answer it directly."""
        return ToolRequestMessage(tool_calls=[ToolCall.from_tool(tool_choice, has_successful_answer=True)])

    monkeypatch.setattr(LiteLLMModel, "select_tool", finish)
    settings = tiny_index.settings.model_copy(deep=True)
    # Ordinary question settings, not exhaustive search's.
    settings.answer.evidence_retrieval = True
    quality_presets.apply_preset(settings, "Fast")
    model_config = canned_model(FAST_REPLY).config
    settings.llm = settings.summary_llm = settings.agent.agent_llm = "gpt-4o-mini"
    settings.llm_config = settings.summary_llm_config = settings.agent.agent_llm_config = model_config
    return settings


def test_fast_preset_answers_from_the_pdf_with_fixed_search_steps(tiny_index, monkeypatch):
    """The Fast preset's fixed steps run through PaperQA's real agent, with every model call canned."""
    from paperqa import Docs
    from paperqa.agents.main import run_agent

    from pdf_research_assistant import quality_presets

    settings = fast_question_settings(tiny_index, monkeypatch)
    reply = FAST_REPLY
    docs = Docs()

    response = asyncio.run(run_agent(docs, QUESTION, settings, agent_type=settings.agent.agent_type))

    assert settings.agent.agent_type == quality_presets.FIXED_STEPS
    assert response.status.value == "success"
    assert response.session.answer == json.dumps(reply)
    assert [text.doc.docname for text in docs.texts]


def test_an_answered_question_is_kept_beside_the_index_and_found_again(tiny_index, monkeypatch):
    """The query process keeps a real PaperQA answer, which a search then finds with its passages."""
    from pdf_research_assistant import bootstrap, query_once, saved_answers

    settings = fast_question_settings(tiny_index, monkeypatch)
    monkeypatch.setattr(query_once, "build_settings", lambda: settings)

    payload = query_once.run_query_payload(QUESTION)
    found = saved_answers.search(settings.agent.index.index_directory, "enthusiasm")

    assert payload["ok"] is True
    assert "save_error" not in payload
    assert [answer["question"] for answer in found["answers"]] == [QUESTION]
    assert found["answers"][0]["contexts"] == payload["contexts"]
    assert found["answers"][0]["settings"] == payload["settings"]
    assert saved_answers.answers_folder(settings.agent.index.index_directory).parent == tiny_index.index_dir.parent
    # The kept answers sit beside the PDF index and are not counted as PDFs.
    assert bootstrap.get_indexed_doc_count(tiny_index.index_dir.parent) == 1


# The sidebar in real Streamlit, which keeps each control's value between runs as a browser does.

APP_PATH = PROJECT_ROOT / "src" / "pdf_research_assistant" / "app.py"


@pytest.fixture
def sidebar(tiny_index):
    """Load the real page against the tiny index, with no saved sidebar settings, and return its runner."""
    from streamlit.testing.v1 import AppTest

    from pdf_research_assistant import saved_settings

    saved_settings.SETTINGS_PATH.unlink(missing_ok=True)
    page = AppTest.from_file(str(APP_PATH), default_timeout=120)
    page.run()
    assert not page.exception
    yield page
    saved_settings.SETTINGS_PATH.unlink(missing_ok=True)


def control(controls, label: str):
    return next(item for item in controls if item.label == label)


def test_a_changed_number_of_passages_is_kept_and_shows_the_preset_as_custom(sidebar):
    from pdf_research_assistant import quality_presets, saved_settings

    control(sidebar.number_input, "Passages gathered").set_value(7).run()

    assert control(sidebar.number_input, "Passages gathered").value == 7
    assert control(sidebar.selectbox, "Quality preset").value == quality_presets.NO_PRESET
    assert saved_settings.load()["evidence_k"] == 7


def test_a_changed_answer_length_is_kept_after_picking_a_preset(sidebar):
    from pdf_research_assistant import quality_presets, saved_settings

    control(sidebar.selectbox, "Quality preset").set_value("Fast").run()
    long_answer = control(sidebar.selectbox, "Answer length").options[2]
    control(sidebar.selectbox, "Answer length").set_value(long_answer).run()

    assert "400 words" in long_answer
    assert "400 words" in control(sidebar.selectbox, "Answer length").value
    assert control(sidebar.selectbox, "Quality preset").value == quality_presets.NO_PRESET
    assert saved_settings.load()["quality_preset"] == "Fast"
    assert "400 words" in saved_settings.load()["answer_length"]


def test_a_preset_picked_after_changing_a_number_still_replaces_it(sidebar):
    from pdf_research_assistant import saved_settings

    control(sidebar.number_input, "Passages gathered").set_value(7).run()
    control(sidebar.selectbox, "Quality preset").set_value("High quality").run()

    assert control(sidebar.number_input, "Passages gathered").value == 20
    assert control(sidebar.selectbox, "Quality preset").value == "High quality"
    assert "evidence_k" not in saved_settings.load()
