import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
# Imported before the fixture swaps in a fake paperqa, so tests can build real PaperQA contexts.
from paperqa.docs import Docs
from paperqa.types import Context, Doc, ParsedMedia, PQASession, Text


def _stub_settings():
    return SimpleNamespace(agent=SimpleNamespace(agent_type="stub-agent", index=SimpleNamespace(index_directory="index-dir")))


@pytest.fixture
def query_once_module():
    paperqa_module = ModuleType("paperqa")
    agents_module = ModuleType("paperqa.agents")
    agents_main_module = ModuleType("paperqa.agents.main")
    utils_module = ModuleType("paperqa.utils")
    bootstrap_module = ModuleType("pdf_research_assistant.bootstrap")

    # Mock the PaperQA Docs constructor so importing query_once never needs the real dependency.
    # Like a real Docs, it starts with no chunks.
    paperqa_module.Docs = mock.Mock(name="Docs", side_effect=lambda: SimpleNamespace(texts=[]))

    # Mock PaperQA's async agent entry point so each test can control the query behavior explicitly.
    agents_main_module.run_agent = mock.Mock(name="run_agent")

    # Mock the utility that executes the coroutine so tests can simulate success and failure paths.
    utils_module.run_or_ensure = mock.Mock(name="run_or_ensure")

    # Mock project bootstrap wiring so query_once imports cleanly without requiring .env configuration.
    bootstrap_module.build_settings = mock.Mock(name="build_settings")

    with mock.patch.dict(
        sys.modules,
        {
            "paperqa": paperqa_module,
            "paperqa.agents": agents_module,
            "paperqa.agents.main": agents_main_module,
            "paperqa.utils": utils_module,
            "pdf_research_assistant.bootstrap": bootstrap_module,
        },
    ):
        sys.modules.pop("pdf_research_assistant.query_once", None)
        module = importlib.import_module("pdf_research_assistant.query_once")
        # Never write to a real index of past answers; tests that check saving look at this mock.
        with mock.patch.object(module.saved_answers, "save_answer", return_value=None):
            yield module
        sys.modules.pop("pdf_research_assistant.query_once", None)


def test_serialize_contexts_returns_empty_list_when_no_contexts_are_provided(query_once_module):
    assert query_once_module.serialize_contexts(None) == []


def test_serialize_contexts_uses_source_defaults_when_context_has_no_text_object(query_once_module):
    context = SimpleNamespace(context="Quoted passage")

    result = query_once_module.serialize_contexts([context])

    assert result == [
        {
            "id": "",
            "name": "Source",
            "cited": False,
            "context": "Quoted passage",
            "raw_text": "",
            "media": [],
            "score": -1,
        }
    ]


def test_serialize_contexts_records_the_relevance_score_paperqa_gave_the_chunk(query_once_module):
    context = SimpleNamespace(context="Quoted", score=8, text=SimpleNamespace(name="My Paper", text="Chunk"))

    result = query_once_module.serialize_contexts([context])

    assert result[0]["score"] == 8


def test_serialize_contexts_keeps_paperqa_unset_score_for_a_real_context(query_once_module):
    chunk = Text(text="Chunk", name="Paper p1", doc=Doc(docname="d", citation="c", dockey="k"))
    contexts = [Context(context="Unscored", text=chunk), Context(context="Scored", text=chunk, score=7)]

    result = query_once_module.serialize_contexts(contexts)

    assert [item["score"] for item in result] == [Context.UNSET_RELEVANCE, 7]


def test_serialize_contexts_uses_text_name_when_present_on_context_text_object(query_once_module):
    context = SimpleNamespace(
        context="Quoted passage",
        text=SimpleNamespace(name="My Paper", text="Chunk text"),
    )

    result = query_once_module.serialize_contexts([context])

    assert result[0]["name"] == "My Paper"


def test_serialize_contexts_uses_empty_string_when_context_string_is_missing(query_once_module):
    context = SimpleNamespace(text=SimpleNamespace(name="My Paper", text="Chunk text"))

    result = query_once_module.serialize_contexts([context])

    assert result[0]["context"] == ""


def test_serialize_contexts_uses_raw_text_from_context_text_object_when_present(query_once_module):
    context = SimpleNamespace(
        context="Quoted passage",
        text=SimpleNamespace(name="My Paper", text="Chunk text"),
    )

    result = query_once_module.serialize_contexts([context])

    assert result[0]["raw_text"] == "Chunk text"


def test_serialize_contexts_uses_empty_raw_text_when_context_text_object_is_missing(query_once_module):
    context = SimpleNamespace(context="Quoted passage")

    result = query_once_module.serialize_contexts([context])

    assert result[0]["raw_text"] == ""


def test_serialize_contexts_records_the_context_id_and_whether_the_answer_cited_it(query_once_module):
    chunk = SimpleNamespace(name="My Paper", text="Chunk text")
    contexts = [
        SimpleNamespace(id="pqac-1", context="Used", text=chunk),
        SimpleNamespace(id="pqac-2", context="Unused", text=chunk),
    ]

    result = query_once_module.serialize_contexts(contexts, cited_ids={"pqac-1"})

    assert [(item["id"], item["cited"]) for item in result] == [("pqac-1", True), ("pqac-2", False)]


def test_serialize_contexts_includes_the_media_stored_with_the_chunk(query_once_module):
    media = SimpleNamespace(index=0, data=b"image", url=None, text=None, info={})
    context = SimpleNamespace(context="Quoted", text=SimpleNamespace(name="My Paper", text="Chunk", media=[media]))

    result = query_once_module.serialize_contexts([context])

    assert result[0]["media"] == [query_once_module.serialize_media(media)]


def test_serialize_contexts_gives_an_empty_media_list_when_the_chunk_has_none(query_once_module):
    with_text = SimpleNamespace(context="Quoted", text=SimpleNamespace(name="My Paper", text="Chunk", media=None))
    without_text = SimpleNamespace(context="Quoted")

    result = query_once_module.serialize_contexts([with_text, without_text])

    assert [item["media"] for item in result] == [[], []]


def stripped_session_and_docs(*chunks):
    """Build real PaperQA passages and contexts, then empty the contexts the way AnswerResponse does."""
    docs = Docs()
    docs.texts = list(chunks)
    session = PQASession(
        question="Q",
        contexts=[Context(id=f"pqac-{n}", context=f"Summary {n}", text=chunk) for n, chunk in enumerate(chunks)],
    )
    session.filter_content_for_user()
    return session, docs


def test_serialize_contexts_takes_chunk_text_and_media_from_the_passages_the_question_read(query_once_module):
    figure = ParsedMedia(index=1, data=b"png", info={"type": "picture", "page_num": 4})
    chunk = Text(
        text="Full chunk text", name="Paper pages 3-4", media=[figure], doc=Doc(docname="d", citation="c", dockey="k")
    )
    session, docs = stripped_session_and_docs(chunk)
    assert session.contexts[0].text.text == ""
    assert not session.contexts[0].text.media

    result = query_once_module.serialize_contexts(session.contexts, passages=docs.texts)

    assert result[0]["raw_text"] == "Full chunk text"
    assert result[0]["media"] == [query_once_module.serialize_media(figure)]
    assert result[0]["context"] == "Summary 0"


def test_serialize_contexts_matches_passages_by_document_as_well_as_name(query_once_module):
    wanted = Text(text="Wanted", name="Paper pages 1-2", doc=Doc(docname="d", citation="c", dockey="k1"))
    same_name = Text(text="Other paper", name="Paper pages 1-2", doc=Doc(docname="d", citation="c", dockey="k2"))
    session, _ = stripped_session_and_docs(wanted)

    result = query_once_module.serialize_contexts(session.contexts, passages=[same_name, wanted])

    assert result[0]["raw_text"] == "Wanted"


def test_serialize_contexts_keeps_the_context_text_when_no_passage_matches(query_once_module):
    chunk = Text(text="Chunk", name="Paper pages 1-2", doc=Doc(docname="d", citation="c", dockey="k"))
    session, _ = stripped_session_and_docs(chunk)
    elsewhere = Text(text="Other", name="Paper pages 1-2", doc=Doc(docname="d", citation="c", dockey="other"))

    result = query_once_module.serialize_contexts(session.contexts, passages=[elsewhere])

    assert result[0]["raw_text"] == ""
    assert result[0]["media"] == []
    assert result[0]["name"] == "Paper pages 1-2"


def test_serialize_contexts_uses_source_defaults_when_context_has_no_text_even_with_passages(query_once_module):
    passage = Text(text="Chunk", name="Source", doc=Doc(docname="d", citation="c", dockey="k"))

    result = query_once_module.serialize_contexts([SimpleNamespace(context="Quoted")], passages=[passage])

    assert (result[0]["name"], result[0]["raw_text"], result[0]["media"]) == ("Source", "", [])


def test_serialize_media_turns_image_bytes_and_details_into_json_safe_values(query_once_module):
    media = SimpleNamespace(
        index=2,
        data=b"image",
        url=None,
        text="| a | b |",
        info={"type": "table", "page_num": 3, "enriched_description": "A table of prices"},
    )

    result = query_once_module.serialize_media(media)

    assert result == {
        "index": 2,
        "kind": "table",
        "page": 3,
        "text": "| a | b |",
        "description": "A table of prices",
        "image": "aW1hZ2U=",
        "url": "",
    }
    json.dumps(result)


def test_serialize_media_keeps_a_link_when_the_image_is_stored_elsewhere(query_once_module):
    media = SimpleNamespace(index=0, data=b"", url="https://example.org/figure.png", text=None, info={})

    result = query_once_module.serialize_media(media)

    assert result == {
        "index": 0,
        "kind": None,
        "page": None,
        "text": "",
        "description": "",
        "image": "",
        "url": "https://example.org/figure.png",
    }


def test_serialize_media_uses_defaults_when_details_are_missing(query_once_module):
    result = query_once_module.serialize_media(SimpleNamespace())

    assert result == {"index": 0, "kind": None, "page": None, "text": "", "description": "", "image": "", "url": ""}


def test_run_query_payload_returns_success_payload_for_completed_query(query_once_module):
    settings = _stub_settings()
    response = SimpleNamespace(
        session=SimpleNamespace(
            formatted_answer="Answer text",
            cost=1.25,
            contexts=["ctx-1"],
            used_contexts={"pqac-1"},
        )
    )

    # Mock settings construction so the test controls the agent type used by run_query_payload.
    query_once_module.build_settings = mock.Mock(return_value=settings)

    # Mock Docs creation so the function can pass a predictable docs object into run_agent.
    # PaperQA fills it with the passages the question read.
    docs_instance = SimpleNamespace(texts=["text-1"])
    query_once_module.Docs = mock.Mock(return_value=docs_instance)

    # Mock run_agent so we can verify the exact coroutine inputs without invoking PaperQA.
    fake_coro = object()
    query_once_module.run_agent = mock.Mock(return_value=fake_coro)

    # Mock run_or_ensure to short-circuit coroutine execution and hand back a canned response object.
    query_once_module.run_or_ensure = mock.Mock(return_value=response)

    # Mock context serialization so this test stays focused on the success flow orchestration.
    query_once_module.serialize_contexts = mock.Mock(return_value=[{"name": "Source"}])

    # Mock the settings record so this test only checks that it comes from the settings the query used.
    recorded = {"index": "pqa_index_x", "values": []}
    record_settings = mock.patch.object(
        query_once_module.settings_groups, "record_answer_settings", return_value=recorded
    )

    with record_settings as record_answer_settings:
        result = query_once_module.run_query_payload("What is the answer?")

    assert result == {
        "ok": True,
        "answer": "Answer text",
        "cost": 1.25,
        "contexts": [{"name": "Source"}],
        "settings": recorded,
    }
    record_answer_settings.assert_called_once_with(settings, one_off=[])
    query_once_module.build_settings.assert_called_once_with()
    query_once_module.Docs.assert_called_once_with()
    query_once_module.run_agent.assert_called_once_with(
        docs_instance,
        "What is the answer?",
        settings,
        agent_type="stub-agent",
    )
    query_once_module.run_or_ensure.assert_called_once_with(coro=fake_coro)
    query_once_module.serialize_contexts.assert_called_once_with(
        ["ctx-1"], cited_ids={"pqac-1"}, passages=["text-1"]
    )


def test_run_query_payload_keeps_the_answer_with_its_passages_and_settings_for_searching(query_once_module):
    settings = _stub_settings()
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="A", cost=0.0, contexts=[]))
    query_once_module.build_settings = mock.Mock(return_value=settings)
    query_once_module.run_or_ensure = mock.Mock(return_value=response)
    query_once_module.serialize_contexts = mock.Mock(return_value=[{"name": "Source"}])
    recorded = {"index": "pqa_index_x", "values": []}

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings", return_value=recorded),             mock.patch.object(query_once_module.saved_answers, "build_record", return_value={"r": 1}) as build_record:
        result = query_once_module.run_query_payload("Q")

    build_record.assert_called_once_with([{"name": "Source"}], recorded)
    query_once_module.saved_answers.save_answer.assert_called_once_with("index-dir", response, {"r": 1})
    assert "save_error" not in result


def test_run_query_payload_still_answers_and_says_why_when_the_answer_could_not_be_kept(query_once_module):
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="A", cost=0.5, contexts=[]))
    query_once_module.build_settings = mock.Mock(return_value=_stub_settings())
    query_once_module.run_or_ensure = mock.Mock(return_value=response)
    query_once_module.saved_answers.save_answer.return_value = "OSError: disk full"

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings", return_value=None):
        result = query_once_module.run_query_payload("Q")

    assert result["ok"] is True
    assert result["answer"] == "A"
    assert result["save_error"] == "OSError: disk full"


def test_answer_text_shows_the_answer_when_paperqa_wrote_one(query_once_module):
    session = SimpleNamespace(formatted_answer="Question: Q\n\nAn answer (Moon2004 pages 1-2).", answer="An answer")

    assert query_once_module.answer_text(session) == "Question: Q\n\nAn answer (Moon2004 pages 1-2)."


def test_answer_text_explains_a_missing_answer_instead_of_showing_nothing(query_once_module):
    # What PaperQA 2026.3.18 leaves when the agent stops before gen_answer, for example after timeouts.
    session = SimpleNamespace(formatted_answer="", answer="No answer generated.")

    text = query_once_module.answer_text(session)

    assert text.startswith("No answer was written")
    assert '"No answer generated."' in text
    assert "ask it another way" in text


def test_answer_text_still_explains_when_paperqa_gives_no_reason(query_once_module):
    session = SimpleNamespace(formatted_answer="   ", answer=None)

    assert '"No answer generated."' in query_once_module.answer_text(session)


def test_run_query_payload_never_returns_a_blank_answer(query_once_module):
    response = SimpleNamespace(
        session=SimpleNamespace(formatted_answer="", answer="No answer generated.", cost=0.0, contexts=[])
    )
    query_once_module.build_settings = mock.Mock(return_value=_stub_settings())
    query_once_module.run_or_ensure = mock.Mock(return_value=response)

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings", return_value=None):
        result = query_once_module.run_query_payload("Q")

    assert result["ok"] is True
    assert result["answer"].startswith("No answer was written")


def test_run_query_payload_keeps_nothing_when_the_question_fails(query_once_module):
    query_once_module.build_settings = mock.Mock(return_value=_stub_settings())
    query_once_module.run_or_ensure = mock.Mock(side_effect=RuntimeError("boom"))

    query_once_module.run_query_payload("Q")

    query_once_module.saved_answers.save_answer.assert_not_called()


def test_run_query_payload_marks_nothing_as_cited_when_the_session_does_not_say(query_once_module):
    settings = _stub_settings()
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="Answer text", cost=0.0, contexts=["ctx-1"]))
    query_once_module.build_settings = mock.Mock(return_value=settings)
    query_once_module.run_or_ensure = mock.Mock(return_value=response)
    query_once_module.serialize_contexts = mock.Mock(return_value=[])

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings"):
        query_once_module.run_query_payload("What is the answer?")

    query_once_module.serialize_contexts.assert_called_once_with(
        ["ctx-1"], cited_ids=(), passages=[]
    )


def test_run_query_payload_returns_error_payload_when_query_execution_raises(query_once_module):
    settings = _stub_settings()

    # Mock settings construction so the function gets far enough to exercise the error handler.
    query_once_module.build_settings = mock.Mock(return_value=settings)

    # Mock Docs so query construction itself is not the reason for the failure.
    query_once_module.Docs = mock.Mock(return_value=object())

    # Mock run_agent to provide a placeholder coroutine object for run_or_ensure.
    query_once_module.run_agent = mock.Mock(return_value=object())

    # Mock run_or_ensure to raise the application error that should be converted into an error payload.
    query_once_module.run_or_ensure = mock.Mock(side_effect=RuntimeError("boom"))

    result = query_once_module.run_query_payload("What failed?")

    assert result == {
        "ok": False,
        "error_type": "RuntimeError",
        "error": "boom",
    }


def test_main_returns_exit_code_1_when_question_argument_count_is_incorrect(query_once_module):
    # Mock argv so main sees the wrong number of arguments and takes the validation branch.
    with mock.patch.object(query_once_module.sys, "argv", ["query_once.py"]):
        # Mock print so the test can assert on the emitted JSON error payload.
        with mock.patch("builtins.print") as print_mock:
            exit_code = query_once_module.main()

    assert exit_code == 1
    print_mock.assert_called_once_with(json.dumps({"ok": False, "error": query_once_module.USAGE_ERROR}))


@pytest.mark.parametrize("one_off", ["not json", "[1, 2]"])
def test_main_returns_exit_code_1_when_the_one_off_settings_are_not_a_json_object(one_off, query_once_module):
    with mock.patch.object(query_once_module.sys, "argv", ["query_once.py", "What is new?", one_off]):
        with mock.patch.object(query_once_module, "run_query_payload") as payload_mock:
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_once_module.main()

    assert exit_code == 1
    payload_mock.assert_not_called()
    print_mock.assert_called_once_with(json.dumps({"ok": False, "error": query_once_module.USAGE_ERROR}))


def test_main_passes_one_off_settings_given_after_the_question(query_once_module):
    payload = {"ok": True}
    argv = ["query_once.py", "What is new?", json.dumps({"evidence_k": 3})]
    with mock.patch.object(query_once_module.sys, "argv", argv):
        with mock.patch.object(query_once_module, "run_query_payload", return_value=payload) as payload_mock:
            with mock.patch("builtins.print"):
                exit_code = query_once_module.main()

    assert exit_code == 0
    payload_mock.assert_called_once_with("What is new?", {"evidence_k": 3})


def test_run_query_payload_applies_one_off_settings_after_the_saved_ones_and_records_them(query_once_module):
    settings = _stub_settings()
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="A", cost=0.0, contexts=[]))
    query_once_module.run_or_ensure = mock.Mock(return_value=response)
    steps = mock.Mock()
    steps.build.return_value = settings
    steps.apply_usable.return_value = ["evidence_k"]
    # Record the order: settings are built, then the one-off values applied, then the question run.
    query_once_module.build_settings = steps.build
    query_once_module.run_agent = mock.Mock(side_effect=lambda *args, **kwargs: steps.run())

    with mock.patch.multiple(
        query_once_module.settings_groups,
        apply_usable=steps.apply_usable,
        record_answer_settings=steps.record,
    ):
        query_once_module.run_query_payload("Q", {"evidence_k": 3, "evidence_x": 1})

    assert steps.mock_calls[:3] == [
        mock.call.build(),
        mock.call.apply_usable(settings, {"evidence_k": 3, "evidence_x": 1}),
        mock.call.run(),
    ]
    steps.record.assert_called_once_with(settings, one_off=["evidence_k"])


def test_run_query_payload_records_no_one_off_settings_when_none_are_given(query_once_module):
    settings = _stub_settings()
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="A", cost=0.0, contexts=[]))
    query_once_module.build_settings = mock.Mock(return_value=settings)
    query_once_module.run_or_ensure = mock.Mock(return_value=response)

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings") as record:
        query_once_module.run_query_payload("Q")

    record.assert_called_once_with(settings, one_off=[])


def test_main_returns_exit_code_0_and_prints_serialized_payload_for_single_question_argument(query_once_module):
    payload = {"ok": True, "answer": "Answer text", "cost": 0.5, "contexts": []}

    # Mock argv so main follows the happy path with exactly one question argument.
    with mock.patch.object(query_once_module.sys, "argv", ["query_once.py", "What is new?"]):
        # Mock run_query_payload so this test verifies CLI wiring rather than query execution details.
        with mock.patch.object(query_once_module, "run_query_payload", return_value=payload) as payload_mock:
            # Mock print so the test can assert that main writes the JSON contract to stdout.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_once_module.main()

    assert exit_code == 0
    payload_mock.assert_called_once_with("What is new?", None)
    print_mock.assert_called_once_with(json.dumps(payload, ensure_ascii=False))
