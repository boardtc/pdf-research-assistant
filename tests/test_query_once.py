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
    return SimpleNamespace(
        agent=SimpleNamespace(agent_type="stub-agent", timeout=900, index=SimpleNamespace(index_directory="index-dir"))
    )


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
        # Never change the wait on the real shared rate limiter; the test that checks it looks at this mock.
        # Never download LiteLLM's price list; patching it here also undoes any reuse_model_cost_map wrapper.
        with mock.patch.object(module.saved_answers, "save_answer", return_value=None), mock.patch.object(
            module.rate_profiles, "allow_limiter_wait"
        ), mock.patch.object(module.litellm, "get_model_cost_map", return_value={}):
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


def test_question_cost_takes_the_larger_of_the_session_and_tracked_costs(query_once_module):
    # A local model: the session counts nothing, the tracker counts the OpenAI embeddings.
    assert query_once_module.question_cost(0.0, 0.00002) == 0.00002
    # The tracker saw nothing (for example in a test): the session's own figure is kept.
    assert query_once_module.question_cost(0.0123, 0.0) == 0.0123
    assert query_once_module.question_cost(None, None) == 0.0


def _run_with_tracked_cost(query_once_module, session_cost, tracked_cost):
    """Run a question whose PaperQA run adds tracked_cost to a fake cost tracker."""
    tracker = SimpleNamespace(lifetime_cost_usd=1.5)
    response = SimpleNamespace(
        session=SimpleNamespace(formatted_answer="A", answer="A", cost=session_cost, contexts=[])
    )

    def run(coro):
        tracker.lifetime_cost_usd += tracked_cost
        return response

    query_once_module.build_settings = mock.Mock(return_value=_stub_settings())
    query_once_module.run_or_ensure = mock.Mock(side_effect=run)
    with mock.patch.object(query_once_module, "GLOBAL_COST_TRACKER", tracker), \
            mock.patch.object(query_once_module, "enable_cost_tracking") as enable, \
            mock.patch.object(query_once_module.settings_groups, "record_answer_settings", return_value=None):
        result = query_once_module.run_query_payload("Q")
    return result, response, enable


def test_run_query_payload_counts_the_embedding_cost_paperqa_leaves_out(query_once_module):
    result, response, enable = _run_with_tracked_cost(query_once_module, session_cost=0.0, tracked_cost=0.00002)

    enable.assert_called_once_with()
    assert result["cost"] == pytest.approx(0.00002)
    # The saved answer reads the session's cost, so it shows the same figure as the chat.
    assert response.session.cost == pytest.approx(0.00002)
    query_once_module.saved_answers.save_answer.assert_called_once()
    assert query_once_module.saved_answers.save_answer.call_args.args[1] is response


def test_run_query_payload_keeps_the_session_cost_when_the_tracker_saw_nothing(query_once_module):
    result, response, _ = _run_with_tracked_cost(query_once_module, session_cost=0.0123, tracked_cost=0.0)

    assert result["cost"] == 0.0123
    assert response.session.cost == 0.0123


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
        "cost": 0.0,
    }


def _fail_after_tracked_cost(query_once_module, tracked_cost, build_error=None):
    """Run a question that fails after its PaperQA run adds tracked_cost to a fake cost tracker."""
    tracker = SimpleNamespace(lifetime_cost_usd=1.5)

    def run(coro):
        tracker.lifetime_cost_usd += tracked_cost
        raise RuntimeError("rate limited")

    query_once_module.build_settings = mock.Mock(return_value=_stub_settings(), side_effect=build_error)
    query_once_module.run_or_ensure = mock.Mock(side_effect=run)
    with mock.patch.object(query_once_module, "GLOBAL_COST_TRACKER", tracker),             mock.patch.object(query_once_module, "enable_cost_tracking"):
        return query_once_module.run_query_payload("Q")


def test_run_query_payload_reports_what_a_failed_question_spent_before_failing(query_once_module):
    result = _fail_after_tracked_cost(query_once_module, tracked_cost=0.0042)

    assert result["ok"] is False
    assert result["error"] == "rate limited"
    assert result["cost"] == pytest.approx(0.0042)


def test_run_query_payload_reports_no_cost_when_the_question_fails_before_tracking_starts(query_once_module):
    result = _fail_after_tracked_cost(query_once_module, tracked_cost=0.0042, build_error=ValueError("bad .env"))

    assert result == {"ok": False, "error_type": "ValueError", "error": "bad .env", "cost": 0.0}
    query_once_module.run_or_ensure.assert_not_called()


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


def test_run_query_payload_lets_requests_wait_for_the_rate_limiter_before_running_the_question(query_once_module):
    settings = _stub_settings()
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="A", cost=0.0, contexts=[]))
    query_once_module.run_or_ensure = mock.Mock(return_value=response)
    steps = mock.Mock()
    steps.build.return_value = settings
    steps.apply_usable.return_value = []
    query_once_module.build_settings = steps.build
    query_once_module.run_agent = mock.Mock(side_effect=lambda *args, **kwargs: steps.run())

    # The wait is the time the question may run, from its settings with one-off values included.
    with mock.patch.object(query_once_module.settings_groups, "apply_usable", steps.apply_usable), mock.patch.object(
        query_once_module.rate_profiles, "allow_limiter_wait", steps.allow_wait
    ):
        query_once_module.run_query_payload("Q", {"evidence_k": 3})

    assert steps.mock_calls[:4] == [
        mock.call.build(),
        mock.call.apply_usable(settings, {"evidence_k": 3}),
        mock.call.allow_wait(900),
        mock.call.run(),
    ]


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


def test_main_writes_utf8_even_when_stdout_uses_a_windows_code_page(query_once_module):
    """A piped stdout on Windows uses the console code page, but the app and the CLI read UTF-8."""
    import io

    # A curly apostrophe is in cp1252 but as a different byte; a non-breaking hyphen is not in it at all.
    payload = {"ok": True, "answer": "A teacher’s passion is long‑lasting.", "cost": 0.1, "contexts": []}
    raw = io.BytesIO()
    stdout = io.TextIOWrapper(raw, encoding="cp1252")
    with mock.patch.object(query_once_module.sys, "argv", ["query_once.py", "What is new?"]):
        with mock.patch.object(query_once_module, "run_query_payload", return_value=payload):
            with mock.patch.object(query_once_module.sys, "stdout", stdout):
                exit_code = query_once_module.main()
    stdout.flush()

    assert exit_code == 0
    assert json.loads(raw.getvalue().decode("utf-8")) == payload


def test_utf8_output_is_left_alone_on_a_stream_that_cannot_be_reconfigured(query_once_module):
    stream = SimpleNamespace()

    query_once_module.use_utf8_output(stream)

    assert vars(stream) == {}


def test_utf8_output_switches_a_stream_to_utf8(query_once_module):
    stream = mock.Mock()

    query_once_module.use_utf8_output(stream)

    stream.reconfigure.assert_called_once_with(encoding="utf-8")


def test_run_query_payload_searches_the_index_as_it_is_without_indexing_new_pdfs(query_once_module):
    """New PDFs are indexed by Rebuild the index, not in the middle of a question."""
    settings = _stub_settings()
    settings.agent.rebuild_index = True
    seen = {}
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="A", cost=0.0, contexts=[]))
    query_once_module.build_settings = mock.Mock(return_value=settings)
    query_once_module.run_agent = mock.Mock(
        side_effect=lambda _docs, _question, used, **_kwargs: seen.update(rebuild_index=used.agent.rebuild_index)
    )
    query_once_module.run_or_ensure = mock.Mock(return_value=response)

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings", return_value={}):
        query_once_module.run_query_payload("Q")

    assert seen == {"rebuild_index": False}


def _run_waiting_main(module, stdin_text, payload=None):
    import io

    with mock.patch.object(module.sys, "argv", ["query_once.py", module.WAIT_FLAG]), \
            mock.patch.object(module.sys, "stdin", io.StringIO(stdin_text)), \
            mock.patch.object(module, "run_query_payload", return_value=payload or {"ok": True}) as payload_mock, \
            mock.patch("builtins.print") as print_mock:
        exit_code = module.main()
    return exit_code, payload_mock, print_mock


def test_main_waits_for_a_question_and_its_one_off_settings_on_stdin(query_once_module):
    payload = {"ok": True, "answer": "Answer text", "cost": 0.5, "contexts": []}
    request = json.dumps({"question": "What is new?", "one_off": {"evidence_k": 3}}) + "\n"

    exit_code, payload_mock, print_mock = _run_waiting_main(query_once_module, request, payload)

    assert exit_code == 0
    payload_mock.assert_called_once_with("What is new?", {"evidence_k": 3})
    print_mock.assert_called_once_with(json.dumps(payload, ensure_ascii=False))


def test_main_waiting_for_a_question_accepts_no_one_off_settings(query_once_module):
    request = json.dumps({"question": "What is new?", "one_off": None}) + "\n"

    exit_code, payload_mock, _print_mock = _run_waiting_main(query_once_module, request)

    assert exit_code == 0
    payload_mock.assert_called_once_with("What is new?", None)


def test_main_exits_quietly_when_stdin_closes_before_a_question_arrives(query_once_module):
    """The app or the CLI closed, so the spare process has nothing to do."""
    exit_code, payload_mock, print_mock = _run_waiting_main(query_once_module, "")

    assert exit_code == 0
    payload_mock.assert_not_called()
    print_mock.assert_not_called()


@pytest.mark.parametrize(
    "request_line",
    [
        "not json\n",
        "[1, 2]\n",
        json.dumps({"one_off": None}) + "\n",
        json.dumps({"question": 3, "one_off": None}) + "\n",
        json.dumps({"question": "Q", "one_off": [1]}) + "\n",
    ],
    ids=["not-json", "not-an-object", "no-question", "question-not-text", "one-off-not-an-object"],
)
def test_main_rejects_a_waited_request_that_is_not_a_question(query_once_module, request_line):
    exit_code, payload_mock, print_mock = _run_waiting_main(query_once_module, request_line)

    assert exit_code == 1
    payload_mock.assert_not_called()
    print_mock.assert_called_once_with(json.dumps({"ok": False, "error": query_once_module.USAGE_ERROR}))


def test_main_loads_the_model_clients_before_waiting_for_a_question(query_once_module):
    """LiteLLM imports the OpenAI client only on its first call, so the spare process does that while it waits."""
    order = []
    request = json.dumps({"question": "Q", "one_off": None}) + "\n"

    class Stdin:
        def readline(self):
            order.append("question")
            return request

    with mock.patch.object(query_once_module, "preload_model_clients", side_effect=lambda: order.append("preload")), \
            mock.patch.object(query_once_module.sys, "argv", ["query_once.py", query_once_module.WAIT_FLAG]), \
            mock.patch.object(query_once_module.sys, "stdin", Stdin()), \
            mock.patch.object(query_once_module, "run_query_payload", return_value={"ok": True}), \
            mock.patch("builtins.print"):
        query_once_module.main()

    assert order == ["preload", "question"]


def test_main_with_the_question_on_the_command_line_loads_nothing_ahead(query_once_module):
    with mock.patch.object(query_once_module, "preload_model_clients") as preload, \
            mock.patch.object(query_once_module.sys, "argv", ["query_once.py", "Q"]), \
            mock.patch.object(query_once_module, "run_query_payload", return_value={"ok": True}), \
            mock.patch("builtins.print"):
        query_once_module.main()

    preload.assert_not_called()


def test_preload_model_clients_imports_each_client_module(query_once_module):
    with mock.patch.object(query_once_module.importlib, "import_module") as import_module,             mock.patch.object(query_once_module.litellm, "get_model_cost_map"):
        query_once_module.preload_model_clients()

    assert [call.args[0] for call in import_module.call_args_list] == list(query_once_module.MODEL_CLIENT_MODULES)


def test_preload_model_clients_skips_a_module_that_a_newer_library_has_moved(query_once_module):
    def import_module(name):
        if name == query_once_module.MODEL_CLIENT_MODULES[0]:
            raise ImportError(name)

    with mock.patch.object(query_once_module.importlib, "import_module", side_effect=import_module) as imported,             mock.patch.object(query_once_module.litellm, "get_model_cost_map"):
        query_once_module.preload_model_clients()

    assert imported.call_count == len(query_once_module.MODEL_CLIENT_MODULES)


def test_the_model_client_modules_exist_in_the_installed_libraries(query_once_module):
    import importlib.util

    for name in query_once_module.MODEL_CLIENT_MODULES:
        assert importlib.util.find_spec(name) is not None, name


def test_preload_model_clients_fetches_the_price_list_the_embedding_model_reads(query_once_module):
    with mock.patch.object(query_once_module.importlib, "import_module"),             mock.patch.object(query_once_module.litellm, "get_model_cost_map") as fetch:
        query_once_module.preload_model_clients()

    fetch.assert_called_once_with(url=query_once_module.EMBEDDING_PRICE_LIST_URL)


def test_the_price_list_url_is_the_one_paperqas_embedding_model_fetches(query_once_module):
    import inspect

    from lmi import embeddings

    assert query_once_module.EMBEDDING_PRICE_LIST_URL in inspect.getsource(embeddings)


def _fake_fetch():
    return mock.Mock(side_effect=lambda url: {"url": url})


def test_reuse_model_cost_map_fetches_each_price_list_once_per_process(query_once_module):
    fetch = _fake_fetch()
    with mock.patch.object(query_once_module.litellm, "get_model_cost_map", fetch):
        query_once_module.reuse_model_cost_map()
        first = query_once_module.litellm.get_model_cost_map("u1")
        again = query_once_module.litellm.get_model_cost_map(url="u1")
        other = query_once_module.litellm.get_model_cost_map("u2")

    assert first is again
    assert other == {"url": "u2"}
    assert [call.args[0] for call in fetch.call_args_list] == ["u1", "u2"]


def test_reuse_model_cost_map_twice_still_fetches_only_once(query_once_module):
    fetch = _fake_fetch()
    with mock.patch.object(query_once_module.litellm, "get_model_cost_map", fetch):
        query_once_module.reuse_model_cost_map()
        query_once_module.reuse_model_cost_map()
        query_once_module.litellm.get_model_cost_map("u1")
        query_once_module.litellm.get_model_cost_map("u1")

    fetch.assert_called_once_with("u1")


@pytest.mark.parametrize("argv_tail", [["Q"], ["--wait"]], ids=["command-line", "waiting"])
def test_main_reuses_the_price_list_however_the_question_arrives(query_once_module, argv_tail):
    import io

    request = json.dumps({"question": "Q", "one_off": None}) + "\n"
    with mock.patch.object(query_once_module, "reuse_model_cost_map") as reuse, \
            mock.patch.object(query_once_module, "preload_model_clients"), \
            mock.patch.object(query_once_module.sys, "argv", ["query_once.py", *argv_tail]), \
            mock.patch.object(query_once_module.sys, "stdin", io.StringIO(request)), \
            mock.patch.object(query_once_module, "run_query_payload", return_value={"ok": True}), \
            mock.patch("builtins.print"):
        query_once_module.main()

    reuse.assert_called_once_with()


MADE_UP = "the need to foster intrinsic motivators rather than depend solely on grades"
REAL = "grades reduce interest in the learning itself"
LANG_CHUNK = "Kohn argues that grades reduce interest in the\nlearning itself."


def _quoted_session(answer, formatted=None):
    return SimpleNamespace(
        answer=answer,
        formatted_answer=f"Question: Q\n\n{answer}\n\nReferences\n\n1. (Lang2021 chunk 1): Lang"
        if formatted is None
        else formatted,
        cost=0.0,
        contexts=[],
    )


def test_run_query_payload_unquotes_a_made_up_quote_in_the_answer_shown_and_the_answer_kept(query_once_module):
    session = _quoted_session(f'Lang stresses "{MADE_UP}" (Lang2021 chunk 1) but quotes "{REAL}".')
    response = SimpleNamespace(session=session)
    query_once_module.build_settings = mock.Mock(return_value=_stub_settings())
    query_once_module.Docs = mock.Mock(return_value=SimpleNamespace(texts=[SimpleNamespace(text=LANG_CHUNK)]))
    query_once_module.run_or_ensure = mock.Mock(return_value=response)

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings", return_value=None):
        result = query_once_module.run_query_payload("Q")

    mark = query_once_module.quote_check.PARAPHRASE_MARK
    expected = f'Lang stresses {MADE_UP} {mark} (Lang2021 chunk 1) but quotes "{REAL}".'
    assert session.answer == expected
    assert session.formatted_answer == f"Question: Q\n\n{expected}\n\nReferences\n\n1. (Lang2021 chunk 1): Lang"
    assert result["answer"] == session.formatted_answer
    # The answer is kept after the check, so a saved answer shows the same text as the chat.
    assert query_once_module.saved_answers.save_answer.call_args.args[1] is response


def test_check_quotes_leaves_an_answer_whose_quotes_are_all_found_as_it_is(query_once_module):
    session = _quoted_session(f'Lang quotes "{REAL}".')
    before = session.formatted_answer

    query_once_module.check_quotes(session, [SimpleNamespace(text=LANG_CHUNK)])

    assert session.answer == f'Lang quotes "{REAL}".'
    assert session.formatted_answer == before


def test_check_quotes_does_nothing_when_paperqa_wrote_no_answer(query_once_module):
    session = SimpleNamespace(formatted_answer="")

    query_once_module.check_quotes(session, [])

    assert session.formatted_answer == ""
    assert not hasattr(session, "answer")


def test_check_quotes_still_unquotes_the_answer_when_the_formatted_answer_does_not_contain_it(query_once_module):
    session = _quoted_session(f'Lang stresses "{MADE_UP}".', formatted="Something PaperQA formatted differently")

    query_once_module.check_quotes(session, [])

    assert session.answer == f"Lang stresses {MADE_UP} {query_once_module.quote_check.PARAPHRASE_MARK}."
    assert session.formatted_answer == "Something PaperQA formatted differently"


def test_check_quotes_treats_a_passage_without_text_as_empty(query_once_module):
    session = _quoted_session(f'Lang quotes "{REAL}".')

    query_once_module.check_quotes(session, ["not a passage", SimpleNamespace(text=None)])

    assert query_once_module.quote_check.PARAPHRASE_MARK in session.answer


def test_check_quotes_reads_the_full_chunks_that_paperqas_answer_has_emptied(query_once_module):
    chunk = Text(text=LANG_CHUNK, name="Lang pages 1-2", doc=Doc(docname="d", citation="c", dockey="k"))
    session, docs = stripped_session_and_docs(chunk)
    session.answer = f'Lang quotes "{REAL}" (pqac-0).'
    session.formatted_answer = f"Question: Q\n\n{session.answer}"
    assert session.contexts[0].text.text == ""

    query_once_module.check_quotes(session, docs.texts)

    assert session.answer == f'Lang quotes "{REAL}" (pqac-0).'


LANG_ID = "pqac-0ed33f3f"
OTHER_ID = "pqac-7a1b2c3d"


def _cited_session(raw_answer, *, ids=(LANG_ID, OTHER_ID), citation="Lang, Small Teaching, 2021."):
    """Build a real answered PaperQA session whose contexts are emptied the way AnswerResponse does it."""
    chunks = [
        Text(
            text=f"Chunk {n}",
            name=f"Lang2021 pages {n}-{n + 1}",
            doc=Doc(docname="Lang2021", citation=citation, dockey="lang"),
        )
        for n in range(len(ids))
    ]
    docs = Docs()
    docs.texts = chunks
    session = PQASession(
        question="Q",
        raw_answer=raw_answer,
        contexts=[Context(id=context_id, context="Summary", text=chunk) for context_id, chunk in zip(ids, chunks)],
    )
    session.populate_formatted_answers_and_bib_from_raw_answer()
    session.filter_content_for_user()
    return session, docs


def test_repair_citation_ids_treats_a_short_id_as_the_one_passage_it_starts(query_once_module):
    session, docs = _cited_session("Blum says grades harm learning (pqac-0ed33f3).")
    assert "pqac-0ed33f3" in session.answer
    assert LANG_ID not in session.used_contexts

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.raw_answer == f"Blum says grades harm learning ({LANG_ID})."
    assert session.answer == "Blum says grades harm learning (Lang2021 pages 0-1)."
    assert LANG_ID in session.used_contexts
    assert session.references == "1. (Lang2021 pages 0-1): Lang, Small Teaching, 2021."
    assert session.formatted_answer.endswith("References\n\n1. (Lang2021 pages 0-1): Lang, Small Teaching, 2021.")


def test_repair_citation_ids_removes_a_short_id_that_starts_no_passage(query_once_module):
    session, docs = _cited_session("Grades harm learning (pqac-9999999).")

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.raw_answer == "Grades harm learning ()."
    assert "pqac" not in session.answer
    assert session.used_contexts == set()


def test_repair_citation_ids_removes_a_short_id_that_starts_more_than_one_passage(query_once_module):
    session, docs = _cited_session("Grades harm learning (pqac-0ed3).", ids=("pqac-0ed33f3f", "pqac-0ed3aaaa"))

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.raw_answer == "Grades harm learning ()."
    assert session.used_contexts == set()


def test_repair_citation_ids_removes_a_short_id_too_short_to_tell_which_passage_it_means(query_once_module):
    session, docs = _cited_session("Grades harm learning (pqac-0ed).")

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.raw_answer == "Grades harm learning ()."
    assert session.used_contexts == set()


def test_repair_citation_ids_treats_a_long_id_as_the_one_passage_it_starts_with(query_once_module):
    session, docs = _cited_session("Grades harm learning (pqac-0ed33f3ff).")

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.raw_answer == f"Grades harm learning ({LANG_ID})."
    assert session.answer == "Grades harm learning (Lang2021 pages 0-1)."


def test_repair_citation_ids_removes_a_long_id_that_starts_with_no_passage_id(query_once_module):
    session, docs = _cited_session("Grades harm learning (pqac-123456789).")

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.raw_answer == "Grades harm learning ()."


def test_repair_citation_ids_fixes_a_short_id_next_to_a_correct_one(query_once_module):
    session, docs = _cited_session(f"Grades harm learning ({OTHER_ID}, pqac-0ed33f3).")

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.answer == "Grades harm learning (Lang2021 pages 1-2, Lang2021 pages 0-1)."
    assert session.used_contexts == {LANG_ID, OTHER_ID}


def test_repair_citation_ids_leaves_paperqas_own_answer_when_every_id_is_eight_characters(query_once_module):
    # Full ids, and an 8-character id that matches no passage, are PaperQA's to handle; it already has.
    session, docs = _cited_session(f"Grades harm learning ({LANG_ID}) and more (pqac-deadbeef).")
    session.answer = "Answer as PaperQA left it"

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.answer == "Answer as PaperQA left it"
    assert session.raw_answer == f"Grades harm learning ({LANG_ID}) and more (pqac-deadbeef)."


def test_repair_citation_ids_does_nothing_without_a_raw_answer(query_once_module):
    session = SimpleNamespace(answer="A", formatted_answer="Q\n\nA", contexts=[])

    query_once_module.repair_citation_ids(session, [])

    assert session.answer == "A"
    assert not hasattr(session, "raw_answer")


def test_repair_citation_ids_takes_references_from_the_full_chunks_and_keeps_the_emptied_contexts(query_once_module):
    session, docs = _cited_session("Grades harm learning (pqac-0ed33f3).", citation="Stripped citation")
    for chunk in docs.texts:
        chunk.doc = Doc(docname="Lang2021", citation="Full citation", dockey="lang")

    query_once_module.repair_citation_ids(session, docs.texts)

    assert session.references == "1. (Lang2021 pages 0-1): Full citation"
    # The session keeps PaperQA's emptied contexts, so the kept answer stays small.
    assert [context.text.text for context in session.contexts] == ["", ""]


def test_repair_citation_ids_uses_the_context_itself_when_no_full_chunk_matches(query_once_module):
    session, _ = _cited_session("Grades harm learning (pqac-0ed33f3).", citation="Context citation")

    query_once_module.repair_citation_ids(session, [])

    assert session.answer == "Grades harm learning (Lang2021 pages 0-1)."
    assert session.references == "1. (Lang2021 pages 0-1): Context citation"


def test_run_query_payload_repairs_short_ids_before_checking_quotes(query_once_module):
    session, docs = _cited_session(f'Blum writes "{MADE_UP}" (pqac-0ed33f3).')
    session.cost = 0.0
    response = SimpleNamespace(session=session)
    query_once_module.build_settings = mock.Mock(return_value=_stub_settings())
    query_once_module.Docs = mock.Mock(return_value=docs)
    query_once_module.run_or_ensure = mock.Mock(return_value=response)

    with mock.patch.object(query_once_module.settings_groups, "record_answer_settings", return_value=None):
        result = query_once_module.run_query_payload("Q")

    mark = query_once_module.quote_check.PARAPHRASE_MARK
    assert session.answer == f"Blum writes {MADE_UP} {mark} (Lang2021 pages 0-1)."
    assert "References\n\n1. (Lang2021 pages 0-1)" in result["answer"]
    assert [context["cited"] for context in result["contexts"]] == [True, False]
