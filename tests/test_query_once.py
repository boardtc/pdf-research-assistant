import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture
def query_once_module():
    paperqa_module = ModuleType("paperqa")
    agents_module = ModuleType("paperqa.agents")
    agents_main_module = ModuleType("paperqa.agents.main")
    utils_module = ModuleType("paperqa.utils")
    bootstrap_module = ModuleType("pdf_research_assistant.bootstrap")

    # Mock the PaperQA Docs constructor so importing query_once never needs the real dependency.
    paperqa_module.Docs = mock.Mock(name="Docs")

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
        }
    ]


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
    settings = SimpleNamespace(agent=SimpleNamespace(agent_type="stub-agent"))
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
    docs_instance = object()
    query_once_module.Docs = mock.Mock(return_value=docs_instance)

    # Mock run_agent so we can verify the exact coroutine inputs without invoking PaperQA.
    fake_coro = object()
    query_once_module.run_agent = mock.Mock(return_value=fake_coro)

    # Mock run_or_ensure to short-circuit coroutine execution and hand back a canned response object.
    query_once_module.run_or_ensure = mock.Mock(return_value=response)

    # Mock context serialization so this test stays focused on the success flow orchestration.
    query_once_module.serialize_contexts = mock.Mock(return_value=[{"name": "Source"}])

    result = query_once_module.run_query_payload("What is the answer?")

    assert result == {
        "ok": True,
        "answer": "Answer text",
        "cost": 1.25,
        "contexts": [{"name": "Source"}],
    }
    query_once_module.build_settings.assert_called_once_with()
    query_once_module.Docs.assert_called_once_with()
    query_once_module.run_agent.assert_called_once_with(
        docs_instance,
        "What is the answer?",
        settings,
        agent_type="stub-agent",
    )
    query_once_module.run_or_ensure.assert_called_once_with(coro=fake_coro)
    query_once_module.serialize_contexts.assert_called_once_with(["ctx-1"], cited_ids={"pqac-1"})


def test_run_query_payload_marks_nothing_as_cited_when_the_session_does_not_say(query_once_module):
    settings = SimpleNamespace(agent=SimpleNamespace(agent_type="stub-agent"))
    response = SimpleNamespace(session=SimpleNamespace(formatted_answer="Answer text", cost=0.0, contexts=["ctx-1"]))
    query_once_module.build_settings = mock.Mock(return_value=settings)
    query_once_module.run_or_ensure = mock.Mock(return_value=response)
    query_once_module.serialize_contexts = mock.Mock(return_value=[])

    query_once_module.run_query_payload("What is the answer?")

    query_once_module.serialize_contexts.assert_called_once_with(["ctx-1"], cited_ids=())


def test_run_query_payload_returns_error_payload_when_query_execution_raises(query_once_module):
    settings = SimpleNamespace(agent=SimpleNamespace(agent_type="stub-agent"))

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
    print_mock.assert_called_once_with(
        json.dumps({"ok": False, "error": "Expected exactly one question argument."})
    )


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
    payload_mock.assert_called_once_with("What is new?")
    print_mock.assert_called_once_with(json.dumps(payload, ensure_ascii=False))
