from contextlib import contextmanager
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


class _ContextManagerStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _fake_settings(**changes):
    """Stand in for PaperQA settings with the values the sidebar reads."""
    values = {
        "llm": "gpt-4o-mini",
        "embedding": "text-embedding-3-small",
        "paper_directory": Path("papers"),
        "use_absolute_paper_directory": False,
        "parse_pdf": _fake_pdf_reader,
        "reader_config": {"chunk_chars": 5000, "overlap": 250},
        "multimodal": False,
    } | changes
    return SimpleNamespace(
        llm=values["llm"],
        embedding=values["embedding"],
        parsing=SimpleNamespace(
            parse_pdf=values["parse_pdf"], reader_config=values["reader_config"], multimodal=values["multimodal"]
        ),
        agent=SimpleNamespace(
            index=SimpleNamespace(
                paper_directory=values["paper_directory"],
                use_absolute_paper_directory=values["use_absolute_paper_directory"],
            )
        ),
    )


def _fake_pdf_reader():
    return None


@contextmanager
def imported_pdf_research_assistant(
    *,
    allowed_paths=None,
    failed_files=None,
    indexed_count=1,
    button_values=None,
    chat_input_value=None,
    app_state=None,
    subprocess_result=None,
    traceback_lines=None,
    search_mode="Query",
    client_patches=None,
    active_index_dir=Path(r"C:\index\active-shard"),
):
    streamlit_module = ModuleType("streamlit")
    bootstrap_module = ModuleType("pdf_research_assistant.bootstrap")

    streamlit_module.session_state = _SessionState()
    if app_state:
        streamlit_module.session_state.update(app_state)
    streamlit_module.sidebar = _ContextManagerStub()
    streamlit_module.cache_resource = lambda fn: fn
    # Mock the button API with a queue of return values so import-time UI flows can be driven deterministically.
    queued_button_values = list(button_values or [False, False])

    def click(_label, key=None, **_kwargs):
        # The local LLM panel is drawn on every page; its buttons stay unclicked and leave the queue alone.
        if str(key or "").startswith("local_llm_"):
            return False
        return queued_button_values.pop(0)

    streamlit_module.button = mock.Mock(side_effect=click)
    streamlit_module.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: options[index])
    streamlit_module.text_input = mock.Mock(side_effect=lambda _label, value="", **_kwargs: value)
    streamlit_module.success = mock.Mock()
    streamlit_module.expander = mock.Mock(return_value=_ContextManagerStub())
    streamlit_module.subheader = mock.Mock()
    streamlit_module.write = mock.Mock()
    streamlit_module.caption = mock.Mock()
    streamlit_module.error = mock.Mock()
    streamlit_module.toast = mock.Mock()
    streamlit_module.title = mock.Mock()
    streamlit_module.warning = mock.Mock()
    streamlit_module.info = mock.Mock()
    streamlit_module.header = mock.Mock()
    streamlit_module.metric = mock.Mock()
    streamlit_module.rerun = mock.Mock()
    streamlit_module.chat_message = mock.Mock(return_value=_ContextManagerStub())
    streamlit_module.chat_input = mock.Mock(return_value=chat_input_value)
    streamlit_module.markdown = mock.Mock()
    streamlit_module.spinner = mock.Mock(return_value=_ContextManagerStub())
    streamlit_module.code = mock.Mock()
    streamlit_module.radio = mock.Mock(return_value=search_mode)
    streamlit_module.progress = mock.Mock()
    streamlit_module.download_button = mock.Mock()
    streamlit_module.fragment = mock.Mock(return_value=lambda fn: fn)

    # Mock bootstrap lookups so each import scenario can model a specific index and manifest state.
    bootstrap_module.get_allowed_paths = mock.Mock(return_value=set() if allowed_paths is None else allowed_paths)
    bootstrap_module.get_failed_files = mock.Mock(return_value=[] if failed_files is None else failed_files)
    bootstrap_module.get_indexed_doc_count = mock.Mock(return_value=indexed_count)
    bootstrap_module.get_active_index_dir = mock.Mock(return_value=active_index_dir)
    bootstrap_module.build_settings = mock.Mock(return_value=_fake_settings())
    bootstrap_module.MODEL_NAME = "gpt-4o-mini"
    bootstrap_module.MANIFEST_PATH = Path(r"D:\papers-list\manifest.csv")
    bootstrap_module.PAPER_DIR = Path(r"D:\papers")

    with mock.patch.dict(
        sys.modules,
        {
            "streamlit": streamlit_module,
            "pdf_research_assistant.bootstrap": bootstrap_module,
        },
    ):
        if subprocess_result is None:
            subprocess_result = SimpleNamespace(returncode=0, stdout='{"ok": true, "answer": "", "cost": 0.0, "contexts": []}', stderr="")

        # Mock subprocess.run before import so question-submission scenarios never hit the real helper process.
        with mock.patch("subprocess.run", return_value=subprocess_result), mock.patch("time.sleep"), \
                mock.patch.multiple(
                    "pdf_research_assistant.exhaustive_client", RUNS_DIR=Path("unused"), **(client_patches or {})
                ):
            # Mock traceback formatting before import so assistant error rendering stays deterministic.
            with mock.patch(
                "traceback.format_exception",
                return_value=(traceback_lines if traceback_lines is not None else ["trace line\n"]),
            ):
                sys.modules.pop("pdf_research_assistant.app", None)
                module = importlib.import_module("pdf_research_assistant.app")
                try:
                    yield module
                finally:
                    sys.modules.pop("pdf_research_assistant.app", None)


@pytest.fixture
def pdf_research_assistant_module():
    with imported_pdf_research_assistant() as module:
        yield module


READY_PLAN = {
    "paper_count": 2,
    "status_counts": {"loadable": 2},
    "chunk_count": 5,
    "cost_usd": {"low": 0.01, "typical": 0.02, "high": 0.03},
    "estimated_minutes": 1.0,
}
EMPTY_PLAN = READY_PLAN | {"status_counts": {"not_indexed": 2}, "chunk_count": 0}
FINISHED_SUMMARY = {
    "paper_count": 2,
    "status_counts": {"relevant": 1},
    "failed_chunks": 0,
    "cost_usd": 0.5,
    "cost_may_be_understated": False,
    "runtime_seconds": 60,
    "stopped_early": False,
    "stop_reason": None,
}


def _empty_state():
    return {"history": [], "total_cost": 0.0, "query_count": 0}


def _markdown_text(module):
    return "\n".join(str(call.args[0]) for call in module.st.markdown.call_args_list)


def _active_run(tmp_dir, poll_result):
    process = mock.Mock()
    process.poll.return_value = poll_result
    return {"question": "q", "output_dir": str(tmp_dir), "process": process, "paper_count": 2}


def test_search_mode_defaults_to_query():
    with imported_pdf_research_assistant(app_state=_empty_state()) as module:
        args, kwargs = module.st.radio.call_args

    assert args[1][0] == "Query"
    assert kwargs["index"] == 0


def test_query_mode_never_asks_for_an_exhaustive_plan():
    request_plan = mock.Mock()

    with imported_pdf_research_assistant(
        app_state=_empty_state(), chat_input_value="q", client_patches={"request_plan": request_plan}
    ):
        request_plan.assert_not_called()


def test_exhaustive_question_shows_a_plan_without_starting_a_run():
    start_run = mock.Mock()
    patches = {"request_plan": mock.Mock(return_value=READY_PLAN), "start_run": start_run}

    with imported_pdf_research_assistant(
        app_state=_empty_state(), chat_input_value="q", search_mode="Exhaustive", client_patches=patches
    ) as module:
        assert module.st.session_state.exhaustive_pending == {"question": "q", "plan": READY_PLAN}
        module.st.rerun.assert_called_once_with()

    start_run.assert_not_called()


def test_exhaustive_planning_failure_is_shown_and_nothing_is_pending():
    patches = {"request_plan": mock.Mock(side_effect=RuntimeError("bad index"))}

    with imported_pdf_research_assistant(
        app_state=_empty_state(), chat_input_value="q", search_mode="Exhaustive", client_patches=patches
    ) as module:
        module.st.error.assert_any_call("Could not plan the search: bad index")
        assert "exhaustive_pending" not in module.st.session_state


def test_pending_plan_shows_cost_and_blocks_new_questions_until_answered():
    start_run = mock.Mock()
    state = _empty_state() | {"exhaustive_pending": {"question": "q", "plan": READY_PLAN}}

    with imported_pdf_research_assistant(
        app_state=state, button_values=[False, False, False], client_patches={"start_run": start_run}
    ) as module:
        assert r"Estimated cost: \$0.01 to \$0.03, typically \$0.02" in _markdown_text(module)
        assert module.st.chat_input.call_args.kwargs["disabled"] is True

    start_run.assert_not_called()


def test_confirming_starts_the_run_and_records_the_question(tmp_path):
    start_run = mock.Mock(return_value="process")
    state = _empty_state() | {"exhaustive_pending": {"question": "q", "plan": READY_PLAN}}
    patches = {"start_run": start_run, "new_run_dir": mock.Mock(return_value=tmp_path)}

    with imported_pdf_research_assistant(app_state=state, button_values=[False, True], client_patches=patches) as module:
        run = module.st.session_state.exhaustive_run
        assert run == {"question": "q", "output_dir": str(tmp_path), "process": "process", "paper_count": 2}
        assert module.st.session_state.exhaustive_pending is None
        assert module.st.session_state.history == [{"role": "user", "content": "q"}]

    start_run.assert_called_once_with("q", tmp_path)


def test_cancelling_clears_the_plan_without_starting_a_run():
    start_run = mock.Mock()
    state = _empty_state() | {"exhaustive_pending": {"question": "q", "plan": READY_PLAN}}

    with imported_pdf_research_assistant(
        app_state=state, button_values=[False, False, True], client_patches={"start_run": start_run}
    ) as module:
        assert module.st.session_state.exhaustive_pending is None

    start_run.assert_not_called()


def test_no_run_is_offered_when_no_paper_can_be_searched():
    state = _empty_state() | {"exhaustive_pending": {"question": "q", "plan": EMPTY_PLAN}}

    with imported_pdf_research_assistant(app_state=state, button_values=[False, False]) as module:
        labels = [call.args[0] for call in module.st.button.call_args_list]
        module.st.error.assert_any_call("No papers can be searched yet. Run pdf-research-rebuild first.")

    assert "Run paid search" not in labels


def test_dismissing_an_unrunnable_plan_clears_it():
    state = _empty_state() | {"exhaustive_pending": {"question": "q", "plan": EMPTY_PLAN}}

    with imported_pdf_research_assistant(app_state=state, button_values=[False, True]) as module:
        assert module.st.session_state.exhaustive_pending is None
        module.st.rerun.assert_called_once_with()


def test_running_search_shows_progress_without_restarting_the_page(tmp_path):
    (tmp_path / "results.jsonl").write_text("{}\n", encoding="utf-8")
    state = _empty_state() | {"exhaustive_run": _active_run(tmp_path, None)}

    with imported_pdf_research_assistant(app_state=state) as module:
        module.st.progress.assert_called_once_with(0.5, text="1 of 2 papers finished")
        assert module.st.chat_input.call_args.kwargs["disabled"] is True
        assert module.st.session_state.exhaustive_run is state["exhaustive_run"]
        # A full-page restart here would leave the old Run button on screen.
        module.st.rerun.assert_not_called()


def test_progress_refreshes_by_itself_on_a_timer():
    with imported_pdf_research_assistant(app_state=_empty_state()) as module:
        module.st.fragment.assert_called_once_with(run_every=module.PROGRESS_POLL_SECONDS)


def test_escape_dollars_stops_streamlit_reading_amounts_as_maths(pdf_research_assistant_module):
    result = pdf_research_assistant_module.escape_dollars("$0.01 to $0.03")

    assert result == r"\$0.01 to \$0.03"


def test_pending_plan_escapes_dollar_amounts():
    state = _empty_state() | {"exhaustive_pending": {"question": "Cost of $5 and $6?", "plan": READY_PLAN}}

    with imported_pdf_research_assistant(app_state=state, button_values=[False, False, False]) as module:
        text = _markdown_text(module)

    assert r"Estimated cost: \$0.01 to \$0.03, typically \$0.02" in text
    assert r"Cost of \$5 and \$6?" in text


def test_finished_search_escapes_dollar_amounts_in_the_message(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(FINISHED_SUMMARY), encoding="utf-8")
    state = _empty_state() | {"exhaustive_run": _active_run(tmp_path, 0)}

    with imported_pdf_research_assistant(app_state=state) as module:
        assert r"Actual cost: \$0.5000" in module.st.session_state.history[-1]["content"]


def test_full_report_escapes_dollars_on_screen_but_not_in_the_download(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("Cost: $1 and $2", encoding="utf-8")
    item = {"role": "assistant", "id": 1, "content": "Done", "cost": 0.5, "contexts": [], "report_path": str(report)}

    with imported_pdf_research_assistant(
        app_state=_empty_state() | {"history": [item]}, button_values=[False, False]
    ) as module:
        assert module.st.download_button.call_args.args[1] == "Cost: $1 and $2"
        module.st.markdown.assert_any_call(r"Cost: \$1 and \$2", unsafe_allow_html=True)


def test_finished_search_records_result_cost_and_report(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(FINISHED_SUMMARY), encoding="utf-8")
    state = _empty_state() | {"exhaustive_run": _active_run(tmp_path, 0)}

    with imported_pdf_research_assistant(app_state=state) as module:
        item = module.st.session_state.history[-1]
        assert "Actual cost" in item["content"]
        assert item["cost"] == 0.5
        assert item["report_path"] == str(tmp_path / "report.md")
        assert module.st.session_state.total_cost == 0.5
        assert module.st.session_state.query_count == 1
        assert module.st.session_state.exhaustive_run is None


def test_crashed_search_says_so_and_counts_saved_cost(tmp_path):
    (tmp_path / "results.jsonl").write_text('{"cost_usd": 0.25}\n', encoding="utf-8")
    state = _empty_state() | {"exhaustive_run": _active_run(tmp_path, 1)}

    with imported_pdf_research_assistant(app_state=state) as module:
        item = module.st.session_state.history[-1]
        assert "stopped unexpectedly" in item["content"]
        assert item["cost"] == 0.25
        assert "report_path" not in item
        assert module.st.session_state.exhaustive_run is None


def test_history_shows_download_and_full_report(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Report", encoding="utf-8")
    item = {"role": "assistant", "id": 1, "content": "Done", "cost": 0.5, "contexts": [], "report_path": str(report)}

    with imported_pdf_research_assistant(
        app_state=_empty_state() | {"history": [item]}, button_values=[False, False]
    ) as module:
        assert module.st.download_button.call_args.args[:2] == ("Download report", "# Report")
        module.st.markdown.assert_any_call("# Report", unsafe_allow_html=True)


def test_history_warns_when_the_report_has_gone(tmp_path):
    missing = tmp_path / "report.md"
    item = {"role": "assistant", "id": 1, "content": "Done", "cost": 0.5, "contexts": [], "report_path": str(missing)}

    with imported_pdf_research_assistant(
        app_state=_empty_state() | {"history": [item]}, button_values=[False, False]
    ) as module:
        module.st.warning.assert_any_call(f"Report not found: {missing}")
        module.st.download_button.assert_not_called()


def test_flatten_exception_messages_formats_top_level_exception_message(pdf_research_assistant_module):
    exc = RuntimeError("boom")

    result = pdf_research_assistant_module.flatten_exception_messages(exc)

    assert result == ["RuntimeError: boom"]


def test_flatten_exception_messages_recursively_includes_child_exception_messages(pdf_research_assistant_module):
    child = ValueError("child problem")
    parent = ExceptionGroup("group", [child])

    result = pdf_research_assistant_module.flatten_exception_messages(parent)

    assert "  ValueError: child problem" in result


def test_flatten_exception_messages_indents_nested_exception_messages_by_depth(pdf_research_assistant_module):
    grandchild = RuntimeError("grandchild")
    child = ExceptionGroup("child group", [grandchild])
    parent = ExceptionGroup("parent group", [child])

    result = pdf_research_assistant_module.flatten_exception_messages(parent)

    assert "    RuntimeError: grandchild" in result


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only clipboard behavior")
def test_windows_only_copy_to_clipboard_invokes_powershell_clipboard_command_with_utf8_text_input(
    pdf_research_assistant_module,
):
    # Mock subprocess.run so the test can assert on the clipboard command without touching the real clipboard.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run") as run_mock:
        pdf_research_assistant_module.copy_to_clipboard("Copied text")

    run_mock.assert_called_once_with(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Set-Clipboard -Value ([Console]::In.ReadToEnd())",
        ],
        input="Copied text",
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def test_render_copy_button_does_nothing_when_button_is_not_clicked(pdf_research_assistant_module):
    # Mock the Streamlit button so the function follows the "not clicked" branch.
    with mock.patch.object(pdf_research_assistant_module.st, "button", return_value=False):
        # Mock copy_to_clipboard so the test can assert it is not called on the idle path.
        with mock.patch.object(pdf_research_assistant_module, "copy_to_clipboard") as copy_mock:
            pdf_research_assistant_module.render_copy_button("Answer text", key="copy-key")

    copy_mock.assert_not_called()


def test_render_copy_button_shows_error_when_copy_to_clipboard_raises(pdf_research_assistant_module):
    # Mock the Streamlit button so the function follows the clicked branch.
    with mock.patch.object(pdf_research_assistant_module.st, "button", return_value=True):
        # Mock copy_to_clipboard to raise the failure that should be surfaced to the user.
        with mock.patch.object(
            pdf_research_assistant_module,
            "copy_to_clipboard",
            side_effect=RuntimeError("clipboard unavailable"),
        ):
            # Mock st.error so the test can assert on the rendered failure message.
            with mock.patch.object(pdf_research_assistant_module.st, "error") as error_mock:
                pdf_research_assistant_module.render_copy_button("Answer text", key="copy-key")

    error_mock.assert_called_once_with("Copy failed: clipboard unavailable")


def test_render_copy_button_shows_toast_when_copy_to_clipboard_succeeds(pdf_research_assistant_module):
    # Mock the Streamlit button so the function follows the clicked branch.
    with mock.patch.object(pdf_research_assistant_module.st, "button", return_value=True):
        # Mock copy_to_clipboard so the happy path does not touch the system clipboard.
        with mock.patch.object(pdf_research_assistant_module, "copy_to_clipboard") as copy_mock:
            # Mock st.toast so the test can assert on the success feedback shown to the user.
            with mock.patch.object(pdf_research_assistant_module.st, "toast") as toast_mock:
                pdf_research_assistant_module.render_copy_button("Answer text", key="copy-key")

    copy_mock.assert_called_once_with("Answer text")
    toast_mock.assert_called_once_with("Answer copied")


def test_render_source_passages_returns_immediately_when_contexts_are_empty(pdf_research_assistant_module):
    # Mock the Streamlit expander so the test can assert that no UI is opened on the empty path.
    with mock.patch.object(pdf_research_assistant_module.st, "expander") as expander_mock:
        pdf_research_assistant_module.render_source_passages([])

    expander_mock.assert_not_called()


def test_render_source_passages_renders_source_subheader_and_context_for_each_passage(pdf_research_assistant_module):
    contexts = [{"name": "Paper A", "context": "Quoted text"}]

    # Mock the expander context manager so the test exercises the rendering branch without real UI state.
    with mock.patch.object(pdf_research_assistant_module.st, "expander", return_value=_ContextManagerStub()):
        # Mock subheader so the test can verify the visible source label.
        with mock.patch.object(pdf_research_assistant_module.st, "subheader") as subheader_mock:
            # Mock write so the test can verify the rendered passage text.
            with mock.patch.object(pdf_research_assistant_module.st, "write") as write_mock:
                pdf_research_assistant_module.render_source_passages(contexts)

    subheader_mock.assert_called_once_with("Paper A")
    assert write_mock.call_args_list[0] == mock.call("Quoted text")


def test_render_source_passages_uses_source_fallback_name_when_context_name_is_missing(pdf_research_assistant_module):
    contexts = [{"context": "Quoted text"}]

    # Mock the expander context manager so the test can execute the rendering branch safely.
    with mock.patch.object(pdf_research_assistant_module.st, "expander", return_value=_ContextManagerStub()):
        # Mock subheader so the test can assert on the fallback label.
        with mock.patch.object(pdf_research_assistant_module.st, "subheader") as subheader_mock:
            pdf_research_assistant_module.render_source_passages(contexts)

    subheader_mock.assert_called_once_with("Source")


def test_render_source_passages_shows_raw_chunk_caption_and_text_when_raw_text_is_present(pdf_research_assistant_module):
    contexts = [{"name": "Paper A", "context": "Quoted text", "raw_text": "Raw chunk"}]

    # Mock the expander context manager so the test can execute the rendering branch safely.
    with mock.patch.object(pdf_research_assistant_module.st, "expander", return_value=_ContextManagerStub()):
        # Mock caption so the test can assert on the raw-text label.
        with mock.patch.object(pdf_research_assistant_module.st, "caption") as caption_mock:
            # Mock write so the test can assert that the raw chunk text is also rendered.
            with mock.patch.object(pdf_research_assistant_module.st, "write") as write_mock:
                pdf_research_assistant_module.render_source_passages(contexts)

    caption_mock.assert_called_once_with("Raw chunk text")
    assert write_mock.call_args_list[-1] == mock.call("Raw chunk")


def test_run_query_subprocess_returns_answer_cost_and_contexts_for_successful_json_payload(pdf_research_assistant_module):
    result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": True, "answer": "Answer", "cost": 0.25, "contexts": [{"name": "Source"}]}),
        stderr="",
    )

    # Mock subprocess.run so the test can supply the exact helper output contract.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result) as run_mock:
        answer, cost, contexts = pdf_research_assistant_module.run_query_subprocess("What is new?")

    assert (answer, cost, contexts) == ("Answer", 0.25, [{"name": "Source"}])
    assert run_mock.call_args.args[0][0] == pdf_research_assistant_module.sys.executable
    assert run_mock.call_args.args[0][-1] == "What is new?"


def test_run_query_subprocess_raises_subprocess_error_message_when_process_fails_without_stdout(
    pdf_research_assistant_module,
):
    result = SimpleNamespace(returncode=1, stdout="   ", stderr="subprocess failed")

    # Mock subprocess.run so the function follows the non-zero-returncode branch with empty stdout.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="subprocess failed"):
            pdf_research_assistant_module.run_query_subprocess("What is new?")


def test_run_query_subprocess_raises_invalid_query_response_when_stdout_is_not_json_and_no_better_error_exists(
    pdf_research_assistant_module,
):
    result = SimpleNamespace(returncode=0, stdout="not json", stderr="")

    # Mock subprocess.run so the function reaches the JSON parsing error branch.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="not json"):
            pdf_research_assistant_module.run_query_subprocess("What is new?")


def test_run_query_subprocess_raises_stderr_message_when_stdout_is_not_json_but_stderr_is_present(
    pdf_research_assistant_module,
):
    result = SimpleNamespace(returncode=0, stdout="not json", stderr="better error")

    # Mock subprocess.run so stderr takes precedence after JSON decoding fails.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="better error"):
            pdf_research_assistant_module.run_query_subprocess("What is new?")


def test_run_query_subprocess_raises_stdout_message_when_stdout_is_not_json_but_contains_text(
    pdf_research_assistant_module,
):
    result = SimpleNamespace(returncode=0, stdout="plain text failure", stderr="   ")

    # Mock subprocess.run so stdout becomes the fallback error message after JSON decoding fails.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="plain text failure"):
            pdf_research_assistant_module.run_query_subprocess("What is new?")


def test_run_query_subprocess_raises_payload_error_type_and_message_when_query_payload_is_not_ok(
    pdf_research_assistant_module,
):
    result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": False, "error_type": "ValueError", "error": "bad payload"}),
        stderr="",
    )

    # Mock subprocess.run so the function reaches the application-level error payload branch.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="ValueError: bad payload"):
            pdf_research_assistant_module.run_query_subprocess("What is new?")


def test_run_query_subprocess_defaults_unknown_error_type_and_message_when_error_payload_fields_are_missing(
    pdf_research_assistant_module,
):
    result = SimpleNamespace(returncode=0, stdout=json.dumps({"ok": False}), stderr="")

    # Mock subprocess.run so the function exercises the default error-type and error-message fallbacks.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="Error: Unknown error"):
            pdf_research_assistant_module.run_query_subprocess("What is new?")


def test_get_persistent_app_state_returns_initial_history_cost_and_query_count_defaults(pdf_research_assistant_module):
    result = pdf_research_assistant_module.get_persistent_app_state()

    assert result == {
        "history": [],
        "total_cost": 0.0,
        "query_count": 0,
    }


def test_sync_persistent_state_copies_history_from_session_state_into_cached_app_state(pdf_research_assistant_module):
    app_state = {"history": [], "total_cost": 0.0, "query_count": 0}
    pdf_research_assistant_module.st.session_state.history = [{"role": "user", "content": "Hi"}]
    pdf_research_assistant_module.st.session_state.total_cost = 1.5
    pdf_research_assistant_module.st.session_state.query_count = 3

    pdf_research_assistant_module.sync_persistent_state(app_state)

    assert app_state == {
        "history": [{"role": "user", "content": "Hi"}],
        "total_cost": 1.5,
        "query_count": 3,
    }


def test_import_warns_when_manifest_index_is_missing():
    with imported_pdf_research_assistant(
        allowed_paths={"paper_a.pdf", "paper_b.pdf"},
        indexed_count=0,
    ) as module:
        module.st.warning.assert_called_once_with(
            "Index not found. The first question will rebuild the index for 2 manifest PDFs, which may take a long time."
        )
        module.st.info.assert_called_once_with("A rebuild starts only after you submit a question below.")


def test_import_warns_when_manifest_index_is_incomplete():
    with imported_pdf_research_assistant(
        allowed_paths={"paper_a.pdf", "paper_b.pdf", "paper_c.pdf"},
        indexed_count=2,
    ) as module:
        module.st.warning.assert_called_once_with(
            "Index is incomplete: 2 of 3 manifest PDFs appear indexed. The next question should continue or trigger indexing work."
        )


def test_import_warns_when_non_manifest_index_is_missing():
    with imported_pdf_research_assistant(
        allowed_paths=set(),
        indexed_count=0,
    ) as module:
        module.st.warning.assert_called_once_with(
            "Index not found. The first question will build the PDF index, which may take a long time."
        )
        module.st.info.assert_called_once_with("No manifest.csv was found, so all PDFs under PAPER_DIR are in scope.")


def test_import_shows_manifest_index_caption_when_manifest_index_is_available():
    with imported_pdf_research_assistant(
        allowed_paths={"paper_a.pdf", "paper_b.pdf"},
        indexed_count=2,
    ) as module:
        module.st.caption.assert_any_call("Index available for 2 manifest PDFs.")


def test_import_uses_the_active_index_directory_for_startup_index_counts_and_failed_files():
    active_index_dir = Path(r"C:\index\active-shard")

    with imported_pdf_research_assistant(
        allowed_paths={"paper_a.pdf", "paper_b.pdf"},
        indexed_count=2,
        failed_files=[],
    ) as module:
        module.bootstrap.get_active_index_dir.assert_called_once_with(module.bootstrap.build_settings.return_value)
        module.bootstrap.get_indexed_doc_count.assert_called_once_with(index_dir=active_index_dir)
        module.bootstrap.get_failed_files.assert_called_once_with(index_dir=active_index_dir)


def test_import_shows_the_active_index_folder_and_paper_list_the_query_path_uses():
    with imported_pdf_research_assistant(allowed_paths={"paper_a.pdf", "paper_b.pdf"}, indexed_count=2) as module:
        module.st.caption.assert_any_call(
            "Active index: `active-shard` · Paper list: `manifest.csv` (2 PDFs)"
        )
        details = _markdown_text(module)
        assert r"`C:\index\active-shard`" in details
        assert r"`D:\papers-list\manifest.csv`" in details
        assert r"`D:\papers`" in details


def test_import_shows_the_new_active_index_after_it_changes():
    with imported_pdf_research_assistant(active_index_dir=Path(r"C:\index\shard-one")) as module:
        module.st.caption.assert_any_call(
            "Active index: `shard-one` · Paper list: none, so all PDFs under PAPER_DIR are in scope"
        )
    with imported_pdf_research_assistant(active_index_dir=Path(r"C:\index\shard-two")) as module:
        module.st.caption.assert_any_call(
            "Active index: `shard-two` · Paper list: none, so all PDFs under PAPER_DIR are in scope"
        )


def test_import_shows_failed_files_error_and_lists_each_failed_file():
    with imported_pdf_research_assistant(
        failed_files=["bad_a.pdf", "bad_b.pdf"],
    ) as module:
        module.st.error.assert_any_call("2 PDF(s) failed to index.")
        assert module.st.code.call_args_list == [mock.call("bad_a.pdf"), mock.call("bad_b.pdf")]


def test_import_clear_chat_button_resets_session_state_and_reruns():
    with imported_pdf_research_assistant(
        app_state={"history": [{"role": "user", "content": "Hi"}], "total_cost": 1.0, "query_count": 2},
        button_values=[True],
    ) as module:
        assert module.st.session_state.history == []
        assert module.st.session_state.total_cost == 0.0
        assert module.st.session_state.query_count == 0
        module.st.rerun.assert_called_once_with()


def test_import_renders_history_and_assistant_history_actions():
    with imported_pdf_research_assistant(
        app_state={
            "history": [
                {"role": "user", "content": "User question"},
                {
                    "role": "assistant",
                    "content": "Assistant answer",
                    "cost": 0.25,
                    "contexts": [{"name": "Source", "context": "Quoted text"}],
                    "id": 7,
                },
            ],
            "total_cost": 0.25,
            "query_count": 1,
        },
    ) as module:
        assert module.st.chat_message.call_args_list[:2] == [mock.call("user"), mock.call("assistant")]
        # The active index details (index and PDF folders, no paper list here) and the sidebar's guide and
        # index settings are Markdown too; they are drawn before the chat.
        lines_before_chat = 2 + 1 + len(module.settings_groups.REINDEX_SETTINGS)
        chat_markdown = module.st.markdown.call_args_list[lines_before_chat:]
        assert chat_markdown[:2] == [mock.call("User question"), mock.call("Assistant answer")]
        module.st.caption.assert_any_call("Cost: $0.2500")


def test_import_question_submission_success_updates_history_cost_and_reruns():
    result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": True, "answer": "Answer text", "cost": 1.5, "contexts": [{"name": "Source", "context": "Quoted text"}]}),
        stderr="",
    )
    with imported_pdf_research_assistant(
        app_state={"history": [], "total_cost": 0.0, "query_count": 0},
        chat_input_value="What is new?",
        subprocess_result=result,
    ) as module:
        assert module.st.session_state.history[0] == {"role": "user", "content": "What is new?"}
        assert module.st.session_state.history[1]["role"] == "assistant"
        assert module.st.session_state.history[1]["content"] == "Answer text"
        assert module.st.session_state.total_cost == 1.5
        assert module.st.session_state.query_count == 1
        module.st.caption.assert_any_call("Cost: $1.5000")
        module.st.rerun.assert_called_once_with()


def test_import_question_submission_error_renders_diagnostics_and_still_appends_assistant_message():
    result = SimpleNamespace(returncode=1, stdout="   ", stderr="query failed")
    with imported_pdf_research_assistant(
        app_state={"history": [], "total_cost": 0.0, "query_count": 0},
        chat_input_value="What is new?",
        subprocess_result=result,
        traceback_lines=["trace line 1\n", "trace line 2\n"],
    ) as module:
        module.st.error.assert_any_call(
            "Indexing/search failed.\n\n"
            "Top-level error:\n"
            "```\nRuntimeError: query failed\n```\n\n"
            "Underlying error(s):\n"
            "```\nRuntimeError: query failed\n```\n\n"
            "This usually means one document or a parsing/indexing step failed."
        )
        module.st.caption.assert_any_call("Cost: $0.0000")
        module.st.code.assert_called_once_with("trace line 1\ntrace line 2\n")
        assert module.st.session_state.history[1]["role"] == "assistant"
        assert module.st.session_state.history[1]["cost"] == 0.0
        assert module.st.session_state.history[1]["contexts"] == []


# Extracted page functions, tested directly. The import tests above cover the page as a whole.


@pytest.mark.parametrize(
    ("indexed_count", "manifest_count", "expected"),
    [
        (
            0,
            2,
            [
                (
                    "warning",
                    "Index not found. The first question will rebuild the index for 2 manifest PDFs, "
                    "which may take a long time.",
                ),
                ("info", "A rebuild starts only after you submit a question below."),
            ],
        ),
        (
            2,
            3,
            [
                (
                    "warning",
                    "Index is incomplete: 2 of 3 manifest PDFs appear indexed. "
                    "The next question should continue or trigger indexing work.",
                )
            ],
        ),
        (
            0,
            0,
            [
                ("warning", "Index not found. The first question will build the PDF index, which may take a long time."),
                ("info", "No manifest.csv was found, so all PDFs under PAPER_DIR are in scope."),
            ],
        ),
        (2, 2, [("caption", "Index available for 2 manifest PDFs.")]),
        (4, 0, [("caption", "Index available for 4 PDFs under PAPER_DIR.")]),
    ],
    ids=["manifest-missing", "manifest-incomplete", "no-manifest-missing", "manifest-ready", "no-manifest-ready"],
)
def test_select_index_status_picks_the_banner_for_each_index_state(
    pdf_research_assistant_module, indexed_count, manifest_count, expected
):
    assert pdf_research_assistant_module.select_index_status(indexed_count, manifest_count) == expected


def test_render_index_status_shows_each_message_at_its_level(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    for shown in (st.warning, st.info, st.caption):
        shown.reset_mock()

    pdf_research_assistant_module.render_index_status([("warning", "W"), ("info", "I"), ("caption", "C")])

    st.warning.assert_called_once_with("W")
    st.info.assert_called_once_with("I")
    st.caption.assert_called_once_with("C")


def test_describe_active_index_names_the_folder_and_the_manifest_in_use(pdf_research_assistant_module):
    details = pdf_research_assistant_module.describe_active_index(
        Path("index") / "pqa_index_abc", Path("lists") / "manifest.csv", 3, Path("papers")
    )

    assert details == {
        "name": "pqa_index_abc",
        "folder": str(Path("index") / "pqa_index_abc"),
        "paper_list": "`manifest.csv` (3 PDFs)",
        "paper_list_path": str(Path("lists") / "manifest.csv"),
        "paper_dir": "papers",
    }


def test_describe_active_index_says_all_pdfs_are_in_scope_without_a_manifest(pdf_research_assistant_module):
    details = pdf_research_assistant_module.describe_active_index(
        Path("index") / "pqa_index_abc", Path("manifest.csv"), 0, Path("papers")
    )

    assert details["paper_list"] == "none, so all PDFs under PAPER_DIR are in scope"
    assert details["paper_list_path"] is None


def test_render_active_index_shows_the_name_and_puts_the_paths_in_details(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    for shown in (st.caption, st.markdown, st.expander):
        shown.reset_mock()

    pdf_research_assistant_module.render_active_index(
        {
            "name": "pqa_index_abc",
            "folder": "C:/index/pqa_index_abc",
            "paper_list": "`manifest.csv` (3 PDFs)",
            "paper_list_path": "D:/lists/manifest.csv",
            "paper_dir": "D:/papers",
        }
    )

    st.caption.assert_called_once_with("Active index: `pqa_index_abc` · Paper list: `manifest.csv` (3 PDFs)")
    st.expander.assert_called_once_with("Active index details")
    assert st.markdown.call_args_list == [
        mock.call("**Index folder:** `C:/index/pqa_index_abc`"),
        mock.call("**Paper list:** `D:/lists/manifest.csv`"),
        mock.call("**PDF folder:** `D:/papers`"),
    ]


def test_render_active_index_skips_the_paper_list_path_without_a_manifest(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.markdown.reset_mock()

    pdf_research_assistant_module.render_active_index(
        {
            "name": "pqa_index_abc",
            "folder": "C:/index/pqa_index_abc",
            "paper_list": "none, so all PDFs under PAPER_DIR are in scope",
            "paper_list_path": None,
            "paper_dir": "D:/papers",
        }
    )

    assert st.markdown.call_args_list == [
        mock.call("**Index folder:** `C:/index/pqa_index_abc`"),
        mock.call("**PDF folder:** `D:/papers`"),
    ]


def test_render_failed_files_shows_nothing_when_no_file_failed(pdf_research_assistant_module):
    pdf_research_assistant_module.st.expander.reset_mock()

    pdf_research_assistant_module.render_failed_files([])

    pdf_research_assistant_module.st.error.assert_not_called()
    pdf_research_assistant_module.st.expander.assert_not_called()


def test_render_failed_files_lists_each_failed_file(pdf_research_assistant_module):
    pdf_research_assistant_module.st.expander.reset_mock()

    pdf_research_assistant_module.render_failed_files(["bad_a.pdf", "bad_b.pdf"])

    pdf_research_assistant_module.st.error.assert_called_once_with("2 PDF(s) failed to index.")
    pdf_research_assistant_module.st.expander.assert_called_once_with("Show failed PDFs")
    assert pdf_research_assistant_module.st.code.call_args_list == [mock.call("bad_a.pdf"), mock.call("bad_b.pdf")]


def test_init_session_state_copies_saved_state_into_a_new_session(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.session_state.clear()
    saved = {"history": [{"role": "user", "content": "Hi"}], "total_cost": 1.5, "query_count": 3}

    pdf_research_assistant_module.init_session_state(saved)

    assert st.session_state == saved
    assert st.session_state.history is not saved["history"]


def test_init_session_state_keeps_values_the_session_already_has(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.session_state.clear()
    st.session_state.update({"history": [{"role": "user", "content": "Mine"}], "total_cost": 9.0, "query_count": 4})

    pdf_research_assistant_module.init_session_state({"history": [], "total_cost": 0.0, "query_count": 0})

    assert st.session_state == {"history": [{"role": "user", "content": "Mine"}], "total_cost": 9.0, "query_count": 4}


def test_render_usage_shows_the_session_counts(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.session_state.update({"history": [], "total_cost": 1.25, "query_count": 2})
    st.button = mock.Mock(return_value=False)
    st.metric.reset_mock()
    st.rerun.reset_mock()

    pdf_research_assistant_module.render_usage({})

    assert st.metric.call_args_list == [
        mock.call("Queries this session", 2),
        mock.call("Total cost this session", "$1.2500"),
    ]
    st.rerun.assert_not_called()


def test_render_usage_clear_chat_resets_the_session_and_the_saved_state(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.session_state.update({"history": [{"role": "user", "content": "Hi"}], "total_cost": 1.0, "query_count": 2})
    st.button = mock.Mock(return_value=True)
    st.rerun.reset_mock()
    saved = {"history": [{"role": "user", "content": "Hi"}], "total_cost": 1.0, "query_count": 2}

    pdf_research_assistant_module.render_usage(saved)

    assert saved == {"history": [], "total_cost": 0.0, "query_count": 0}
    assert st.session_state.history == []
    st.rerun.assert_called_once_with()


# Sidebar settings: one group applies to the next question, the other needs the index rebuilt.


def test_render_instant_settings_returns_the_chosen_mode_and_explains_the_group(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.radio = mock.Mock(return_value="Exhaustive")
    st.subheader.reset_mock()
    st.caption.reset_mock()

    with mock.patch.object(module, "render_local_llm_setup") as local_llm_panel:
        assert module.render_instant_settings() == "Exhaustive"

    st.subheader.assert_called_once_with(module.INSTANT_SETTINGS_TITLE)
    st.caption.assert_called_once_with(module.INSTANT_SETTINGS_NOTE)
    local_llm_panel.assert_called_once_with()


def test_render_reindex_settings_marks_every_index_setting_as_needing_a_rebuild(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.subheader, st.caption, st.expander, st.markdown):
        shown.reset_mock()

    module.render_reindex_settings(_fake_settings())

    st.subheader.assert_called_once_with(module.REINDEX_SETTINGS_TITLE)
    st.caption.assert_any_call(module.REINDEX_SETTINGS_NOTE)
    st.expander.assert_called_once_with("Library and index settings")
    lines = [call.args[0] for call in st.markdown.call_args_list]
    assert lines == [
        f"**PDF folder:** `papers` {module.REBUILD_BADGE}",
        f"**Full folder path kept in the index:** `Off` {module.REBUILD_BADGE}",
        f"**Embedding model:** `text-embedding-3-small` {module.REBUILD_BADGE}",
        f"**PDF reader:** `_fake_pdf_reader` {module.REBUILD_BADGE}",
        f"**Chunk size (characters):** `5000` {module.REBUILD_BADGE}",
        f"**Chunk overlap (characters):** `250` {module.REBUILD_BADGE}",
        f"**Whole-page reading:** `Off` {module.REBUILD_BADGE}",
        f"**Images and tables:** `Off` {module.REBUILD_BADGE}",
    ]
    for setting in module.settings_groups.REINDEX_SETTINGS:
        st.caption.assert_any_call(setting.help)


def test_settings_notes_say_which_group_needs_a_rebuild(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    assert "next question" in module.INSTANT_SETTINGS_NOTE
    assert "not rebuilt" in module.INSTANT_SETTINGS_NOTE
    assert "index every PDF again" in module.REINDEX_SETTINGS_NOTE
    assert "rebuild" in module.REBUILD_BADGE.lower()


def test_render_sidebar_draws_instant_settings_then_reindex_settings_then_usage(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.header.reset_mock()
    settings = _fake_settings()
    app_state = {}
    steps = mock.Mock()
    steps.instant.return_value = "Exhaustive"

    with mock.patch.multiple(
        module,
        render_instant_settings=steps.instant,
        render_reindex_settings=steps.reindex,
        render_usage=steps.usage,
    ):
        assert module.render_sidebar(app_state, settings) == "Exhaustive"

    module.st.header.assert_called_once_with("Settings")
    assert steps.mock_calls == [mock.call.instant(), mock.call.reindex(settings), mock.call.usage(app_state)]


def test_page_shows_both_settings_groups_with_the_local_llm_panel_in_the_instant_group():
    with imported_pdf_research_assistant() as module:
        st = module.st
        titles = [call.args[0] for call in st.subheader.call_args_list]
        panels = [call.args[0] for call in st.expander.call_args_list]

        assert titles == [module.INSTANT_SETTINGS_TITLE, module.REINDEX_SETTINGS_TITLE]
        assert panels == ["Active index details", "Local LLM setup", "Library and index settings"]
        module.bootstrap.build_settings.assert_called_once_with()


def test_render_history_shows_a_user_message_without_answer_actions(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    for shown in (st.chat_message, st.markdown, st.caption, st.button):
        shown.reset_mock()

    pdf_research_assistant_module.render_history([{"role": "user", "content": "User question"}])

    st.chat_message.assert_called_once_with("user")
    st.markdown.assert_called_once_with("User question")
    st.button.assert_not_called()
    st.caption.assert_not_called()


def test_render_history_shows_copy_sources_and_cost_for_an_answer(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.button = mock.Mock(return_value=False)
    st.caption.reset_mock()
    st.subheader.reset_mock()
    st.download_button.reset_mock()
    item = {
        "role": "assistant",
        "content": "Answer",
        "cost": 0.25,
        "contexts": [{"name": "Source", "context": "Quoted text"}],
        "id": 7,
    }

    pdf_research_assistant_module.render_history([item])

    st.button.assert_called_once_with("Copy answer", key="copy_history_7")
    st.subheader.assert_called_once_with("Source")
    st.caption.assert_called_once_with("Cost: $0.2500")
    st.download_button.assert_not_called()


def test_render_history_skips_answer_actions_when_an_assistant_item_has_no_cost(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.button = mock.Mock(return_value=False)
    st.caption.reset_mock()

    pdf_research_assistant_module.render_history([{"role": "assistant", "content": "Note"}])

    st.button.assert_not_called()
    st.caption.assert_not_called()


def test_render_history_offers_the_report_for_an_exhaustive_answer(pdf_research_assistant_module, tmp_path):
    st = pdf_research_assistant_module.st
    st.button = mock.Mock(return_value=False)
    st.download_button.reset_mock()
    report = tmp_path / "report.md"
    report.write_text("# Report", encoding="utf-8")

    pdf_research_assistant_module.render_history(
        [{"role": "assistant", "content": "Done", "cost": 0.5, "report_path": str(report)}]
    )

    st.button.assert_called_once_with("Copy answer", key="copy_history_0")
    st.download_button.assert_called_once()


def test_describe_query_failure_includes_the_top_level_and_nested_errors(pdf_research_assistant_module):
    error = ExceptionGroup("group failed", [ValueError("bad pdf")])

    assert pdf_research_assistant_module.describe_query_failure(error) == (
        "Indexing/search failed.\n\n"
        "Top-level error:\n"
        "```\nExceptionGroup: group failed (1 sub-exception)\n```\n\n"
        "Underlying error(s):\n"
        "```\nExceptionGroup: group failed (1 sub-exception)\n  ValueError: bad pdf\n```\n\n"
        "This usually means one document or a parsing/indexing step failed."
    )


def test_answer_question_returns_and_shows_a_successful_answer(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.session_state.update({"history": [{"role": "user", "content": "Q"}], "total_cost": 0.0, "query_count": 0})
    st.button = mock.Mock(return_value=False)
    for shown in (st.markdown, st.caption, st.error):
        shown.reset_mock()
    contexts = [{"name": "Source", "context": "Quoted text"}]

    with mock.patch.object(module, "run_query_subprocess", return_value=("Answer text", 1.5, contexts)) as run_query:
        result = module.answer_question("Q")

    assert result == ("Answer text", 1.5, contexts)
    run_query.assert_called_once_with("Q")
    st.markdown.assert_called_once_with("Answer text")
    st.button.assert_called_once_with("Copy answer", key="copy_current_1")
    st.caption.assert_called_once_with("Cost: $1.5000")
    st.error.assert_not_called()


def test_answer_question_returns_and_shows_diagnostics_when_the_query_fails(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.markdown, st.caption, st.error, st.code):
        shown.reset_mock()
    failure = RuntimeError("query failed")

    with mock.patch.object(module, "run_query_subprocess", side_effect=failure):
        answer, cost, contexts = module.answer_question("Q")

    assert answer == module.describe_query_failure(failure)
    assert (cost, contexts) == (0.0, [])
    st.error.assert_called_once_with(answer)
    st.caption.assert_called_once_with("Cost: $0.0000")
    st.code.assert_called_once_with("trace line\n")
    st.markdown.assert_not_called()


def test_submit_question_records_the_question_answer_and_cost(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.session_state.update({"history": [], "total_cost": 1.0, "query_count": 2})
    st.rerun.reset_mock()
    st.chat_message.reset_mock()
    saved = {}

    with mock.patch.object(module, "answer_question", return_value=("Answer text", 0.5, [])) as answer_question:
        module.submit_question("What is new?", saved)

    answer_question.assert_called_once_with("What is new?")
    assert st.chat_message.call_args_list == [mock.call("user"), mock.call("assistant")]
    assert st.session_state.history == [
        {"role": "user", "content": "What is new?"},
        {"role": "assistant", "id": 1, "content": "Answer text", "cost": 0.5, "contexts": []},
    ]
    assert (st.session_state.total_cost, st.session_state.query_count) == (1.5, 3)
    assert saved == {"history": st.session_state.history, "total_cost": 1.5, "query_count": 3}
    st.rerun.assert_called_once_with()


def test_main_does_nothing_more_when_there_is_no_question_or_search(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.rerun.reset_mock()

    with mock.patch.object(module, "submit_question") as submit_question, mock.patch.object(
        module, "plan_exhaustive_question"
    ) as plan_question:
        module.main()

    submit_question.assert_not_called()
    plan_question.assert_not_called()
    module.st.rerun.assert_not_called()


def test_main_sends_a_query_mode_question_to_submit_question(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.chat_input = mock.Mock(return_value="What is new?")

    with mock.patch.object(module, "submit_question") as submit_question:
        module.main()

    submit_question.assert_called_once_with("What is new?", module.get_persistent_app_state())


# Local LLM setup panel.

SAVED_LOCAL_LLM = {"provider": "openai_compatible", "address": "http://localhost:8080/v1", "model": "qwen2.5"}
READY_MESSAGE = "Ready: it works."


def draw_local_llm_panel(module, *, saved=None, clicked=(), check=(True, READY_MESSAGE), typed=None, provider=None):
    """Draw the panel with chosen saved settings, typed values, clicked buttons and check result."""
    st = module.st
    for shown in (st.caption, st.success, st.error, st.markdown, st.rerun, st.expander, st.selectbox, st.text_input):
        shown.reset_mock()
    st.button = mock.Mock(side_effect=lambda _label, key=None, **_kwargs: key in clicked)
    if provider:
        st.selectbox = mock.Mock(return_value=provider)
    if typed:
        st.text_input = mock.Mock(side_effect=lambda label, value="", **_kwargs: typed.get(label, value))
    with mock.patch.multiple(
        module.local_llm,
        load_config=mock.Mock(return_value=saved),
        check_connection=mock.Mock(return_value=check),
        save_config=mock.DEFAULT,
        clear_config=mock.DEFAULT,
    ) as patched:
        module.render_local_llm_setup()
        return patched | {"check_connection": module.local_llm.check_connection}


def test_local_llm_panel_says_answers_come_from_openai_when_nothing_is_saved(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_local_llm_panel(module)

    module.st.expander.assert_called_once_with("Local LLM setup")
    module.st.caption.assert_called_once_with("Answers come from OpenAI (gpt-4o-mini).")
    module.st.markdown.assert_called_once_with(module.LOCAL_LLM_GUIDE)
    assert [call.kwargs["key"] for call in module.st.button.call_args_list] == ["local_llm_check", "local_llm_save"]
    patched["check_connection"].assert_not_called()
    module.st.success.assert_not_called()


def test_local_llm_panel_starts_with_ollama_and_its_usual_address(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_local_llm_panel(module)

    select_call = module.st.selectbox.call_args
    assert select_call.args[1] == ["ollama", "openai_compatible"]
    assert select_call.kwargs["index"] == 0
    assert select_call.kwargs["format_func"]("openai_compatible") == "OpenAI-compatible server"
    assert module.st.text_input.call_args_list[0].kwargs["value"] == "http://localhost:11434"
    assert module.st.text_input.call_args_list[1].kwargs["value"] == ""


def test_local_llm_panel_shows_the_saved_settings_and_offers_to_switch_back(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_local_llm_panel(module, saved=SAVED_LOCAL_LLM)

    module.st.caption.assert_called_once_with(
        "Answers come from the local model qwen2.5 (OpenAI-compatible server at http://localhost:8080/v1)."
    )
    assert module.st.selectbox.call_args.kwargs["index"] == 1
    assert module.st.text_input.call_args_list[0].kwargs["value"] == "http://localhost:8080/v1"
    assert module.st.text_input.call_args_list[1].kwargs["value"] == "qwen2.5"
    assert module.st.button.call_args_list[-1] == mock.call("Switch back to OpenAI", key="local_llm_clear")


def test_local_llm_panel_offers_the_usual_address_when_another_server_type_is_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_local_llm_panel(module, saved=SAVED_LOCAL_LLM, provider="ollama")

    address_call = module.st.text_input.call_args_list[0]
    assert address_call.kwargs["value"] == "http://localhost:11434"
    assert address_call.kwargs["key"] == "local_llm_address_ollama"


def test_local_llm_check_button_shows_ready_without_saving(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    typed = {"Server address": " http://localhost:11434/ ", "Model name": " llama3.2 "}

    patched = draw_local_llm_panel(module, clicked={"local_llm_check"}, typed=typed)

    patched["check_connection"].assert_called_once_with(
        {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3.2"}
    )
    module.st.success.assert_called_once_with(READY_MESSAGE)
    patched["save_config"].assert_not_called()
    module.st.rerun.assert_not_called()


def test_local_llm_check_button_shows_what_is_wrong(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_local_llm_panel(module, clicked={"local_llm_check"}, check=(False, "Could not reach Ollama."))

    module.st.error.assert_called_once_with("Could not reach Ollama.")
    module.st.success.assert_not_called()


def test_local_llm_use_button_saves_settings_that_pass_the_check(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    typed = {"Model name": "llama3.2"}

    patched = draw_local_llm_panel(module, clicked={"local_llm_save"}, typed=typed)

    patched["save_config"].assert_called_once_with(
        {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3.2"}
    )
    assert module.st.session_state.local_llm_notice == f"{READY_MESSAGE} Answers will now use this local model."
    module.st.rerun.assert_called_once_with()


def test_local_llm_use_button_saves_nothing_when_the_check_fails(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_local_llm_panel(module, clicked={"local_llm_save"}, check=(False, "Model missing."))

    patched["save_config"].assert_not_called()
    module.st.error.assert_called_once_with("Model missing. Nothing was changed.")
    module.st.rerun.assert_not_called()


def test_local_llm_switch_back_button_removes_the_saved_settings(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_local_llm_panel(module, saved=SAVED_LOCAL_LLM, clicked={"local_llm_clear"})

    patched["clear_config"].assert_called_once_with()
    assert module.st.session_state.local_llm_notice == "Answers will now use OpenAI again."
    module.st.rerun.assert_called_once_with()


def test_local_llm_panel_shows_a_saved_notice_once(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.local_llm_notice = "Answers will now use OpenAI again."

    draw_local_llm_panel(module)

    module.st.success.assert_called_once_with("Answers will now use OpenAI again.")
    assert "local_llm_notice" not in module.st.session_state


def test_local_llm_guide_tells_the_user_what_stays_on_openai(pdf_research_assistant_module):
    guide = pdf_research_assistant_module.LOCAL_LLM_GUIDE

    assert "Check connection" in guide
    assert "embeddings" in guide
    assert "OpenAI key" in guide


def test_page_draws_the_local_llm_panel():
    with imported_pdf_research_assistant() as module:
        module.st.expander.assert_any_call("Local LLM setup")
