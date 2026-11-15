from contextlib import contextmanager
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from pdf_research_assistant import rate_profiles


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
        "evidence_k": 10,
        "answer_max_sources": 5,
        "answer_length": "about 200 words, but can be longer",
        "rate_profile": None,
    } | changes
    settings = SimpleNamespace(
        llm=values["llm"],
        summary_llm=values["llm"],
        summary_llm_config={},
        llm_config={},
        embedding=values["embedding"],
        answer=SimpleNamespace(
            evidence_k=values["evidence_k"],
            answer_max_sources=values["answer_max_sources"],
            answer_length=values["answer_length"],
            evidence_summary_length="about 100 words",
            max_concurrent_requests=4,
        ),
        prompts=SimpleNamespace(summary_json_system="system", summary_json="summary", qa="answer", system="system"),
        parsing=SimpleNamespace(
            parse_pdf=values["parse_pdf"], reader_config=values["reader_config"], multimodal=values["multimodal"]
        ),
        agent=SimpleNamespace(
            index=SimpleNamespace(
                paper_directory=values["paper_directory"],
                use_absolute_paper_directory=values["use_absolute_paper_directory"],
            ),
            agent_llm=values["llm"],
            agent_llm_config={},
        ),
    )
    if values["rate_profile"]:
        rate_profiles.apply_profile(settings, values["rate_profile"])
    return settings


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
    index_changes=None,
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
        # So do the Open chunk buttons drawn beside each source passage.
        if str(key or "").startswith("open_chunk_"):
            return False
        # And the button that puts the sidebar settings back to the defaults.
        if str(key or "").startswith("answer_settings_"):
            return False
        # And the one that does the same for the library and index settings.
        if str(key or "").startswith("index_settings_"):
            return False
        return queued_button_values.pop(0)

    streamlit_module.button = mock.Mock(side_effect=click)
    streamlit_module.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: options[index])
    streamlit_module.text_input = mock.Mock(side_effect=lambda _label, value="", **_kwargs: value)
    streamlit_module.number_input = mock.Mock(side_effect=lambda _label, value=None, **_kwargs: value)
    streamlit_module.toggle = mock.Mock(return_value=False)
    streamlit_module.checkbox = mock.Mock(side_effect=lambda _label, value=False, **_kwargs: value)
    streamlit_module.form = mock.Mock(return_value=_ContextManagerStub())
    streamlit_module.form_submit_button = mock.Mock(return_value=False)
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
    streamlit_module.dialog = mock.Mock(return_value=lambda fn: fn)
    streamlit_module.text = mock.Mock()
    streamlit_module.image = mock.Mock()

    # Mock bootstrap lookups so each import scenario can model a specific index and manifest state.
    bootstrap_module.get_allowed_paths = mock.Mock(return_value=set() if allowed_paths is None else allowed_paths)
    bootstrap_module.get_failed_files = mock.Mock(return_value=[] if failed_files is None else failed_files)
    bootstrap_module.get_indexed_doc_count = mock.Mock(return_value=indexed_count)
    bootstrap_module.get_active_index_dir = mock.Mock(return_value=active_index_dir)
    # Like the real build_settings, which always applies a rate profile.
    bootstrap_module.build_settings = mock.Mock(return_value=_fake_settings(rate_profile=rate_profiles.DEFAULT_PROFILE))
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
        # Point the saved sidebar choices at a file that does not exist, so a real saved file never leaks in.
        with mock.patch("subprocess.run", return_value=subprocess_result), mock.patch("time.sleep"), \
                mock.patch("pdf_research_assistant.saved_settings.SETTINGS_PATH", Path("tests/.tmp/no-saved-settings.json")), \
                mock.patch.multiple(
                    "pdf_research_assistant.exhaustive_client", RUNS_DIR=Path("unused"), **(client_patches or {})
                ), \
                mock.patch.multiple(
                    # Never write a record of the index during tests; report the index changes asked for.
                    "pdf_research_assistant.index_record",
                    remember=mock.Mock(return_value=False),
                    find_changes=mock.Mock(return_value=index_changes),
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


def test_render_source_passages_offers_an_open_chunk_button_for_each_passage(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(return_value=False)
    contexts = [{"name": "Paper A", "context": "First"}, {"name": "Paper B", "context": "Second"}]

    with mock.patch.object(module, "show_chunk_details") as show_chunk:
        module.render_source_passages(contexts, key="history_7")

    assert module.st.button.call_args_list == [
        mock.call("Open chunk", key="open_chunk_history_7_0"),
        mock.call("Open chunk", key="open_chunk_history_7_1"),
    ]
    show_chunk.assert_not_called()


def test_render_source_passages_opens_the_chunk_whose_button_was_clicked(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(side_effect=lambda _label, key: key.endswith("_1"))
    contexts = [{"name": "Paper A", "context": "First"}, {"name": "Paper B", "context": "Second"}]

    with mock.patch.object(module, "show_chunk_details") as show_chunk:
        module.render_source_passages(contexts, key="history_7")

    show_chunk.assert_called_once_with(contexts[1])


def test_render_source_passages_keeps_the_chunk_text_out_of_the_passage_list(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(return_value=False)
    module.st.write.reset_mock()

    module.render_source_passages([{"name": "Paper A", "context": "Summary", "raw_text": "Raw chunk"}])

    module.st.write.assert_called_once_with("Summary")


def test_chunk_details_open_in_a_wide_dialog():
    with imported_pdf_research_assistant() as module:
        module.st.dialog.assert_called_once_with("Chunk details", width="large")


def test_describe_chunk_facts_names_the_source_and_says_whether_it_was_cited(pdf_research_assistant_module):
    describe = pdf_research_assistant_module.describe_chunk_facts

    assert describe({"name": "Paper A pages 1-2", "cited": True}) == [
        ("Source", "Paper A pages 1-2"),
        ("Cited in the answer", "yes"),
    ]
    assert describe({"name": "Paper A pages 1-2", "cited": False})[1] == ("Cited in the answer", "no")


def test_describe_chunk_facts_leaves_out_cited_when_it_was_not_recorded(pdf_research_assistant_module):
    assert pdf_research_assistant_module.describe_chunk_facts({}) == [("Source", "Source")]


def test_describe_chunk_facts_adds_the_relevance_score_when_paperqa_set_one(pdf_research_assistant_module):
    facts = pdf_research_assistant_module.describe_chunk_facts({"name": "Paper A", "cited": True, "score": 8})

    assert facts[-1] == ("Relevance score", "8 out of 10")


def test_describe_chunk_facts_leaves_out_the_score_when_none_was_set(pdf_research_assistant_module):
    assert pdf_research_assistant_module.describe_chunk_facts({"name": "Paper A", "score": -1}) == [
        ("Source", "Paper A")
    ]


def test_describe_score_shows_a_score_of_zero(pdf_research_assistant_module):
    assert pdf_research_assistant_module.describe_score({"score": 0}) == "0 out of 10"


@pytest.mark.parametrize("ctx", [{}, {"score": -1}, {"score": None}, {"score": "8"}, {"score": True}])
def test_describe_score_gives_none_when_there_is_no_usable_score(pdf_research_assistant_module, ctx):
    assert pdf_research_assistant_module.describe_score(ctx) is None


def test_render_source_passages_shows_each_passage_score_under_its_source(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(return_value=False)
    module.st.caption.reset_mock()
    contexts = [{"name": "Paper A", "context": "First", "score": 9}, {"name": "Paper B", "context": "Second"}]

    module.render_source_passages(contexts)

    assert module.st.caption.call_args_list == [
        mock.call(module.SCORE_NOTE),
        mock.call("Relevance score: 9 out of 10"),
    ]


def test_render_source_passages_leaves_out_the_score_note_when_no_passage_has_a_score(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(return_value=False)
    module.st.caption.reset_mock()

    module.render_source_passages([{"name": "Paper A", "context": "First", "score": -1}])

    module.st.caption.assert_not_called()


def test_show_chunk_details_explains_the_score_when_one_is_shown(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.caption.reset_mock()

    module.show_chunk_details({"name": "Paper A", "cited": True, "score": 8, "raw_text": "Chunk", "media": []})

    assert module.st.caption.call_args_list[:2] == [
        mock.call("Source: Paper A · Cited in the answer: yes · Relevance score: 8 out of 10"),
        mock.call(module.SCORE_NOTE),
    ]


def test_show_chunk_details_leaves_out_the_score_note_without_a_score(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.caption.reset_mock()

    module.show_chunk_details({"name": "Paper A", "raw_text": "Chunk", "media": []})

    assert mock.call(module.SCORE_NOTE) not in module.st.caption.call_args_list


def test_show_chunk_details_shows_the_facts_then_the_chunk_text_then_the_media(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.caption, st.subheader, st.text):
        shown.reset_mock()
    ctx = {"name": "Paper A pages 1-2", "cited": True, "raw_text": "Costs $5 and *more*", "media": [{"text": "T"}]}

    with mock.patch.object(module, "render_chunk_media") as render_media:
        module.show_chunk_details(ctx)

    st.caption.assert_called_once_with("Source: Paper A pages 1-2 · Cited in the answer: yes")
    assert st.subheader.call_args_list == [mock.call("Chunk text"), mock.call("Images and tables")]
    # Plain text, so nothing in the PDF text is read as Markdown or maths.
    st.text.assert_called_once_with("Costs $5 and *more*")
    render_media.assert_called_once_with([{"text": "T"}])


def test_show_chunk_details_says_so_when_a_passage_has_no_chunk_text_or_media(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.caption, st.text):
        shown.reset_mock()

    with mock.patch.object(module, "render_chunk_media") as render_media:
        module.show_chunk_details({"name": "Paper A"})

    st.text.assert_not_called()
    assert st.caption.call_args_list == [mock.call("Source: Paper A"), mock.call(module.NO_CHUNK_TEXT_NOTE)]
    render_media.assert_called_once_with([])


def test_render_chunk_media_explains_when_a_chunk_has_no_media(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.caption, st.image, st.markdown):
        shown.reset_mock()

    module.render_chunk_media([])

    st.caption.assert_called_once_with(module.NO_MEDIA_NOTE)
    assert "Images and tables" in module.NO_MEDIA_NOTE
    st.image.assert_not_called()
    st.markdown.assert_not_called()


def test_render_chunk_media_shows_the_image_table_text_and_description(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.caption, st.image, st.markdown):
        shown.reset_mock()
    media = {"kind": "table", "page": 3, "image": "aW1hZ2U=", "url": "", "text": "| a | $5 |", "description": "Prices"}

    module.render_chunk_media([media])

    assert st.markdown.call_args_list == [mock.call("**Table, page 3**"), mock.call(r"| a | \$5 |")]
    st.image.assert_called_once_with(b"image")
    st.caption.assert_called_once_with("Prices")


def test_render_chunk_media_shows_a_linked_image_without_text_or_description(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.caption, st.image, st.markdown):
        shown.reset_mock()

    module.render_chunk_media([{"url": "https://example.org/figure.png"}])

    st.markdown.assert_called_once_with("**Image**")
    st.image.assert_called_once_with("https://example.org/figure.png")
    st.caption.assert_not_called()


def test_render_chunk_media_shows_table_text_when_there_is_no_image(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.caption, st.image, st.markdown):
        shown.reset_mock()

    module.render_chunk_media([{"kind": "table", "text": "| a | b |"}])

    assert st.markdown.call_args_list == [mock.call("**Table**"), mock.call("| a | b |")]
    st.image.assert_not_called()


@pytest.mark.parametrize("media", [{"image": "not base64!"}, {"image": "aW1hZ2U="}])
def test_render_chunk_media_says_when_an_image_cannot_be_shown(pdf_research_assistant_module, media):
    module = pdf_research_assistant_module
    st = module.st
    st.caption.reset_mock()
    # Streamlit raises when the stored bytes are not an image it can read.
    st.image = mock.Mock(side_effect=ValueError("cannot identify image"))

    module.render_chunk_media([media])

    st.caption.assert_called_once_with(module.IMAGE_FAILED_NOTE)


@pytest.mark.parametrize(
    ("media", "label"),
    [
        ({"kind": "table", "page": 3}, "Table, page 3"),
        ({"kind": "drawing"}, "Drawing"),
        ({"page": 0}, "Image, page 0"),
        ({"kind": None, "page": None}, "Image"),
    ],
)
def test_describe_media_names_the_kind_and_page_when_known(pdf_research_assistant_module, media, label):
    assert pdf_research_assistant_module.describe_media(media) == label


def test_run_query_subprocess_returns_answer_cost_contexts_and_settings_for_successful_json_payload(
    pdf_research_assistant_module,
):
    recorded = {"index": "pqa_index_x", "values": []}
    result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {"ok": True, "answer": "Answer", "cost": 0.25, "contexts": [{"name": "Source"}], "settings": recorded}
        ),
        stderr="",
    )

    # Mock subprocess.run so the test can supply the exact helper output contract.
    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result) as run_mock:
        answer, cost, contexts, settings = pdf_research_assistant_module.run_query_subprocess("What is new?")

    assert (answer, cost, contexts, settings) == ("Answer", 0.25, [{"name": "Source"}], recorded)
    assert run_mock.call_args.args[0][0] == pdf_research_assistant_module.sys.executable
    assert run_mock.call_args.args[0][-1] == "What is new?"


def test_run_query_subprocess_sends_one_off_settings_after_the_question(pdf_research_assistant_module):
    result = SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "answer": "Answer", "cost": 0.0}), stderr="")

    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result) as run_mock:
        pdf_research_assistant_module.run_query_subprocess("What is new?", {"evidence_k": 3})

    assert run_mock.call_args.args[0][-2:] == ["What is new?", json.dumps({"evidence_k": 3})]


def test_run_query_subprocess_returns_no_settings_when_the_payload_has_none(pdf_research_assistant_module):
    result = SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "answer": "Answer", "cost": 0.0}), stderr="")

    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        answer, cost, contexts, settings = pdf_research_assistant_module.run_query_subprocess("What is new?")

    assert (contexts, settings) == ([], None)


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
        # fixed index settings are Markdown too; they are drawn before the chat.
        fixed_index_settings = [s for s in module.settings_groups.REINDEX_SETTINGS if s.write is None]
        lines_before_chat = 2 + 1 + len(fixed_index_settings)
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

    settings = _fake_settings()

    steps = mock.Mock()
    steps.one_off.return_value = {"evidence_k": 3}
    with mock.patch.multiple(
        module,
        render_local_llm_setup=steps.local_llm,
        render_source_settings=steps.sources,
        render_answer_length=steps.length,
        render_rate_profile=steps.profile,
        render_settings_reset=steps.reset,
        render_one_off_settings=steps.one_off,
    ):
        assert module.render_instant_settings(settings) == ("Exhaustive", {"evidence_k": 3})

    st.subheader.assert_called_once_with(module.INSTANT_SETTINGS_TITLE)
    st.caption.assert_called_once_with(module.INSTANT_SETTINGS_NOTE)
    assert steps.mock_calls == [
        mock.call.sources(settings),
        mock.call.length(settings),
        mock.call.profile(settings),
        mock.call.reset(),
        mock.call.one_off(settings),
        mock.call.local_llm(),
    ]


def draw_index_settings(module, *, settings=None, changes=None, chosen=None, submitted=False, saved=None, clicked=()):
    """Draw the library and index settings, with the user choosing values and perhaps clicking a button.

    chosen maps a control's label to the value the user leaves in it.
    """
    st = module.st
    chosen = chosen or {}
    for shown in (st.subheader, st.caption, st.expander, st.markdown, st.rerun, st.error, st.info, st.success):
        shown.reset_mock()
    st.number_input = mock.Mock(side_effect=lambda label, value=None, **_kwargs: chosen.get(label, value))
    st.selectbox = mock.Mock(side_effect=lambda label, options, index=0, **_kwargs: chosen.get(label, options[index]))
    st.checkbox = mock.Mock(side_effect=lambda label, value=False, **_kwargs: chosen.get(label, value))
    st.form = mock.Mock(return_value=_ContextManagerStub())
    st.form_submit_button = mock.Mock(return_value=submitted)
    st.button = mock.Mock(side_effect=lambda _label, key=None, **_kwargs: key in clicked)
    with mock.patch.multiple(
        module.saved_settings, load=mock.Mock(return_value=saved or {}), save_value=mock.DEFAULT, forget=mock.DEFAULT
    ) as patched:
        module.render_reindex_settings(settings or _fake_settings(), changes)
        return patched


def _control(widget, setting_label):
    return next(call for call in widget.call_args_list if call.args[0].startswith(setting_label))


def test_render_reindex_settings_shows_the_fixed_settings_and_marks_them_as_needing_a_rebuild(
    pdf_research_assistant_module,
):
    module = pdf_research_assistant_module
    st = module.st

    draw_index_settings(module)

    st.subheader.assert_called_once_with(module.REINDEX_SETTINGS_TITLE)
    st.caption.assert_any_call(module.REINDEX_SETTINGS_NOTE)
    st.expander.assert_called_once_with("Library and index settings")
    lines = [call.args[0] for call in st.markdown.call_args_list]
    assert lines == [
        f"**PDF folder:** `papers` {module.REBUILD_BADGE}",
        f"**Full folder path kept in the index:** `Off` {module.REBUILD_BADGE}",
        f"**PDF reader:** `_fake_pdf_reader` {module.REBUILD_BADGE}",
    ]
    for key in ("paper_directory", "use_absolute_paper_directory", "parse_pdf"):
        st.caption.assert_any_call(module.settings_groups.get_setting(key).help)


def test_index_settings_that_can_change_get_a_control_starting_at_the_value_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    get_setting = module.settings_groups.get_setting

    draw_index_settings(module, settings=_fake_settings(reader_config={"chunk_chars": 7000, "overlap": 300}))

    chunk = _control(st.number_input, "Chunk size (characters)")
    assert chunk.args[0] == f"Chunk size (characters) {module.REBUILD_BADGE}"
    assert chunk.kwargs == {
        "min_value": 2000,
        "max_value": 20000,
        "value": 7000,
        "step": 500,
        "help": get_setting("chunk_chars").help,
        "key": "index_setting_chunk_chars_0",
    }
    overlap = _control(st.number_input, "Chunk overlap (characters)")
    assert (overlap.kwargs["min_value"], overlap.kwargs["max_value"], overlap.kwargs["value"]) == (0, 1750, 300)
    assert overlap.kwargs["step"] == 50
    embedding = _control(st.selectbox, "Embedding model")
    assert embedding.args[1] == ["text-embedding-3-small", "text-embedding-3-large"]
    assert embedding.kwargs["index"] == 0
    assert embedding.kwargs["key"] == "index_setting_embedding_0"
    assert embedding.kwargs["help"] == get_setting("embedding").help
    media = _control(st.selectbox, "Images and tables")
    assert media.args[0] == f"Images and tables {module.REBUILD_BADGE}"
    assert media.args[1] == ["off", "on_without_enrichment", "on_with_enrichment"]
    assert media.kwargs["index"] == 0
    assert [media.kwargs["format_func"](value) for value in media.args[1]] == [
        "Off",
        "On without enrichment",
        "On with enrichment",
    ]
    whole_page = _control(st.checkbox, "Whole-page pictures")
    assert whole_page.kwargs == {
        "value": False,
        "help": get_setting("full_page").help,
        "key": "index_setting_full_page_0",
    }


def test_index_controls_sit_in_one_form_so_nothing_changes_until_it_is_saved(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_index_settings(module, chosen={"Chunk size (characters)": 3000})

    module.st.form.assert_called_once_with("index_settings_form")
    module.st.form_submit_button.assert_called_once_with(module.INDEX_SETTINGS_SAVE_LABEL)
    patched["save_value"].assert_not_called()
    module.st.rerun.assert_not_called()


def test_an_index_value_that_is_not_offered_is_kept_as_a_choice(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_index_settings(module, settings=_fake_settings(embedding="sparse"))

    embedding = _control(module.st.selectbox, "Embedding model")
    assert embedding.args[1] == ["text-embedding-3-small", "text-embedding-3-large", "sparse"]
    assert embedding.kwargs["index"] == 2
    assert embedding.kwargs["format_func"]("sparse") == "sparse"


def test_an_index_number_outside_the_limits_starts_at_the_nearest_allowed_value(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_index_settings(module, settings=_fake_settings(reader_config={"chunk_chars": 500, "overlap": 9000}))

    assert _control(module.st.number_input, "Chunk size (characters)").kwargs["value"] == 2000
    assert _control(module.st.number_input, "Chunk overlap (characters)").kwargs["value"] == 1750


def test_saving_index_settings_saves_only_what_changed_and_draws_the_page_again(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_index_settings(
        module,
        chosen={
            f"Chunk size (characters) {module.REBUILD_BADGE}": 3000,
            f"Whole-page pictures {module.REBUILD_BADGE}": True,
        },
        submitted=True,
    )

    assert patched["save_value"].call_args_list == [mock.call("chunk_chars", 3000), mock.call("full_page", True)]
    # The page is drawn again so the warning about the index appears at once.
    module.st.rerun.assert_called_once_with()
    assert module.st.session_state.index_settings_notice == module.INDEX_SETTINGS_SAVED_NOTICE


def test_saving_index_settings_without_a_change_saves_nothing(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_index_settings(module, submitted=True)

    patched["save_value"].assert_not_called()
    module.st.rerun.assert_not_called()
    module.st.info.assert_called_once_with(module.INDEX_SETTINGS_UNCHANGED_NOTE)


def test_images_and_tables_cannot_be_turned_on_without_the_package_that_reads_them(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    chosen = {
        f"Images and tables {module.REBUILD_BADGE}": "on_without_enrichment",
        f"Chunk size (characters) {module.REBUILD_BADGE}": 3000,
    }

    with mock.patch.object(module, "media_reading_available", return_value=False):
        patched = draw_index_settings(module, chosen=chosen, submitted=True)

    module.st.error.assert_called_once_with(module.MEDIA_PACKAGE_NOTE)
    patched["save_value"].assert_not_called()
    module.st.rerun.assert_not_called()


def test_images_and_tables_can_be_turned_on_when_the_package_is_installed(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    chosen = {f"Images and tables {module.REBUILD_BADGE}": "on_with_enrichment"}

    with mock.patch.object(module, "media_reading_available", return_value=True):
        patched = draw_index_settings(module, chosen=chosen, submitted=True)

    patched["save_value"].assert_called_once_with("multimodal", "on_with_enrichment")
    module.st.error.assert_not_called()


def test_turning_images_and_tables_off_never_needs_the_package(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    settings = _fake_settings(multimodal=True)

    with mock.patch.object(module, "media_reading_available", return_value=False):
        patched = draw_index_settings(
            module, settings=settings, chosen={f"Images and tables {module.REBUILD_BADGE}": "off"}, submitted=True
        )

    patched["save_value"].assert_called_once_with("multimodal", "off")


@pytest.mark.parametrize(("found", "available"), [(object(), True), (None, False)])
def test_media_reading_is_available_when_its_package_is_installed(pdf_research_assistant_module, found, available):
    module = pdf_research_assistant_module

    with mock.patch("importlib.util.find_spec", return_value=found) as find_spec:
        assert module.media_reading_available() is available

    find_spec.assert_called_once_with("pypdfium2")


def test_index_settings_show_a_saved_notice_once(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.index_settings_notice = "Saved."

    draw_index_settings(module)
    module.st.success.assert_called_once_with("Saved.")

    draw_index_settings(module)
    module.st.success.assert_not_called()


def test_the_index_defaults_button_is_hidden_until_an_index_setting_is_saved(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_index_settings(module, saved={"evidence_k": 3})

    module.st.button.assert_not_called()


def test_the_index_defaults_button_forgets_every_saved_index_setting(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_index_settings(module, saved={"chunk_chars": 3000}, clicked={"index_settings_reset"})

    module.st.button.assert_called_once_with(module.INDEX_SETTINGS_RESET_LABEL, key="index_settings_reset")
    patched["forget"].assert_called_once_with(["embedding", "chunk_chars", "overlap", "full_page", "multimodal"])
    assert module.st.session_state.index_settings_round == 1
    module.st.rerun.assert_called_once_with()


def test_the_index_defaults_button_changes_nothing_until_clicked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_index_settings(module, saved={"multimodal": "on_without_enrichment"})

    module.st.button.assert_called_once()
    patched["forget"].assert_not_called()
    assert "index_settings_round" not in module.st.session_state


def test_index_controls_get_new_keys_after_going_back_to_the_defaults(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.index_settings_round = 2

    draw_index_settings(module)

    assert _control(module.st.number_input, "Chunk size (characters)").kwargs["key"] == "index_setting_chunk_chars_2"
    assert _control(module.st.checkbox, "Whole-page pictures").kwargs["key"] == "index_setting_full_page_2"


def test_the_index_defaults_leave_the_answer_settings_alone(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    assert not set(module.INDEX_SETTING_KEYS) & set(module.SAVED_SETTING_KEYS)


CHUNK_CHANGES = {
    "index": "pqa_index_old",
    "settings": [
        {"key": "chunk_chars", "label": "Chunk size (characters)", "was": "5000", "now": "3000"},
        {"key": "paper_directory", "label": "PDF folder", "was": "old-papers", "now": "papers"},
    ],
}


def test_render_reindex_settings_says_what_each_changed_setting_was(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_index_settings(module, changes=CHUNK_CHANGES)

    captions = [call.args[0] for call in module.st.caption.call_args_list]
    folder_help = module.settings_groups.get_setting("paper_directory").help
    # Under a fixed setting's explanation, and under a control.
    assert captions[captions.index(folder_help) + 1] == "Changed. The index `pqa_index_old` was built with `old-papers`."
    assert "Changed. The index `pqa_index_old` was built with `5000`." in captions
    assert len([caption for caption in captions if caption.startswith("Changed.")]) == 2


def test_render_reindex_settings_marks_nothing_as_changed_without_changes(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_index_settings(module, changes=None)

    captions = [call.args[0] for call in module.st.caption.call_args_list]
    assert not [caption for caption in captions if caption.startswith("Changed.")]


def test_describe_index_changes_lists_each_change_and_how_to_end_the_warning(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    text = module.describe_index_changes(CHUNK_CHANGES)

    assert text == (
        "**Your library and index settings no longer match the index.** They have changed since the index "
        "`pqa_index_old` was built, so the app is now looking for an index built with the new settings, "
        "and there is none yet.\n\n"
        "- Chunk size (characters): was `5000`, now `3000`\n"
        "- PDF folder: was `old-papers`, now `papers`\n\n"
        + module.INDEX_CHANGES_NOTE
    )


def test_describe_index_changes_leaves_out_the_list_when_no_change_is_known(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    text = module.describe_index_changes({"index": "pqa_index_old", "settings": []})

    assert "- " not in text
    assert text.endswith("there is none yet.\n\n" + module.INDEX_CHANGES_NOTE)


def test_index_changes_note_says_how_to_end_the_warning(pdf_research_assistant_module):
    note = pdf_research_assistant_module.INDEX_CHANGES_NOTE

    assert "put these settings back" in note
    assert "Library and index settings" in note
    assert "index every PDF again" in note


def test_render_index_changes_warns_only_when_there_are_changes(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.warning.reset_mock()

    module.render_index_changes(None)
    module.st.warning.assert_not_called()

    module.render_index_changes(CHUNK_CHANGES)
    module.st.warning.assert_called_once_with(module.describe_index_changes(CHUNK_CHANGES))


def test_page_records_the_active_index_when_it_holds_pdfs_and_shows_no_change_warning():
    active_index_dir = Path(r"C:\index\active-shard")

    with imported_pdf_research_assistant(indexed_count=2, active_index_dir=active_index_dir) as module:
        settings = module.bootstrap.build_settings.return_value
        module.index_record.remember.assert_called_once_with(settings, active_index_dir)
        module.index_record.find_changes.assert_not_called()
        module.st.warning.assert_not_called()


def test_page_warns_about_changed_index_settings_before_the_missing_index_banner():
    active_index_dir = Path(r"C:\index\active-shard")

    with imported_pdf_research_assistant(
        indexed_count=0, active_index_dir=active_index_dir, index_changes=CHUNK_CHANGES
    ) as module:
        settings = module.bootstrap.build_settings.return_value
        module.index_record.remember.assert_not_called()
        module.index_record.find_changes.assert_called_once_with(settings, active_index_dir)
        assert module.st.warning.call_args_list == [
            mock.call(module.describe_index_changes(CHUNK_CHANGES)),
            mock.call("Index not found. The first question will build the PDF index, which may take a long time."),
        ]
        captions = [call.args[0] for call in module.st.caption.call_args_list]
        assert "Changed. The index `pqa_index_old` was built with `5000`." in captions


def test_page_shows_only_the_missing_index_banner_when_no_setting_changed():
    with imported_pdf_research_assistant(indexed_count=0, index_changes=None) as module:
        module.st.warning.assert_called_once_with(
            "Index not found. The first question will build the PDF index, which may take a long time."
        )


def test_settings_notes_say_which_group_needs_a_rebuild(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    assert "next question" in module.INSTANT_SETTINGS_NOTE
    assert "not rebuilt" in module.INSTANT_SETTINGS_NOTE
    assert "index every PDF again" in module.REINDEX_SETTINGS_NOTE
    assert module.INDEX_SETTINGS_SAVE_LABEL in module.REINDEX_SETTINGS_NOTE
    assert "indexed again" in module.INDEX_SETTINGS_SAVED_NOTICE
    assert "pypdfium2" in module.MEDIA_PACKAGE_NOTE
    assert "rebuild" in module.REBUILD_BADGE.lower()


def test_render_sidebar_draws_instant_settings_then_reindex_settings_then_usage(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.header.reset_mock()
    settings = _fake_settings()
    app_state = {}
    steps = mock.Mock()
    steps.instant.return_value = ("Exhaustive", {"evidence_k": 3})

    with mock.patch.multiple(
        module,
        render_instant_settings=steps.instant,
        render_reindex_settings=steps.reindex,
        render_usage=steps.usage,
    ):
        assert module.render_sidebar(app_state, settings, CHUNK_CHANGES) == ("Exhaustive", {"evidence_k": 3})

    module.st.header.assert_called_once_with("Settings")
    assert steps.mock_calls == [
        mock.call.instant(settings),
        mock.call.reindex(settings, CHUNK_CHANGES),
        mock.call.usage(app_state),
    ]


def test_page_shows_both_settings_groups_with_the_local_llm_panel_in_the_instant_group():
    with imported_pdf_research_assistant() as module:
        st = module.st
        titles = [call.args[0] for call in st.subheader.call_args_list]
        panels = [call.args[0] for call in st.expander.call_args_list]

        assert titles == [module.INSTANT_SETTINGS_TITLE, module.REINDEX_SETTINGS_TITLE]
        assert panels == [
            "Active index details",
            module.ONE_OFF_TITLE,
            "Local LLM setup",
            "Library and index settings",
            # Drawn under the chat, after the sidebar.
            module.ESTIMATE_DETAILS_TITLE,
        ]
        # Once for the page, and once more for the estimate, after the sidebar may have saved a change.
        assert module.bootstrap.build_settings.call_args_list == [mock.call(), mock.call()]


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

    assert st.button.call_args_list == [
        mock.call("Copy answer", key="copy_history_7"),
        mock.call("Open chunk", key="open_chunk_history_7_0"),
    ]
    st.subheader.assert_called_once_with("Source")
    st.caption.assert_called_once_with("Cost: $0.2500")
    st.download_button.assert_not_called()


def test_render_history_shows_the_settings_kept_with_an_answer(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(return_value=False)
    recorded = {"index": "pqa_index_x", "values": []}
    item = {"role": "assistant", "content": "Answer", "cost": 0.25, "contexts": [], "id": 7, "settings": recorded}

    with mock.patch.object(module, "render_answer_settings") as render_answer_settings:
        module.render_history([item])

    render_answer_settings.assert_called_once_with(recorded)


def test_render_history_passes_no_settings_for_an_answer_saved_without_them(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(return_value=False)

    with mock.patch.object(module, "render_answer_settings") as render_answer_settings:
        module.render_history([{"role": "assistant", "content": "Answer", "cost": 0.25}])

    render_answer_settings.assert_called_once_with(None)


def test_render_answer_settings_shows_nothing_when_no_settings_were_kept(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.expander.reset_mock()
    st.markdown.reset_mock()

    pdf_research_assistant_module.render_answer_settings(None)

    st.expander.assert_not_called()
    st.markdown.assert_not_called()


def test_render_answer_settings_lists_the_index_and_each_setting_in_an_expander(pdf_research_assistant_module):
    st = pdf_research_assistant_module.st
    st.expander.reset_mock()
    st.markdown.reset_mock()
    recorded = {
        "index": "pqa_index_x",
        "values": [
            {"key": "answer_model", "label": "Answer model", "value": "gpt-4o-mini", "source": None},
            {"key": "evidence_k", "label": "Passages gathered", "value": "3", "source": "This question only"},
        ],
    }

    pdf_research_assistant_module.render_answer_settings(recorded)

    st.expander.assert_called_once_with("Show settings used")
    assert st.markdown.call_args_list == [
        mock.call("**Index:** `pqa_index_x`"),
        mock.call("**Answer model:** `gpt-4o-mini`"),
        mock.call("**Passages gathered:** `3` :gray-badge[This question only]"),
    ]


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
    recorded = {"index": "pqa_index_x", "values": []}
    returned = ("Answer text", 1.5, contexts, recorded)

    with mock.patch.object(module, "run_query_subprocess", return_value=returned) as run_query, mock.patch.object(
        module, "render_answer_settings"
    ) as render_answer_settings:
        result = module.answer_question("Q", {"evidence_k": 3})

    assert result == returned
    run_query.assert_called_once_with("Q", {"evidence_k": 3})
    render_answer_settings.assert_called_once_with(recorded)
    st.markdown.assert_called_once_with("Answer text")
    assert st.button.call_args_list == [
        mock.call("Copy answer", key="copy_current_1"),
        mock.call("Open chunk", key="open_chunk_current_1_0"),
    ]
    st.caption.assert_called_once_with("Cost: $1.5000")
    st.error.assert_not_called()


def test_answer_question_returns_and_shows_diagnostics_when_the_query_fails(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.markdown, st.caption, st.error, st.code):
        shown.reset_mock()
    failure = RuntimeError("query failed")

    with mock.patch.object(module, "run_query_subprocess", side_effect=failure):
        answer, cost, contexts, settings = module.answer_question("Q")

    assert answer == module.describe_query_failure(failure)
    assert (cost, contexts, settings) == (0.0, [], None)
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

    recorded = {"index": "pqa_index_x", "values": []}
    with mock.patch.object(
        module, "answer_question", return_value=("Answer text", 0.5, [], recorded)
    ) as answer_question:
        module.submit_question("What is new?", saved, {"evidence_k": 3})

    answer_question.assert_called_once_with("What is new?", {"evidence_k": 3})
    # The one-off settings are used once: their controls start again, switched off, for the next question.
    assert st.session_state.one_off_round == 1
    assert st.chat_message.call_args_list == [mock.call("user"), mock.call("assistant")]
    assert st.session_state.history == [
        {"role": "user", "content": "What is new?"},
        {"role": "assistant", "id": 1, "content": "Answer text", "cost": 0.5, "contexts": [], "settings": recorded},
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

    with mock.patch.object(module, "submit_question") as submit_question, mock.patch.object(
        module, "render_one_off_settings", return_value={"evidence_k": 3}
    ):
        module.main()

    submit_question.assert_called_once_with("What is new?", module.get_persistent_app_state(), {"evidence_k": 3})


# The rough cost and speed shown before a question is sent.


def draw_estimate(module, *, search_mode="Query", one_off=None, settings=None, local=None):
    """Draw the estimate from freshly built settings, as the page does after the sidebar."""
    st = module.st
    for shown in (st.markdown, st.caption, st.expander):
        shown.reset_mock()
    module.bootstrap.build_settings = mock.Mock(return_value=settings or _fake_settings())
    with mock.patch.object(module.local_llm, "load_config", return_value=local):
        module.render_estimate(search_mode, one_off or {})
    return st.markdown.call_args.args[0]


def test_estimate_shows_a_label_with_a_rough_cost_and_time(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    line = draw_estimate(module)

    assert line.startswith(f"{module.ESTIMATE_LABEL} :")
    assert "roughly less than 1 cent" in line
    module.st.expander.assert_called_once_with(module.ESTIMATE_DETAILS_TITLE)
    shown = [call.args[0] for call in module.st.caption.call_args_list]
    assert shown[0].startswith("PaperQA reads 10 passages")
    assert shown[-1] == module.estimate.ROUGH_NOTE


def test_estimate_uses_the_settings_built_after_the_sidebar(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    line = draw_estimate(module, settings=_fake_settings(evidence_k=30, answer_max_sources=15))

    module.bootstrap.build_settings.assert_called_once_with()
    assert module.estimate.badge(module.estimate.HIGHER_COST) in line


def test_estimate_puts_one_off_values_on_top_of_the_usual_settings(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    usual = draw_estimate(module)
    one_off = draw_estimate(module, one_off={"evidence_k": 30, "answer_max_sources": 15})

    assert module.estimate.badge(module.estimate.HIGHER_COST) not in usual
    assert module.estimate.badge(module.estimate.HIGHER_COST) in one_off


def test_estimate_says_a_local_model_has_no_cost(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    line = draw_estimate(module, local={"model": "llama3.2"})

    assert "no cost for the answer model" in line
    assert module.estimate.LOCAL_COST_NOTE in [call.args[0] for call in module.st.caption.call_args_list]


def test_estimate_for_exhaustive_search_points_to_its_own_estimate(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    line = draw_estimate(module, search_mode=module.EXHAUSTIVE_MODE)

    assert line == f"{module.ESTIMATE_LABEL} {module.estimate.badge(module.estimate.HIGHER_COST)}"
    module.st.caption.assert_called_once_with(module.EXHAUSTIVE_ESTIMATE_NOTE)
    module.bootstrap.build_settings.assert_not_called()


def test_exhaustive_estimate_note_says_it_asks_before_starting(pdf_research_assistant_module):
    assert "confirm" in pdf_research_assistant_module.EXHAUSTIVE_ESTIMATE_NOTE


def test_main_shows_the_estimate_with_the_search_mode_and_one_off_values(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    with mock.patch.object(module, "render_estimate") as render_estimate, mock.patch.object(
        module, "render_one_off_settings", return_value={"evidence_k": 3}
    ):
        module.main()

    render_estimate.assert_called_once_with("Query", {"evidence_k": 3})


@pytest.mark.parametrize("state", ["exhaustive_pending", "exhaustive_run"])
def test_main_hides_the_estimate_while_an_exhaustive_search_is_waiting_or_running(pdf_research_assistant_module, state):
    module = pdf_research_assistant_module
    module.st.session_state[state] = {"question": "q"}

    with mock.patch.object(module, "render_estimate") as render_estimate, mock.patch.multiple(
        module, render_exhaustive_pending=mock.DEFAULT, render_exhaustive_run=mock.DEFAULT
    ):
        module.main()

    render_estimate.assert_not_called()


# Sidebar controls for how many passages are gathered and how many sources an answer may use.


def draw_source_settings(module, *, settings=None, typed=None):
    """Draw the source settings with chosen typed numbers."""
    st = module.st
    st.caption.reset_mock()
    typed = typed or {}
    st.number_input = mock.Mock(side_effect=lambda label, value=None, **_kwargs: typed.get(label, value))
    with mock.patch.multiple(module.saved_settings, save_value=mock.DEFAULT) as patched:
        module.render_source_settings(settings or _fake_settings())
        return patched


def _number_box(module, label):
    return next(call for call in module.st.number_input.call_args_list if call.args[0] == label)


def test_source_settings_start_at_the_values_the_next_question_will_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_source_settings(module, settings=_fake_settings(evidence_k=12, answer_max_sources=4))

    gathered = _number_box(module, "Passages gathered")
    sources = _number_box(module, "Most sources in the answer")
    assert gathered.kwargs == {
        "min_value": 1,
        "max_value": 30,
        "value": 12,
        "step": 1,
        "help": module.settings_groups.get_setting("evidence_k").help,
        # A fixed key keeps the box the same widget when its starting value changes, so no click is lost.
        "key": "setting_evidence_k_0",
    }
    assert sources.kwargs["key"] == "setting_answer_max_sources_0"
    assert (sources.kwargs["min_value"], sources.kwargs["max_value"], sources.kwargs["value"]) == (1, 12, 4)
    assert sources.kwargs["help"] == module.settings_groups.get_setting("answer_max_sources").help
    patched["save_value"].assert_not_called()


def test_source_settings_say_exhaustive_search_ignores_them(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_source_settings(module)

    module.st.caption.assert_any_call(module.SOURCE_SETTINGS_NOTE)
    assert "Exhaustive search ignores" in module.SOURCE_SETTINGS_NOTE


def test_choosing_a_new_number_saves_it_for_the_next_question(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_source_settings(module, typed={"Passages gathered": 20, "Most sources in the answer": 8})

    assert patched["save_value"].call_args_list == [
        mock.call("evidence_k", 20),
        mock.call("answer_max_sources", 8),
    ]


def test_most_sources_can_never_be_more_than_the_passages_gathered(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_source_settings(module, typed={"Passages gathered": 3})

    sources = _number_box(module, "Most sources in the answer")
    assert (sources.kwargs["max_value"], sources.kwargs["value"]) == (3, 3)
    assert patched["save_value"].call_args_list == [mock.call("evidence_k", 3), mock.call("answer_max_sources", 3)]


def test_a_current_value_outside_the_limits_is_brought_inside_and_saved(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_source_settings(module, settings=_fake_settings(evidence_k=50, answer_max_sources=0))

    assert _number_box(module, "Passages gathered").kwargs["value"] == 30
    assert _number_box(module, "Most sources in the answer").kwargs["value"] == 1
    assert patched["save_value"].call_args_list == [mock.call("evidence_k", 30), mock.call("answer_max_sources", 1)]


def draw_settings_reset(module, *, saved=None, clicked=()):
    """Draw the button that puts the saved sidebar settings back to PaperQA's defaults."""
    st = module.st
    st.rerun.reset_mock()
    st.button = mock.Mock(side_effect=lambda _label, key=None, **_kwargs: key in clicked)
    with mock.patch.multiple(
        module.saved_settings, load=mock.Mock(return_value=saved or {}), forget=mock.DEFAULT
    ) as patched:
        module.render_settings_reset()
        return patched


def test_the_defaults_button_is_hidden_until_a_sidebar_setting_is_saved(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_settings_reset(module, saved={"local_llm": "anything"})

    module.st.button.assert_not_called()


@pytest.mark.parametrize(
    "saved", [{"evidence_k": 3}, {"answer_length": "50 to 100 words"}, {"rate_profile": "Low limits"}]
)
def test_the_defaults_button_forgets_every_saved_sidebar_setting(saved, pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_settings_reset(module, saved=saved, clicked={"answer_settings_reset"})

    module.st.button.assert_called_once_with(module.SETTINGS_RESET_LABEL, key="answer_settings_reset")
    patched["forget"].assert_called_once_with(["evidence_k", "answer_max_sources", "answer_length", "rate_profile"])
    # The browser keeps a control's old value, so the controls get new keys and start again from the defaults.
    assert module.st.session_state.answer_settings_round == 1
    module.st.rerun.assert_called_once_with()


def test_controls_get_new_keys_after_going_back_to_the_defaults(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.answer_settings_round = 1

    draw_source_settings(module)
    draw_answer_length(module)

    keys = [call.kwargs["key"] for call in module.st.number_input.call_args_list]
    assert keys == ["setting_evidence_k_1", "setting_answer_max_sources_1"]
    assert module.st.selectbox.call_args.kwargs["key"] == "setting_answer_length_1"
    draw_rate_profile(module)
    assert module.st.selectbox.call_args.kwargs["key"] == "setting_rate_profile_1"


def test_the_defaults_button_changes_nothing_until_clicked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_settings_reset(module, saved={"answer_max_sources": 2})

    assert "answer_settings_round" not in module.st.session_state
    module.st.button.assert_called_once()
    patched["forget"].assert_not_called()
    module.st.rerun.assert_not_called()


# Sidebar control for how long the answer is asked to be.

SHORT = "50 to 100 words"
STANDARD = "about 200 words, but can be longer"
LONG = "about 400 words, but can be longer"


def draw_answer_length(module, *, settings=None, picked=None):
    """Draw the answer length control, with the user picking a phrase or leaving it as it is."""
    st = module.st
    st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: picked or options[index])
    with mock.patch.object(module.saved_settings, "save_value") as save_value:
        module.render_answer_length(settings or _fake_settings())
        return save_value


def test_answer_length_offers_the_named_choices_starting_at_the_one_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_answer_length(module, settings=_fake_settings(answer_length=SHORT))

    call = module.st.selectbox.call_args
    assert call.args == ("Answer length", [SHORT, STANDARD, LONG])
    assert call.kwargs["index"] == 0
    assert call.kwargs["help"] == module.settings_groups.get_setting("answer_length").help
    assert call.kwargs["key"] == "setting_answer_length_0"
    assert [call.kwargs["format_func"](phrase) for phrase in (SHORT, STANDARD, LONG)] == [
        "Short (50 to 100 words)",
        "Standard (about 200 words, but can be longer)",
        "Long (about 400 words, but can be longer)",
    ]
    save_value.assert_not_called()


def test_picking_another_answer_length_saves_it_for_the_next_question(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_answer_length(module, settings=_fake_settings(answer_length=STANDARD), picked=LONG)

    save_value.assert_called_once_with("answer_length", LONG)


def test_an_answer_length_that_is_not_offered_is_kept_until_another_is_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_answer_length(module, settings=_fake_settings(answer_length="one paragraph"))

    call = module.st.selectbox.call_args
    assert call.args[1] == [SHORT, STANDARD, LONG, "one paragraph"]
    assert call.kwargs["index"] == 3
    assert call.kwargs["format_func"]("one paragraph") == "one paragraph"
    save_value.assert_not_called()


def test_page_draws_the_answer_length_from_the_settings_it_built():
    with imported_pdf_research_assistant() as module:
        call = next(c for c in module.st.selectbox.call_args_list if c.args[0] == "Answer length")

        assert call.args[1][call.kwargs["index"]] == STANDARD


def test_page_draws_the_source_settings_from_the_settings_it_built():
    with imported_pdf_research_assistant() as module:
        labels = [call.args[0] for call in module.st.number_input.call_args_list]

        # The index settings further down the sidebar have number boxes too.
        assert labels[:2] == ["Passages gathered", "Most sources in the answer"]
        assert module.st.number_input.call_args_list[0].kwargs["value"] == 10


# Sidebar control for the rate-limit profile.


def draw_rate_profile(module, *, settings=None, picked=None, local=None):
    """Draw the rate profile control, with the user picking a profile or leaving it as it is."""
    st = module.st
    st.caption.reset_mock()
    st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: picked or options[index])
    with mock.patch.object(module.saved_settings, "save_value") as save_value, \
            mock.patch.object(module.local_llm, "load_config", return_value=local):
        module.render_rate_profile(settings or _fake_settings(rate_profile="High limits"))
        return save_value


PROFILE_NAMES = ["Low limits", "Standard", "High limits"]


def test_rate_profile_offers_the_profiles_starting_at_the_one_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_rate_profile(module, settings=_fake_settings(rate_profile="Standard"))

    call = module.st.selectbox.call_args
    assert call.args == ("Speed limits", PROFILE_NAMES)
    assert call.kwargs["index"] == 1
    assert call.kwargs["help"] == module.settings_groups.get_setting("rate_profile").help
    assert call.kwargs["key"] == "setting_rate_profile_0"
    save_value.assert_not_called()


def test_rate_profile_explains_the_profile_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_rate_profile(module, settings=_fake_settings(rate_profile="Low limits"))

    profile = rate_profiles.get_profile("Low limits")
    module.st.caption.assert_called_once_with(rate_profiles.describe_profile(profile))


def test_picking_another_rate_profile_saves_it_for_the_next_question(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_rate_profile(module, settings=_fake_settings(rate_profile="High limits"), picked="Low limits")

    save_value.assert_called_once_with("rate_profile", "Low limits")


def test_limits_that_match_no_profile_are_kept_until_a_profile_is_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_rate_profile(module, settings=_fake_settings())

    call = module.st.selectbox.call_args
    assert call.args[1] == PROFILE_NAMES + [rate_profiles.NO_PROFILE]
    assert call.kwargs["index"] == 3
    module.st.caption.assert_called_once_with(module.NO_PROFILE_NOTE)
    save_value.assert_not_called()


def test_rate_profile_says_what_it_means_for_a_local_model(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    local = {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3.2"}

    draw_rate_profile(module, settings=_fake_settings(rate_profile="Standard"), local=local)

    shown = [call.args[0] for call in module.st.caption.call_args_list]
    assert shown == [rate_profiles.describe_profile(rate_profiles.get_profile("Standard")), module.LOCAL_PROFILE_NOTE]


def test_page_draws_the_rate_profile_from_the_settings_it_built():
    with imported_pdf_research_assistant() as module:
        call = next(c for c in module.st.selectbox.call_args_list if c.args[0] == "Speed limits")

        assert call.args[1][call.kwargs["index"]] == rate_profiles.DEFAULT_PROFILE


# Settings for the next question only, which leave the saved settings alone.


def draw_one_off_settings(module, *, settings=None, switched_on=True, typed=None, picked=None):
    """Draw the one-off settings panel, switched on or off, with chosen numbers and answer length."""
    st = module.st
    for shown in (st.caption, st.expander):
        shown.reset_mock()
    typed = typed or {}
    st.toggle = mock.Mock(return_value=switched_on)
    st.number_input = mock.Mock(side_effect=lambda label, value=None, **_kwargs: typed.get(label, value))
    st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: picked or options[index])
    with mock.patch.object(module.saved_settings, "save_value") as save_value:
        chosen = module.render_one_off_settings(settings or _fake_settings())
    # Nothing chosen here is ever saved as a usual setting.
    save_value.assert_not_called()
    return chosen


def test_one_off_settings_are_off_until_switched_on_and_draw_no_controls(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    assert draw_one_off_settings(module, switched_on=False) == {}

    st = module.st
    st.expander.assert_called_once_with(module.ONE_OFF_TITLE)
    st.caption.assert_called_once_with(module.ONE_OFF_NOTE)
    st.toggle.assert_called_once_with(module.ONE_OFF_TOGGLE_LABEL, key="one_off_enabled_0")
    st.number_input.assert_not_called()
    st.selectbox.assert_not_called()


def test_one_off_note_says_the_values_are_used_once_and_ignored_by_exhaustive_search(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    assert "next question" in module.ONE_OFF_NOTE
    assert "usual settings" in module.ONE_OFF_NOTE
    assert "Exhaustive search ignores" in module.ONE_OFF_NOTE


def test_switched_on_one_off_settings_start_at_the_usual_values_and_change_nothing(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, settings=_fake_settings(evidence_k=12, answer_max_sources=4))

    assert chosen == {}
    gathered = _number_box(module, "Passages gathered")
    sources = _number_box(module, "Most sources in the answer")
    assert (gathered.kwargs["value"], gathered.kwargs["max_value"]) == (12, 30)
    assert (sources.kwargs["value"], sources.kwargs["max_value"]) == (4, 12)
    # The usual value is part of each key, so a change to the usual settings starts the box again from it.
    assert gathered.kwargs["key"] == "one_off_evidence_k_0_12"
    assert sources.kwargs["key"] == "one_off_answer_max_sources_0_4"
    length = module.st.selectbox.call_args
    assert length.args == ("Answer length", [SHORT, STANDARD, LONG])
    assert length.kwargs["index"] == 1
    assert length.kwargs["key"] == f"one_off_answer_length_0_{STANDARD}"
    assert length.kwargs["format_func"](SHORT) == "Short (50 to 100 words)"


def test_one_off_settings_return_only_the_values_that_differ_from_the_usual_ones(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, typed={"Passages gathered": 20}, picked=SHORT)

    assert chosen == {"evidence_k": 20, "answer_length": SHORT}


def test_one_off_sources_can_never_be_more_than_the_one_off_passages(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, typed={"Passages gathered": 3})

    sources = _number_box(module, "Most sources in the answer")
    assert (sources.kwargs["max_value"], sources.kwargs["value"]) == (3, 3)
    assert chosen == {"evidence_k": 3, "answer_max_sources": 3}


def test_one_off_answer_length_offers_a_usual_length_that_is_not_one_of_the_choices(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, settings=_fake_settings(answer_length="one paragraph"))

    length = module.st.selectbox.call_args
    assert length.args[1] == [SHORT, STANDARD, LONG, "one paragraph"]
    assert length.kwargs["index"] == 3
    assert chosen == {}


def test_one_off_controls_get_new_keys_after_a_question_has_used_them(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.one_off_round = 2

    draw_one_off_settings(module)

    module.st.toggle.assert_called_once_with(module.ONE_OFF_TOGGLE_LABEL, key="one_off_enabled_2")
    assert _number_box(module, "Passages gathered").kwargs["key"] == "one_off_evidence_k_2_10"


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
    assert module.st.button.call_args_list[-1] == mock.call(
        "Switch back to OpenAI", key="local_llm_clear", help=module.LOCAL_LLM_CLEAR_HELP
    )


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


def test_local_llm_guide_says_how_a_local_model_differs_from_openai(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    guide = module.LOCAL_LLM_GUIDE

    assert "How this differs from OpenAI" in guide
    assert "no cost" in guide
    assert "slower" in guide
    assert "Speed limits" in guide
    assert "plans the searches" in guide


def test_local_llm_panel_explains_each_input(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_local_llm_panel(module)

    assert module.st.selectbox.call_args.kwargs["help"] == module.LOCAL_LLM_PROVIDER_HELP
    address_call, model_call = module.st.text_input.call_args_list
    assert address_call.kwargs["help"] == module.LOCAL_LLM_ADDRESS_HELP
    assert model_call.kwargs["help"] == module.LOCAL_LLM_MODEL_HELP


def test_local_llm_panel_explains_each_button(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_local_llm_panel(module, saved=SAVED_LOCAL_LLM)

    helps = {call.kwargs["key"]: call.kwargs["help"] for call in module.st.button.call_args_list}
    assert helps == {
        "local_llm_check": module.LOCAL_LLM_CHECK_HELP,
        "local_llm_save": module.LOCAL_LLM_SAVE_HELP,
        "local_llm_clear": module.LOCAL_LLM_CLEAR_HELP,
    }


def test_local_llm_help_names_both_usual_addresses_and_what_the_model_needs(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    for provider in module.local_llm.PROVIDERS.values():
        assert provider["default_address"] in module.LOCAL_LLM_ADDRESS_HELP
    assert "tool calling" in module.LOCAL_LLM_MODEL_HELP
    assert "ollama list" in module.LOCAL_LLM_MODEL_HELP


def test_page_draws_the_local_llm_panel():
    with imported_pdf_research_assistant() as module:
        module.st.expander.assert_any_call("Local LLM setup")
