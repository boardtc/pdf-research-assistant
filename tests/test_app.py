from contextlib import contextmanager
from datetime import datetime
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from pdf_research_assistant import answer_templates, grounding_modes, quality_presets, rate_profiles, research_presets


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


def _template_prompt(name):
    from pdf_research_assistant import answer_templates

    return answer_templates.template_prompt(answer_templates.get_template(name))


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
        "enrichment_llm": "gpt-4o",
        "evidence_k": 10,
        "answer_max_sources": 5,
        "answer_length": "about 200 words, but can be longer",
        "rate_profile": None,
        "answer_template": None,
        "grounding": None,
        "pre_prompt": None,
        "manifest_file": None,
        "recurse_subdirectories": True,
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
            evidence_relevance_score_cutoff=1,
        ),
        prompts=SimpleNamespace(
            summary_json_system="system",
            summary_json="summary",
            qa=_template_prompt(values["answer_template"]) if values["answer_template"] else "answer",
            system="system",
            pre=values["pre_prompt"],
        ),
        parsing=SimpleNamespace(
            parse_pdf=values["parse_pdf"],
            reader_config=values["reader_config"],
            multimodal=values["multimodal"],
            enrichment_llm=values["enrichment_llm"],
        ),
        agent=SimpleNamespace(
            index=SimpleNamespace(
                paper_directory=values["paper_directory"],
                use_absolute_paper_directory=values["use_absolute_paper_directory"],
                manifest_file=values["manifest_file"],
                recurse_subdirectories=values["recurse_subdirectories"],
            ),
            agent_llm=values["llm"],
            agent_llm_config={},
            agent_type="ToolSelector",
        ),
    )
    if values["rate_profile"]:
        rate_profiles.apply_profile(settings, values["rate_profile"])
    if values["grounding"]:
        grounding_modes.apply_mode(settings, values["grounding"])
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
    clicked_keys=(),
    rebuild_patches=None,
    index_names=None,
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
        # And the scope settings' confirm and cancel buttons.
        if str(key or "").startswith("scope_"):
            return key in clicked_keys
        # And the one that does the same for the library and index settings.
        if str(key or "").startswith("index_settings_"):
            return False
        # The rebuild, research preset and answer search buttons are clicked by key, through clicked_keys.
        if str(key or "").startswith(("rebuild_", "research_preset_", "answer_search_")):
            return key in clicked_keys
        return queued_button_values.pop(0)

    streamlit_module.button = mock.Mock(side_effect=click)
    streamlit_module.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: options[index])
    streamlit_module.text_input = mock.Mock(side_effect=lambda _label, value="", **_kwargs: value)
    streamlit_module.text_area = mock.Mock(side_effect=lambda _label, value="", **_kwargs: value)
    streamlit_module.number_input = mock.Mock(side_effect=lambda _label, value=None, **_kwargs: value)
    streamlit_module.toggle = mock.Mock(return_value=False)
    streamlit_module.checkbox = mock.Mock(side_effect=lambda _label, value=False, **_kwargs: value)
    streamlit_module.form = mock.Mock(return_value=_ContextManagerStub())
    streamlit_module.form_submit_button = mock.Mock(return_value=False)
    streamlit_module.success = mock.Mock()
    streamlit_module.expander = mock.Mock(return_value=_ContextManagerStub())
    streamlit_module.container = mock.Mock(return_value=_ContextManagerStub())
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
    streamlit_module.multiselect = mock.Mock(side_effect=lambda _label, options, default=None, **_kwargs: list(default or []))
    streamlit_module.file_uploader = mock.Mock(return_value=None)
    streamlit_module.fragment = mock.Mock(return_value=lambda fn: fn)
    streamlit_module.dialog = mock.Mock(return_value=lambda fn: fn)
    streamlit_module.text = mock.Mock()
    streamlit_module.image = mock.Mock()

    # Mock bootstrap lookups so each import scenario can model a specific index and manifest state.
    bootstrap_module.get_allowed_paths = mock.Mock(return_value=set() if allowed_paths is None else allowed_paths)
    bootstrap_module.saved_scope = mock.Mock(return_value=("manifest", True))
    bootstrap_module.manifest_exists = mock.Mock(return_value=bool(allowed_paths))
    bootstrap_module.load_allowed_manifest_paths = mock.Mock(return_value=set(allowed_paths or ()))
    bootstrap_module.get_index_entries = mock.Mock(return_value=set())
    bootstrap_module.get_failed_files = mock.Mock(return_value=[] if failed_files is None else failed_files)
    bootstrap_module.get_indexed_doc_count = mock.Mock(return_value=indexed_count)
    bootstrap_module.get_active_index_dir = mock.Mock(return_value=active_index_dir)
    # Like the real build_settings, which always applies a rate profile.
    bootstrap_module.build_settings = mock.Mock(return_value=_fake_settings(
        rate_profile=rate_profiles.DEFAULT_PROFILE,
        answer_template=answer_templates.DEFAULT_TEMPLATE,
        grounding=grounding_modes.DEFAULT_MODE,
    ))
    bootstrap_module.MODEL_NAME = "gpt-4o-mini"
    bootstrap_module.MANIFEST_PATH = Path(r"D:\papers-list\manifest.csv")
    bootstrap_module.PAPER_DIR = Path(r"D:\papers")
    bootstrap_module.INDEX_DIR = active_index_dir.parent

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
                mock.patch("pdf_research_assistant.saved_settings.SETTINGS_PATH", Path("tests/.tmp/no-saved-settings.json")),                 mock.patch("pdf_research_assistant.research_presets.PRESETS_PATH", Path("tests/.tmp/no-presets.json")), \
                mock.patch.multiple(
                    "pdf_research_assistant.exhaustive_client", RUNS_DIR=Path("unused"), **(client_patches or {})
                ), \
                mock.patch.multiple(
                    # No rebuild is running unless a test says so, and none is ever started for real.
                    "pdf_research_assistant.rebuild_client",
                    **(
                        {
                            "get_run": mock.Mock(return_value=None),
                            "take_result": mock.Mock(return_value=None),
                            "collect_result": mock.Mock(return_value=None),
                            "start": mock.Mock(),
                            "count_pdfs": mock.Mock(return_value=0),
                        }
                        | (rebuild_patches or {})
                    ),
                ), \
                mock.patch.multiple(
                    # Never write a record of the index during tests; report the index changes asked for.
                    "pdf_research_assistant.index_record",
                    remember=mock.Mock(return_value=False),
                    find_changes=mock.Mock(return_value=index_changes),
                ), \
                mock.patch.multiple(
                    # Read only the index names a test gives, and never save one for real.
                    "pdf_research_assistant.index_names",
                    load=mock.Mock(return_value=dict(index_names or {})),
                    save_name=mock.Mock(),
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
        # Once for an exhaustive search and once for an index rebuild.
        assert module.st.fragment.call_args_list == [mock.call(run_every=module.PROGRESS_POLL_SECONDS)] * 2


def test_escape_dollars_stops_streamlit_reading_amounts_as_maths(pdf_research_assistant_module):
    result = pdf_research_assistant_module.escape_dollars("$0.01 to $0.03")

    assert result == r"\$0.01 to \$0.03"


def test_escape_dollars_leaves_an_already_escaped_dollar_alone(pdf_research_assistant_module):
    escape = pdf_research_assistant_module.escape_dollars

    assert escape(r"\$5 and $10") == r"\$5 and \$10"
    assert escape(escape("$5 and $10")) == r"\$5 and \$10"


def test_escape_dollars_leaves_text_without_dollars_alone(pdf_research_assistant_module):
    assert pdf_research_assistant_module.escape_dollars("No money here") == "No money here"


def test_pending_plan_escapes_dollar_amounts():
    state = _empty_state() | {"exhaustive_pending": {"question": "Cost of $5 and $6?", "plan": READY_PLAN}}

    with imported_pdf_research_assistant(app_state=state, button_values=[False, False, False]) as module:
        text = _markdown_text(module)

    assert r"Estimated cost: \$0.01 to \$0.03, typically \$0.02" in text
    assert r"Cost of \$5 and \$6?" in text


def test_finished_search_keeps_the_message_unescaped_so_it_copies_as_typed(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(FINISHED_SUMMARY), encoding="utf-8")
    state = _empty_state() | {"exhaustive_run": _active_run(tmp_path, 0)}

    with imported_pdf_research_assistant(app_state=state) as module:
        content = module.st.session_state.history[-1]["content"]
        assert "Actual cost: $0.5000" in content
        assert r"\$" not in content


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


def test_render_chunk_media_shows_a_picture_kept_beside_a_saved_answer(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.caption, st.image, st.markdown):
        shown.reset_mock()
    st.image = mock.Mock()
    name = "a" * 64

    with mock.patch.object(module.saved_answers, "load_picture", return_value=b"page") as load_picture:
        module.render_chunk_media([{"kind": "image", "page": 2, "picture": name, "url": ""}])

    load_picture.assert_called_once_with(module.bootstrap.INDEX_DIR, name)
    st.markdown.assert_called_once_with("**Image, page 2**")
    st.image.assert_called_once_with(b"page")
    st.caption.assert_not_called()


def test_render_chunk_media_says_when_a_kept_picture_is_gone(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.caption.reset_mock()
    st.image = mock.Mock()

    with mock.patch.object(module.saved_answers, "load_picture", side_effect=FileNotFoundError("gone")):
        module.render_chunk_media([{"picture": "a" * 64}])

    st.image.assert_not_called()
    st.caption.assert_called_once_with(module.IMAGE_FAILED_NOTE)


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
        answer, cost, contexts, settings, save_error = pdf_research_assistant_module.run_query_subprocess("What is new?")

    assert (answer, cost, contexts, settings, save_error) == ("Answer", 0.25, [{"name": "Source"}], recorded, None)
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
        answer, cost, contexts, settings, save_error = pdf_research_assistant_module.run_query_subprocess("What is new?")

    assert (contexts, settings, save_error) == ([], None, None)


def test_run_query_subprocess_passes_on_why_the_answer_could_not_be_kept(pdf_research_assistant_module):
    payload = {"ok": True, "answer": "Answer", "cost": 0.0, "save_error": "OSError: disk full"}
    result = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        returned = pdf_research_assistant_module.run_query_subprocess("What is new?")

    assert returned[-1] == "OSError: disk full"


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
        with pytest.raises(pdf_research_assistant_module.QueryFailedError, match="ValueError: bad payload") as caught:
            pdf_research_assistant_module.run_query_subprocess("What is new?")
    # A payload from before failed questions carried a cost counts as spending nothing.
    assert caught.value.cost == 0.0


def test_run_query_subprocess_passes_on_what_a_failed_question_spent(pdf_research_assistant_module):
    payload = {"ok": False, "error_type": "RateLimitError", "error": "slow down", "cost": 0.0042}
    result = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    with mock.patch.object(pdf_research_assistant_module.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="RateLimitError: slow down") as caught:
            pdf_research_assistant_module.run_query_subprocess("What is new?")

    assert caught.value.cost == 0.0042


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
        module.st.info.assert_called_once_with(
            "A rebuild starts only when you click **Rebuild the index** or submit a question below."
        )


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
        module.st.info.assert_called_once_with("No paper list is in use, so all PDFs under PAPER_DIR are in scope.")


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
        # fixed index settings are Markdown too, as are the research preset line, the save form's title and
        # the import title; they are drawn before the chat.
        fixed_index_settings = [s for s in module.settings_groups.REINDEX_SETTINGS if s.write is None]
        lines_before_chat = 2 + 2 + 2 + len(fixed_index_settings)
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
                ("info", "A rebuild starts only when you click **Rebuild the index** or submit a question below."),
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
                ("info", "No paper list is in use, so all PDFs under PAPER_DIR are in scope."),
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
        "name": None,
        "folder_name": "pqa_index_abc",
        "folder": str(Path("index") / "pqa_index_abc"),
        "paper_list": "`manifest.csv` (3 PDFs)",
        "paper_list_path": str(Path("lists") / "manifest.csv"),
        "paper_dir": "papers",
    }


def test_describe_active_index_keeps_the_friendly_name_given_to_the_index(pdf_research_assistant_module):
    details = pdf_research_assistant_module.describe_active_index(
        Path("index") / "pqa_index_abc", Path("manifest.csv"), 0, Path("papers"), "My library"
    )

    assert details["name"] == "My library"
    assert details["folder_name"] == "pqa_index_abc"


def test_describe_active_index_says_all_pdfs_are_in_scope_without_a_manifest(pdf_research_assistant_module):
    details = pdf_research_assistant_module.describe_active_index(
        Path("index") / "pqa_index_abc", Path("manifest.csv"), 0, Path("papers")
    )

    assert details["paper_list"] == "none, so all PDFs under PAPER_DIR are in scope"
    assert details["paper_list_path"] is None


def _active_index_details(**changes):
    return {
        "name": None,
        "folder_name": "pqa_index_abc",
        "folder": "C:/index/pqa_index_abc",
        "paper_list": "`manifest.csv` (3 PDFs)",
        "paper_list_path": "D:/lists/manifest.csv",
        "paper_dir": "D:/papers",
    } | changes


def draw_active_index(module, details, *, typed=None, submitted=False):
    """Draw the active index line and details, with the user perhaps typing a name and saving it."""
    st = module.st
    for shown in (st.caption, st.markdown, st.expander, st.success, st.error, st.rerun):
        shown.reset_mock()
    st.form = mock.Mock(return_value=_ContextManagerStub())
    st.text_input = mock.Mock(side_effect=lambda _label, value="", **_kwargs: value if typed is None else typed)
    st.form_submit_button = mock.Mock(return_value=submitted)
    module.index_names.save_name.reset_mock()
    module.render_active_index(details)


def test_describe_index_name_shows_the_folder_when_the_index_has_no_name(pdf_research_assistant_module):
    assert pdf_research_assistant_module.describe_index_name("pqa_index_abc") == "`pqa_index_abc`"


def test_describe_index_name_puts_the_friendly_name_first_and_keeps_the_folder(pdf_research_assistant_module):
    assert (
        pdf_research_assistant_module.describe_index_name("pqa_index_abc", "My library")
        == "**My library** (`pqa_index_abc`)"
    )


def test_escape_markdown_stops_a_name_being_read_as_formatting(pdf_research_assistant_module):
    assert pdf_research_assistant_module.escape_markdown("My *big* [x]:red `y`") == r"My \*big\* \[x\]\:red \`y\`"


def test_render_active_index_shows_the_folder_and_puts_the_paths_and_name_box_in_details(
    pdf_research_assistant_module,
):
    module = pdf_research_assistant_module

    draw_active_index(module, _active_index_details())

    st = module.st
    st.caption.assert_called_once_with("Active index: `pqa_index_abc` · Paper list: `manifest.csv` (3 PDFs)")
    st.expander.assert_called_once_with("Active index details")
    assert st.markdown.call_args_list == [
        mock.call("**Index folder:** `C:/index/pqa_index_abc`"),
        mock.call("**Paper list:** `D:/lists/manifest.csv`"),
        mock.call("**PDF folder:** `D:/papers`"),
    ]
    st.form.assert_called_once_with("index_name_form")
    name_box = st.text_input.call_args
    assert name_box.args == (module.INDEX_NAME_LABEL,)
    assert name_box.kwargs["value"] == ""
    assert name_box.kwargs["key"] == "index_name_pqa_index_abc"
    assert name_box.kwargs["max_chars"] == module.index_names.MAX_LENGTH
    assert name_box.kwargs["help"] == module.INDEX_NAME_HELP
    st.form_submit_button.assert_called_once_with(module.INDEX_NAME_SAVE_LABEL)
    module.index_names.save_name.assert_not_called()
    st.success.assert_not_called()


def test_render_active_index_shows_the_friendly_name_and_starts_the_box_with_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_active_index(module, _active_index_details(name="My *big* library"))

    module.st.caption.assert_called_once_with(
        r"Active index: **My \*big\* library** · Paper list: `manifest.csv` (3 PDFs)"
    )
    module.st.markdown.assert_any_call("**Index folder:** `C:/index/pqa_index_abc`")
    assert module.st.text_input.call_args.kwargs["value"] == "My *big* library"


def test_render_active_index_skips_the_paper_list_path_without_a_manifest(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_active_index(
        module,
        _active_index_details(paper_list="none, so all PDFs under PAPER_DIR are in scope", paper_list_path=None),
    )

    assert module.st.markdown.call_args_list == [
        mock.call("**Index folder:** `C:/index/pqa_index_abc`"),
        mock.call("**PDF folder:** `D:/papers`"),
    ]


def test_saving_a_name_saves_it_for_the_active_folder_and_draws_the_page_again(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_active_index(module, _active_index_details(), typed="My library", submitted=True)

    module.index_names.save_name.assert_called_once_with(Path("C:/index/pqa_index_abc"), "My library")
    assert module.st.session_state.index_name_notice == module.INDEX_NAME_SAVED_NOTICE
    module.st.rerun.assert_called_once_with()
    module.st.error.assert_not_called()


def test_saving_an_empty_name_says_the_folder_name_is_shown_again(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_active_index(module, _active_index_details(name="My library"), typed="  ", submitted=True)

    module.index_names.save_name.assert_called_once_with(Path("C:/index/pqa_index_abc"), "  ")
    assert module.st.session_state.index_name_notice == module.INDEX_NAME_REMOVED_NOTICE
    module.st.rerun.assert_called_once_with()


def test_saving_a_name_that_is_refused_says_why_and_keeps_the_page(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.index_names.save_name.side_effect = ValueError("“My *library*” is taken.")
    try:
        module.st.error.reset_mock()
        draw_active_index(module, _active_index_details(), typed="My *library*", submitted=True)
    finally:
        module.index_names.save_name.side_effect = None

    module.st.error.assert_called_once_with(r"“My \*library\*” is taken\.")
    module.st.rerun.assert_not_called()
    assert "index_name_notice" not in module.st.session_state


def test_saving_a_name_that_cannot_be_written_says_so(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.index_names.save_name.side_effect = OSError("read-only")
    try:
        draw_active_index(module, _active_index_details(), typed="My library", submitted=True)
    finally:
        module.index_names.save_name.side_effect = None

    module.st.error.assert_called_once_with(rf"{module.INDEX_NAME_SAVE_FAILED_NOTICE} read\-only")
    module.st.rerun.assert_not_called()


def test_render_active_index_shows_the_notice_left_by_saving_a_name_once(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.index_name_notice = module.INDEX_NAME_SAVED_NOTICE

    draw_active_index(module, _active_index_details(name="My library"))

    module.st.success.assert_called_once_with(module.INDEX_NAME_SAVED_NOTICE)
    assert "index_name_notice" not in module.st.session_state


def test_page_shows_the_friendly_name_of_the_active_index():
    with imported_pdf_research_assistant(
        active_index_dir=Path(r"C:\index\active-shard"), index_names={"active-shard": "My library"}
    ) as module:
        module.index_names.load.assert_any_call(Path(r"C:\index"))
        captions = [call.args[0] for call in module.st.caption.call_args_list]
        assert "Active index: **My library** · Paper list: none, so all PDFs under PAPER_DIR are in scope" in captions


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
        render_quality_preset=steps.preset,
        render_source_settings=steps.sources,
        render_answer_length=steps.length,
        render_answer_template=steps.template,
        render_grounding=steps.grounding,
        render_pre_prompt=steps.pre_prompt,
        render_rate_profile=steps.profile,
        render_settings_reset=steps.reset,
        render_one_off_settings=steps.one_off,
    ):
        assert module.render_instant_settings(settings) == ("Exhaustive", {"evidence_k": 3})

    st.subheader.assert_called_once_with(module.INSTANT_SETTINGS_TITLE)
    st.caption.assert_called_once_with(module.INSTANT_SETTINGS_NOTE)
    assert steps.mock_calls == [
        mock.call.preset(settings),
        mock.call.sources(settings),
        mock.call.length(settings),
        mock.call.template(settings),
        mock.call.grounding(settings),
        mock.call.pre_prompt(settings),
        mock.call.profile(settings),
        mock.call.reset(),
        mock.call.one_off(settings),
        mock.call.local_llm(),
    ]


def draw_index_settings(
    module, *, settings=None, changes=None, chosen=None, submitted=False, saved=None, clicked=(), busy=False
):
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
        module.render_reindex_settings(settings or _fake_settings(), changes, busy)
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
    module.st.form_submit_button.assert_called_once_with(module.INDEX_SETTINGS_SAVE_LABEL, disabled=False)
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

    assert "index_settings_reset" not in _button_keys(module)


def test_the_index_defaults_button_forgets_every_saved_index_setting(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_index_settings(module, saved={"chunk_chars": 3000}, clicked={"index_settings_reset"})

    module.st.button.assert_any_call(module.INDEX_SETTINGS_RESET_LABEL, key="index_settings_reset")
    patched["forget"].assert_called_once_with(["embedding", "chunk_chars", "overlap", "full_page", "multimodal"])
    assert module.st.session_state.index_settings_round == 1
    module.st.rerun.assert_called_once_with()


def test_the_index_defaults_button_changes_nothing_until_clicked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_index_settings(module, saved={"multimodal": "on_without_enrichment"})

    assert "index_settings_reset" in _button_keys(module)
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


def test_render_reindex_settings_explains_when_images_and_tables_are_worth_turning_on(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    changes = {"index": "pqa_index_old", "settings": [{"key": "multimodal", "label": "Images and tables", "was": "On"}]}

    draw_index_settings(module, changes=changes)

    captions = [call.args[0] for call in module.st.caption.call_args_list]
    assert captions.count(module.MEDIA_GUIDE) == 1
    # Below the setting's own changed note, so that note stays next to its control.
    changed_note = "Changed. The index `pqa_index_old` was built with `On`."
    assert captions.index(changed_note) + 1 == captions.index(module.MEDIA_GUIDE)


def test_the_images_and_tables_guide_says_which_picture_mode_catches_what(pdf_research_assistant_module):
    guide = pdf_research_assistant_module.MEDIA_GUIDE

    for phrase in ("charts", "tables", "Whole-page pictures", "ruled lines", "Leave it off"):
        assert phrase in guide


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


def test_describe_index_changes_names_the_old_index_by_its_friendly_name(pdf_research_assistant_module):
    text = pdf_research_assistant_module.describe_index_changes(CHUNK_CHANGES | {"name": "Old library"})

    assert "since the index **Old library** (`pqa_index_old`) was built" in text


def test_page_names_the_old_index_by_its_friendly_name_in_the_warning_and_the_changed_notes():
    with imported_pdf_research_assistant(
        indexed_count=0, index_changes=CHUNK_CHANGES, index_names={"pqa_index_old": "Old library"}
    ) as module:
        module.st.warning.assert_any_call(module.describe_index_changes(CHUNK_CHANGES | {"name": "Old library"}))
        captions = [call.args[0] for call in module.st.caption.call_args_list]
        assert "Changed. The index **Old library** (`pqa_index_old`) was built with `5000`." in captions


def test_index_changes_note_says_how_to_end_the_warning(pdf_research_assistant_module):
    note = pdf_research_assistant_module.INDEX_CHANGES_NOTE

    assert "put these settings back" in note
    assert "Library and index settings" in note
    assert "index every PDF again" in note
    assert f"**{pdf_research_assistant_module.REBUILD_LABEL}**" in note


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
    assert f"**{module.REBUILD_LABEL}**" in module.INDEX_SETTINGS_SAVED_NOTICE
    assert "pypdfium2" in module.MEDIA_PACKAGE_NOTE
    assert "rebuild" in module.REBUILD_BADGE.lower()


def test_render_sidebar_draws_instant_then_reindex_settings_then_usage_then_research_presets_at_the_top(
    pdf_research_assistant_module,
):
    module = pdf_research_assistant_module
    module.st.header.reset_mock()
    settings = _fake_settings()
    app_state = {}
    steps = mock.Mock()
    steps.instant.return_value = ("Exhaustive", {"evidence_k": 3})

    with mock.patch.multiple(
        module,
        render_research_presets=steps.presets,
        render_instant_settings=steps.instant,
        render_reindex_settings=steps.reindex,
        render_scope_settings=steps.scope,
        render_usage=steps.usage,
    ):
        assert module.render_sidebar(app_state, settings, CHUNK_CHANGES, True) == ("Exhaustive", {"evidence_k": 3})

    module.st.header.assert_called_once_with("Settings")
    # The presets go in a slot kept under the heading, but are drawn last, after every control has saved.
    slot = module.st.container.return_value
    assert steps.mock_calls == [
        mock.call.instant(settings),
        mock.call.reindex(settings, CHUNK_CHANGES, True),
        mock.call.scope(settings),
        mock.call.usage(app_state),
        mock.call.presets(slot),
    ]


def test_page_shows_both_settings_groups_with_the_local_llm_panel_in_the_instant_group():
    with imported_pdf_research_assistant() as module:
        st = module.st
        titles = [call.args[0] for call in st.subheader.call_args_list]
        panels = [call.args[0] for call in st.expander.call_args_list]

        # The research presets are drawn last, into a slot at the top of the sidebar.
        assert titles == [module.INSTANT_SETTINGS_TITLE, module.REINDEX_SETTINGS_TITLE, module.RESEARCH_PRESETS_TITLE]
        assert panels == [
            "Active index details",
            module.ONE_OFF_TITLE,
            "Local LLM setup",
            "Library and index settings",
            module.SCOPE_SETTINGS_PANEL,
            module.RESEARCH_PRESETS_PANEL,
            module.RESEARCH_PRESET_FILES_PANEL,
            # Drawn above the chat, after the sidebar.
            module.ANSWER_SEARCH_TITLE,
            # Drawn under the chat.
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


def test_render_history_escapes_dollars_but_copies_the_text_as_typed(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.button = mock.Mock(return_value=False)
    st.markdown.reset_mock()
    items = [
        {"role": "user", "content": "Is it $5 or $10?"},
        {"role": "assistant", "content": "Between $5 and $10.", "cost": 0.25, "id": 3},
    ]

    with mock.patch.object(module, "render_copy_button") as copy_button:
        module.render_history(items)

    assert [call.args[0] for call in st.markdown.call_args_list] == [r"Is it \$5 or \$10?", r"Between \$5 and \$10."]
    copy_button.assert_called_once_with("Between $5 and $10.", key="copy_history_3")


def test_render_history_does_not_escape_an_exhaustive_answer_kept_already_escaped(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.button = mock.Mock(return_value=False)
    st.markdown.reset_mock()

    module.render_history([{"role": "assistant", "content": r"Cost \$5 to \$10", "cost": 0.5, "id": 0}])

    assert st.markdown.call_args_list[0] == mock.call(r"Cost \$5 to \$10")


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


def test_render_answer_settings_names_the_index_by_its_friendly_name(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.markdown.reset_mock()
    module.index_names.load.reset_mock()
    module.index_names.load.return_value = {"pqa_index_x": "My library"}
    try:
        module.render_answer_settings({"index": "pqa_index_x", "values": []})
    finally:
        module.index_names.load.return_value = {}

    module.index_names.load.assert_called_once_with(module.bootstrap.INDEX_DIR)
    module.st.markdown.assert_called_once_with("**Index:** **My library** (`pqa_index_x`)")


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
    returned = ("Answer text", 1.5, contexts, recorded, "OSError: disk full")

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


def test_answer_question_escapes_dollars_on_screen_but_copies_the_text_as_typed(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.session_state.update({"history": [], "total_cost": 0.0, "query_count": 0})
    st.markdown.reset_mock()

    with mock.patch.object(
        module, "run_query_subprocess", return_value=("From $5 to $10", 0.1, [], None, None)
    ), mock.patch.object(module, "render_copy_button") as copy_button, mock.patch.object(
        module, "render_source_passages"
    ), mock.patch.object(module, "render_answer_settings"):
        module.answer_question("Q")

    st.markdown.assert_called_once_with(r"From \$5 to \$10")
    copy_button.assert_called_once_with("From $5 to $10", key="copy_current_0")


def test_answer_question_returns_and_shows_diagnostics_when_the_query_fails(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.markdown, st.caption, st.error, st.code):
        shown.reset_mock()
    failure = RuntimeError("query failed")

    with mock.patch.object(module, "run_query_subprocess", side_effect=failure):
        answer, cost, contexts, settings, save_error = module.answer_question("Q")

    assert answer == module.describe_query_failure(failure)
    assert (cost, contexts, settings, save_error) == (0.0, [], None, None)
    st.error.assert_called_once_with(answer)
    st.caption.assert_called_once_with("Cost: $0.0000")
    st.code.assert_called_once_with("trace line\n")
    st.markdown.assert_not_called()


def test_answer_question_shows_and_returns_what_a_failed_question_spent(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.markdown, st.caption, st.error, st.code):
        shown.reset_mock()
    failure = module.QueryFailedError("RateLimitError: slow down", cost=0.0042)

    with mock.patch.object(module, "run_query_subprocess", side_effect=failure):
        answer, cost, contexts, settings, save_error = module.answer_question("Q")

    assert answer == module.describe_query_failure(failure)
    assert (cost, contexts, settings, save_error) == (0.0042, [], None, None)
    st.error.assert_called_once_with(answer)
    st.caption.assert_called_once_with("Cost: $0.0042")


def test_answer_question_shows_a_tiny_failed_cost_as_less_than_a_hundredth_of_a_cent(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.caption.reset_mock()
    failure = module.QueryFailedError("APIError: down", cost=0.00002)

    with mock.patch.object(module, "run_query_subprocess", side_effect=failure):
        _, cost, _, _, _ = module.answer_question("Q")

    assert cost == 0.00002
    st.caption.assert_called_once_with("Cost: less than $0.0001")


def test_submit_question_escapes_dollars_in_the_question_but_records_it_as_typed(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.session_state.update({"history": [], "total_cost": 0.0, "query_count": 0})
    st.markdown.reset_mock()

    with mock.patch.object(module, "answer_question", return_value=("A", 0.0, [], None, None)):
        module.submit_question("Is it $5 or $10?", {})

    st.markdown.assert_called_once_with(r"Is it \$5 or \$10?")
    assert st.session_state.history[0] == {"role": "user", "content": "Is it $5 or $10?"}


def test_submit_question_records_the_question_answer_and_cost(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.session_state.update({"history": [], "total_cost": 1.0, "query_count": 2})
    st.rerun.reset_mock()
    st.chat_message.reset_mock()
    saved = {}

    recorded = {"index": "pqa_index_x", "values": []}
    with mock.patch.object(
        module, "answer_question", return_value=("Answer text", 0.5, [], recorded, None)
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


def test_estimate_counts_pictures_when_images_and_tables_are_on(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    usual = draw_estimate(module)
    with_pictures = draw_estimate(module, settings=_fake_settings(multimodal=True))

    assert module.estimate.badge(module.estimate.HIGHER_COST) not in usual
    assert module.estimate.badge(module.estimate.HIGHER_COST) in with_pictures
    shown = [call.args[0] for call in module.st.caption.call_args_list]
    assert shown[1].startswith("Images and tables are on")


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
    st.rerun.reset_mock()
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
    module.st.rerun.assert_not_called()


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
    # The quality preset above was drawn from the old numbers, so the page is drawn again after each save.
    assert module.st.rerun.call_count == 2


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
    "saved",
    [
        {"evidence_k": 3},
        {"answer_length": "50 to 100 words"},
        {"answer_template": "Study notes"},
        {"grounding": "Strictly grounded"},
        {"pre_prompt": "List the terms in: {question}"},
        {"rate_profile": "Low limits"},
    ],
)
def test_the_defaults_button_forgets_every_saved_sidebar_setting(saved, pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_settings_reset(module, saved=saved, clicked={"answer_settings_reset"})

    module.st.button.assert_called_once_with(module.SETTINGS_RESET_LABEL, key="answer_settings_reset")
    patched["forget"].assert_called_once_with(
        [
            "quality_preset",
            "evidence_k",
            "answer_max_sources",
            "answer_length",
            "answer_template",
            "grounding",
            "pre_prompt",
            "rate_profile",
        ]
    )
    # The browser keeps a control's old value, so the controls get new keys and start again from the defaults.
    assert module.st.session_state.answer_settings_round == 1
    module.st.rerun.assert_called_once_with()


def test_controls_get_new_keys_after_going_back_to_the_defaults(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.answer_settings_round = 1

    draw_quality_preset(module)
    assert module.st.selectbox.call_args.kwargs["key"] == "setting_quality_preset_1_Balanced"
    draw_source_settings(module)
    draw_answer_length(module)

    keys = [call.kwargs["key"] for call in module.st.number_input.call_args_list]
    assert keys == ["setting_evidence_k_1", "setting_answer_max_sources_1"]
    assert module.st.selectbox.call_args.kwargs["key"] == "setting_answer_length_1"
    draw_rate_profile(module)
    assert module.st.selectbox.call_args.kwargs["key"] == "setting_rate_profile_1"
    draw_answer_template(module)
    assert module.st.selectbox.call_args.kwargs["key"] == "setting_answer_template_1"
    draw_grounding(module)
    assert module.st.selectbox.call_args.kwargs["key"] == "setting_grounding_1"
    draw_pre_prompt(module)
    assert module.st.text_area.call_args.kwargs["key"] == "setting_pre_prompt_1"


def test_the_defaults_button_changes_nothing_until_clicked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_settings_reset(module, saved={"answer_max_sources": 2})

    assert "answer_settings_round" not in module.st.session_state
    module.st.button.assert_called_once()
    patched["forget"].assert_not_called()
    module.st.rerun.assert_not_called()


# Sidebar control for the quality preset.

PRESET_NAMES = ["Fast", "Balanced", "High quality", "Debug"]


def draw_quality_preset(module, *, settings=None, picked=None, saved=None):
    """Draw the quality preset control, with the user picking a preset or leaving it as it is."""
    st = module.st
    st.caption.reset_mock()
    st.rerun.reset_mock()
    st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: picked or options[index])
    with (
        mock.patch.object(module.saved_settings, "save_value") as save_value,
        mock.patch.object(module.saved_settings, "forget") as forget,
        mock.patch.object(module.saved_settings, "load", return_value=saved or {}),
    ):
        module.render_quality_preset(settings or _fake_settings())
        return {"save_value": save_value, "forget": forget}


def test_quality_preset_offers_the_presets_starting_at_the_one_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_quality_preset(module, settings=_fake_settings(evidence_k=20))

    call = module.st.selectbox.call_args
    assert call.args == ("Quality preset", PRESET_NAMES)
    assert call.kwargs["index"] == 2
    assert call.kwargs["help"] == module.settings_groups.get_setting("quality_preset").help
    assert call.kwargs["key"] == "setting_quality_preset_0_High quality"
    module.st.caption.assert_called_once_with(quality_presets.get_preset("High quality").description)
    patched["save_value"].assert_not_called()
    patched["forget"].assert_not_called()
    module.st.rerun.assert_not_called()


def test_picking_a_preset_saves_it_forgets_the_numbers_it_covers_and_redraws_the_controls(
    pdf_research_assistant_module,
):
    module = pdf_research_assistant_module

    patched = draw_quality_preset(module, picked="Fast")

    patched["save_value"].assert_called_once_with("quality_preset", "Fast")
    patched["forget"].assert_called_once_with(quality_presets.KEYS_WITH_CONTROLS)
    # The number boxes and answer length were drawn with the old values, so they get new keys.
    assert module.st.session_state.answer_settings_round == 1
    module.st.rerun.assert_called_once_with()


def test_the_preset_control_gets_a_new_key_when_the_preset_in_use_changes(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_quality_preset(module)
    balanced = module.st.selectbox.call_args.kwargs["key"]
    draw_quality_preset(module, settings=_fake_settings(evidence_k=7))
    custom = module.st.selectbox.call_args.kwargs["key"]

    # The browser keeps the old pick under the old key, which would otherwise undo the changed number.
    assert balanced == "setting_quality_preset_0_Balanced"
    assert custom == "setting_quality_preset_0_Custom"


def test_the_preset_control_key_follows_the_defaults_round(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.answer_settings_round = 2

    assert module.quality_preset_widget_key("Fast") == "setting_quality_preset_2_Fast"


def test_numbers_that_match_no_preset_show_as_custom_until_a_preset_is_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_quality_preset(module, settings=_fake_settings(evidence_k=7))

    call = module.st.selectbox.call_args
    assert call.args[1] == PRESET_NAMES + [quality_presets.NO_PRESET]
    assert call.kwargs["index"] == 4
    module.st.caption.assert_called_once_with(module.NO_PRESET_NOTE)
    patched["save_value"].assert_not_called()


def test_custom_says_which_saved_preset_it_started_from(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_quality_preset(module, settings=_fake_settings(evidence_k=7), saved={"quality_preset": "Fast"})

    module.st.caption.assert_called_once_with(module.CHANGED_PRESET_NOTE.format(name="Fast"))
    assert "Fast" in module.CHANGED_PRESET_NOTE.format(name="Fast")


def test_page_draws_the_quality_preset_from_the_settings_it_built_above_the_numbers_it_covers():
    with imported_pdf_research_assistant() as module:
        call = next(c for c in module.st.selectbox.call_args_list if c.args[0] == "Quality preset")

        assert call.args[1][call.kwargs["index"]] == quality_presets.DEFAULT_PRESET
        assert module.st.selectbox.call_args_list.index(call) == 0


# Sidebar control for how long the answer is asked to be.

SHORT = "50 to 100 words"
STANDARD = "about 200 words, but can be longer"
LONG = "about 400 words, but can be longer"


def draw_answer_length(module, *, settings=None, picked=None):
    """Draw the answer length control, with the user picking a phrase or leaving it as it is."""
    st = module.st
    st.rerun.reset_mock()
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
    module.st.rerun.assert_not_called()


def test_picking_another_answer_length_saves_it_for_the_next_question(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_answer_length(module, settings=_fake_settings(answer_length=STANDARD), picked=LONG)

    save_value.assert_called_once_with("answer_length", LONG)
    # The quality preset above was drawn from the old length, so the page is drawn again.
    module.st.rerun.assert_called_once_with()


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


# Sidebar control for the answer template.

TEMPLATE_NAMES = ["Direct answer", "Study notes", "Compare sources", "Contradictions only"]


def draw_answer_template(module, *, settings=None, picked=None):
    """Draw the answer template control, with the user picking a template or leaving it as it is."""
    st = module.st
    st.caption.reset_mock()
    st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: picked or options[index])
    with mock.patch.object(module.saved_settings, "save_value") as save_value:
        module.render_answer_template(settings or _fake_settings(answer_template="Direct answer"))
        return save_value


def test_answer_template_offers_the_templates_starting_at_the_one_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_answer_template(module, settings=_fake_settings(answer_template="Compare sources"))

    call = module.st.selectbox.call_args
    assert call.args == ("Answer template", TEMPLATE_NAMES)
    assert call.kwargs["index"] == 2
    assert call.kwargs["help"] == module.settings_groups.get_setting("answer_template").help
    assert call.kwargs["key"] == "setting_answer_template_0"
    save_value.assert_not_called()


def test_answer_template_explains_the_template_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_answer_template(module, settings=_fake_settings(answer_template="Study notes"))

    module.st.caption.assert_called_once_with(answer_templates.get_template("Study notes").description)


def test_picking_another_answer_template_saves_it_and_explains_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_answer_template(
        module, settings=_fake_settings(answer_template="Direct answer"), picked="Contradictions only"
    )

    save_value.assert_called_once_with("answer_template", "Contradictions only")
    module.st.caption.assert_called_once_with(answer_templates.get_template("Contradictions only").description)


def test_an_answer_prompt_that_matches_no_template_is_kept_until_a_template_is_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_answer_template(module, settings=_fake_settings())

    call = module.st.selectbox.call_args
    assert call.args[1] == TEMPLATE_NAMES + [answer_templates.NO_TEMPLATE]
    assert call.kwargs["index"] == 4
    module.st.caption.assert_called_once_with(module.NO_TEMPLATE_NOTE)
    save_value.assert_not_called()


def test_page_draws_the_answer_template_from_the_settings_it_built():
    with imported_pdf_research_assistant() as module:
        call = next(c for c in module.st.selectbox.call_args_list if c.args[0] == "Answer template")

        assert call.args[1][call.kwargs["index"]] == answer_templates.DEFAULT_TEMPLATE


# Sidebar control for the grounding mode.

MODE_NAMES = ["Strictly grounded", "Balanced", "Exploratory"]


def draw_grounding(module, *, settings=None, picked=None):
    """Draw the grounding control, with the user picking a mode or leaving it as it is."""
    st = module.st
    st.caption.reset_mock()
    st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: picked or options[index])
    with mock.patch.object(module.saved_settings, "save_value") as save_value:
        module.render_grounding(settings or _fake_settings(grounding="Balanced"))
        return save_value


def test_grounding_offers_the_modes_starting_at_the_one_in_use(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_grounding(module, settings=_fake_settings(grounding="Exploratory"))

    call = module.st.selectbox.call_args
    assert call.args == ("Grounding", MODE_NAMES)
    assert call.kwargs["index"] == 2
    assert call.kwargs["help"] == module.settings_groups.get_setting("grounding").help
    assert call.kwargs["key"] == "setting_grounding_0"
    module.st.caption.assert_called_once_with(grounding_modes.get_mode("Exploratory").description)
    save_value.assert_not_called()


def test_picking_another_grounding_mode_saves_it_and_explains_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_grounding(module, settings=_fake_settings(grounding="Balanced"), picked="Strictly grounded")

    save_value.assert_called_once_with("grounding", "Strictly grounded")
    module.st.caption.assert_called_once_with(grounding_modes.get_mode("Strictly grounded").description)


def test_settings_that_match_no_grounding_mode_are_kept_until_a_mode_is_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_grounding(module, settings=_fake_settings())

    call = module.st.selectbox.call_args
    assert call.args[1] == MODE_NAMES + [grounding_modes.NO_MODE]
    assert call.kwargs["index"] == 3
    module.st.caption.assert_called_once_with(module.NO_MODE_NOTE)
    save_value.assert_not_called()


def test_page_draws_the_grounding_mode_from_the_settings_it_built():
    with imported_pdf_research_assistant() as module:
        call = next(c for c in module.st.selectbox.call_args_list if c.args[0] == "Grounding")

        assert call.args[1][call.kwargs["index"]] == grounding_modes.DEFAULT_MODE


# Sidebar control for the before-answer prompt.


def draw_pre_prompt(module, *, settings=None, typed=None, refused=None):
    """Draw the before-answer prompt box, with the user typing new text or leaving it as it is."""
    st = module.st
    st.caption.reset_mock()
    st.error.reset_mock()
    st.text_area = mock.Mock(side_effect=lambda _label, value="", **_kwargs: value if typed is None else typed)
    side_effect = ValueError(refused) if refused else None
    with mock.patch.object(module.saved_settings, "save_value", side_effect=side_effect) as save_value:
        module.render_pre_prompt(settings or _fake_settings())
        return save_value


def test_pre_prompt_box_starts_empty_when_paperqa_has_none(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_pre_prompt(module)

    call = module.st.text_area.call_args
    setting = module.settings_groups.get_setting("pre_prompt")
    assert call.args == (setting.label,)
    assert call.kwargs["value"] == ""
    assert call.kwargs["max_chars"] == module.prompt_hooks.PRE_PROMPT_CHECK.limit
    assert call.kwargs["placeholder"] == module.prompt_hooks.PRE_PROMPT_EXAMPLE
    assert call.kwargs["help"] == setting.help
    assert call.kwargs["key"] == "setting_pre_prompt_0"
    save_value.assert_not_called()
    assert module.st.caption.call_args_list == [
        mock.call(module.PRE_PROMPT_OFF_NOTE),
        mock.call(module.prompt_hooks.POST_PROMPT_NOTE),
    ]


def test_pre_prompt_box_starts_at_the_prompt_in_use_and_says_it_costs_a_call(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_pre_prompt(module, settings=_fake_settings(pre_prompt="Terms in {question}"))

    assert module.st.text_area.call_args.kwargs["value"] == "Terms in {question}"
    save_value.assert_not_called()
    assert module.st.caption.call_args_list == [
        mock.call(module.PRE_PROMPT_ON_NOTE),
        mock.call(module.prompt_hooks.POST_PROMPT_NOTE),
    ]


def test_typing_a_pre_prompt_saves_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_pre_prompt(module, typed="Terms in {question}")

    save_value.assert_called_once_with("pre_prompt", "Terms in {question}")
    module.st.error.assert_not_called()
    assert module.st.caption.call_args_list[0] == mock.call(module.PRE_PROMPT_ON_NOTE)


def test_emptying_the_pre_prompt_box_saves_it_as_off(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_pre_prompt(module, settings=_fake_settings(pre_prompt="Terms in {question}"), typed="  ")

    save_value.assert_called_once_with("pre_prompt", "  ")
    assert module.st.caption.call_args_list[0] == mock.call(module.PRE_PROMPT_OFF_NOTE)


def test_space_around_the_prompt_in_use_is_not_a_change(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    save_value = draw_pre_prompt(module, settings=_fake_settings(pre_prompt="Terms"), typed=" Terms\n")

    save_value.assert_not_called()


def test_a_pre_prompt_that_cannot_be_used_is_not_saved_and_says_why(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_pre_prompt(module, typed="Summarise {answer}", refused="Only {question} can be used.")

    module.st.error.assert_called_once_with(f"{module.PRE_PROMPT_REFUSED_NOTE} Only {{question}} can be used.")
    # The prompt in use is still described, since nothing changed.
    assert module.st.caption.call_args_list[0] == mock.call(module.PRE_PROMPT_OFF_NOTE)


def test_page_draws_the_pre_prompt_box_from_the_settings_it_built():
    with imported_pdf_research_assistant() as module:
        call = module.st.text_area.call_args

        assert call.args == ("Before-answer prompt",)
        assert call.kwargs["value"] == ""


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
    # A bare phrase is an answer length; a dict picks by control label.
    picked = picked if isinstance(picked, dict) else ({"Answer length": picked} if picked else {})
    st.selectbox = mock.Mock(side_effect=lambda label, options, index=0, **_kwargs: picked.get(label, options[index]))
    with mock.patch.object(module.saved_settings, "save_value") as save_value:
        chosen = module.render_one_off_settings(
            settings or _fake_settings(answer_template="Direct answer", grounding="Balanced")
        )
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

    chosen = draw_one_off_settings(
        module,
        settings=_fake_settings(evidence_k=12, answer_max_sources=4, answer_template="Study notes", grounding="Balanced"),
    )

    assert chosen == {}
    gathered = _number_box(module, "Passages gathered")
    sources = _number_box(module, "Most sources in the answer")
    assert (gathered.kwargs["value"], gathered.kwargs["max_value"]) == (12, 30)
    assert (sources.kwargs["value"], sources.kwargs["max_value"]) == (4, 12)
    # The usual value is part of each key, so a change to the usual settings starts the box again from it.
    assert gathered.kwargs["key"] == "one_off_evidence_k_0_12"
    assert sources.kwargs["key"] == "one_off_answer_max_sources_0_4"
    length = _control(module.st.selectbox, "Answer length")
    assert length.args == ("Answer length", [SHORT, STANDARD, LONG])
    assert length.kwargs["index"] == 1
    assert length.kwargs["key"] == f"one_off_answer_length_0_{STANDARD}"
    assert length.kwargs["format_func"](SHORT) == "Short (50 to 100 words)"
    template = _control(module.st.selectbox, "Answer template")
    assert template.args == ("Answer template", TEMPLATE_NAMES)
    assert template.kwargs["index"] == 1
    assert template.kwargs["key"] == "one_off_answer_template_0_Study notes"
    grounding = _control(module.st.selectbox, "Grounding")
    assert grounding.args == ("Grounding", MODE_NAMES)
    assert grounding.kwargs["index"] == 1
    assert grounding.kwargs["key"] == "one_off_grounding_0_Balanced"


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

    length = _control(module.st.selectbox, "Answer length")
    assert length.args[1] == [SHORT, STANDARD, LONG, "one paragraph"]
    assert length.kwargs["index"] == 3
    assert chosen == {}


def test_one_off_answer_template_can_be_picked_for_the_next_question_only(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, picked={"Answer template": "Compare sources"})

    assert chosen == {"answer_template": "Compare sources"}


def test_one_off_answer_template_offers_a_usual_prompt_that_matches_no_template(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, settings=_fake_settings())

    template = _control(module.st.selectbox, "Answer template")
    assert template.args[1] == TEMPLATE_NAMES + [answer_templates.NO_TEMPLATE]
    assert template.kwargs["index"] == 4
    assert chosen == {}


def test_one_off_grounding_can_be_picked_for_the_next_question_only(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, picked={"Grounding": "Strictly grounded", "Answer template": "Study notes"})

    assert chosen == {"answer_template": "Study notes", "grounding": "Strictly grounded"}


def test_one_off_grounding_offers_usual_settings_that_match_no_mode(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    chosen = draw_one_off_settings(module, settings=_fake_settings(answer_template="Direct answer"))

    grounding = _control(module.st.selectbox, "Grounding")
    assert grounding.args[1] == MODE_NAMES + [grounding_modes.NO_MODE]
    assert grounding.kwargs["index"] == 3
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


def test_local_llm_guide_suggests_fast_and_low_limits_for_small_models(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    guide = module.LOCAL_LLM_GUIDE

    assert "**Fast**" in guide
    assert "**Low limits**" in guide
    assert "no papers" in guide
    # The names in the tip must match the options the sidebar offers.
    assert "Fast" in [preset.name for preset in module.quality_presets.PRESETS]
    assert "Low limits" in [profile.name for profile in module.rate_profiles.PROFILES]


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


# Rebuilding the index from the app


def _rebuild_run(tmp_path, done=0, pdf_count=4):
    return {
        "process": mock.Mock(),
        "index_dir": tmp_path / "pqa_index_abc",
        "log_path": tmp_path / "rebuild.log",
        "pdf_count": pdf_count,
        "done": done,
        "started": datetime(2026, 1, 2, 9, 5),
    }


def _button_keys(module):
    return [call.kwargs.get("key") for call in module.st.button.call_args_list]


@pytest.mark.parametrize(
    "indexed_count, manifest_count, failed_files, expected",
    [
        (0, 0, [], True),
        (0, 3, [], True),
        (2, 3, [], True),
        (3, 3, ["a.pdf"], True),
        (3, 3, [], False),
        (5, 0, [], False),
    ],
)
def test_index_needs_rebuild_when_it_is_empty_incomplete_or_has_failed_pdfs(
    pdf_research_assistant_module, indexed_count, manifest_count, failed_files, expected
):
    module = pdf_research_assistant_module

    assert module.index_needs_rebuild(indexed_count, manifest_count, failed_files) is expected


def test_a_rebuild_is_offered_on_the_page_when_the_index_is_empty():
    with imported_pdf_research_assistant(indexed_count=0) as module:
        offer = next(call for call in module.st.button.call_args_list if call.kwargs.get("key") == "rebuild_offer_main")

        assert offer.args[0] == "Rebuild the index"
        assert offer.kwargs["help"] == module.REBUILD_HELP
        assert "rebuild_pending" not in module.st.session_state


def test_a_rebuild_is_offered_on_the_page_when_pdfs_failed_to_index():
    with imported_pdf_research_assistant(failed_files=["bad.pdf"]) as module:
        assert "rebuild_offer_main" in _button_keys(module)


def test_an_up_to_date_index_offers_a_rebuild_in_the_sidebar_only():
    with imported_pdf_research_assistant() as module:
        keys = _button_keys(module)

    assert "rebuild_offer_main" not in keys
    assert "rebuild_offer_sidebar" in keys


def test_no_rebuild_is_offered_while_an_exhaustive_search_waits_to_be_confirmed():
    state = _empty_state() | {"exhaustive_pending": {"question": "q", "plan": READY_PLAN}}

    with imported_pdf_research_assistant(app_state=state, indexed_count=0, button_values=[False, False, False]) as module:
        keys = _button_keys(module)

    assert "rebuild_offer_main" not in keys
    assert "rebuild_offer_sidebar" not in keys


@pytest.mark.parametrize("key", ["rebuild_offer_main", "rebuild_offer_sidebar"])
def test_clicking_a_rebuild_button_asks_for_confirmation_without_starting_anything(key):
    start = mock.Mock()

    with imported_pdf_research_assistant(
        indexed_count=0, clicked_keys={key}, rebuild_patches={"start": start}
    ) as module:
        assert module.st.session_state.rebuild_pending is True
        module.st.rerun.assert_called_with()

    start.assert_not_called()


def test_a_pending_rebuild_shows_what_it_involves_and_switches_questions_off():
    start = mock.Mock()
    state = _empty_state() | {"rebuild_pending": True}

    with imported_pdf_research_assistant(
        app_state=state,
        allowed_paths={"a.pdf", "b.pdf"},
        indexed_count=0,
        chat_input_value="q",
        rebuild_patches={"start": start},
    ) as module:
        text = _markdown_text(module)
        keys = _button_keys(module)
        confirm = next(call for call in module.st.button.call_args_list if call.kwargs.get("key") == "rebuild_confirm")

        assert "- PDFs in scope: 2" in text
        assert "`text-embedding-3-small`, which costs money" in text
        assert confirm.args[0] == "Start paid rebuild"
        assert "rebuild_cancel" in keys
        assert "rebuild_offer_main" not in keys
        assert "rebuild_offer_sidebar" not in keys
        assert module.st.chat_input.call_args.kwargs["disabled"] is True
        # A question that still arrives is not run.
        assert module.st.session_state.history == []
        module.st.warning.assert_any_call(module.REBUILD_QUESTION_NOTE)

    start.assert_not_called()


def test_confirming_starts_the_rebuild_for_the_active_index_and_the_manifest_pdfs():
    start = mock.Mock()
    state = _empty_state() | {"rebuild_pending": True}

    with imported_pdf_research_assistant(
        app_state=state,
        allowed_paths={"a.pdf", "b.pdf"},
        indexed_count=0,
        clicked_keys={"rebuild_confirm"},
        rebuild_patches={"start": start},
    ) as module:
        assert module.st.session_state.rebuild_pending is False
        module.st.rerun.assert_called_with()

    start.assert_called_once_with(Path(r"C:\index\active-shard"), 2)


def test_confirming_without_a_manifest_counts_the_pdfs_in_the_pdf_folder():
    start = mock.Mock()
    count_pdfs = mock.Mock(return_value=7)
    state = _empty_state() | {"rebuild_pending": True}

    with imported_pdf_research_assistant(
        app_state=state,
        indexed_count=0,
        clicked_keys={"rebuild_confirm"},
        rebuild_patches={"start": start, "count_pdfs": count_pdfs},
    ):
        pass

    count_pdfs.assert_called_once_with(Path(r"D:\papers"), True)
    start.assert_called_once_with(Path(r"C:\index\active-shard"), 7)


def test_cancelling_a_pending_rebuild_starts_nothing():
    start = mock.Mock()
    state = _empty_state() | {"rebuild_pending": True}

    with imported_pdf_research_assistant(
        app_state=state, indexed_count=0, clicked_keys={"rebuild_cancel"}, rebuild_patches={"start": start}
    ) as module:
        assert module.st.session_state.rebuild_pending is False
        module.st.rerun.assert_called_with()

    start.assert_not_called()


def test_describe_rebuild_plan_says_how_many_pdfs_are_already_indexed(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    lines = module.describe_rebuild_plan(_fake_settings(), Path("index/pqa_index_abc"), 10, 4)

    assert lines[0] == "PDFs in scope: 10. 4 are already in this index and are not indexed again"
    assert lines[1].startswith("Index: `pqa_index_abc`")
    assert lines[-1] == module.REBUILD_TIME_NOTE
    assert not any("enrichment" in line for line in lines)


def test_describe_rebuild_plan_names_the_index_by_its_friendly_name(pdf_research_assistant_module):
    lines = pdf_research_assistant_module.describe_rebuild_plan(
        _fake_settings(), Path("index/pqa_index_abc"), 10, 0, "My library"
    )

    assert lines[1].startswith("Index: **My library** (`pqa_index_abc`), built with")


def test_rebuild_confirm_step_names_the_active_index_by_its_friendly_name():
    with imported_pdf_research_assistant(
        indexed_count=0,
        app_state=_empty_state() | {"rebuild_pending": True},
        active_index_dir=Path(r"C:\index\active-shard"),
        index_names={"active-shard": "My library"},
    ) as module:
        plans = [call.args[0] for call in module.st.markdown.call_args_list if call.args[0].startswith("- PDFs")]
        assert len(plans) == 1
        assert "- Index: **My library** (`active-shard`), built with" in plans[0]


def test_describe_rebuild_plan_leaves_out_the_indexed_count_for_an_empty_index(pdf_research_assistant_module):
    lines = pdf_research_assistant_module.describe_rebuild_plan(_fake_settings(), Path("index/pqa_index_abc"), 10, 0)

    assert lines[0] == "PDFs in scope: 10"


def test_describe_rebuild_plan_warns_that_enrichment_pays_a_model_for_every_image(pdf_research_assistant_module):
    lines = pdf_research_assistant_module.describe_rebuild_plan(
        _fake_settings(multimodal=True), Path("index/pqa_index_abc"), 10, 0
    )

    assert any("`gpt-4o` is also paid to describe every image and table" in line for line in lines)


def test_a_running_rebuild_shows_progress_and_switches_questions_and_index_settings_off(tmp_path):
    run = _rebuild_run(tmp_path, pdf_count=4)
    patches = {
        "get_run": mock.Mock(return_value=run),
        "describe_progress": mock.Mock(return_value="1 of 4 PDFs done"),
    }

    with imported_pdf_research_assistant(
        indexed_count=1, failed_files=[], chat_input_value="q", rebuild_patches=patches
    ) as module:
        keys = _button_keys(module)

        module.st.progress.assert_called_once_with(0.25, text="1 of 4 PDFs done")
        module.st.info.assert_any_call(module.REBUILD_RUNNING_NOTE)
        module.st.caption.assert_any_call(f"Details are written to `{run['log_path']}`.")
        module.st.caption.assert_any_call(module.REBUILD_LOCKED_NOTE)
        module.st.form_submit_button.assert_any_call(module.INDEX_SETTINGS_SAVE_LABEL, disabled=True)
        module.bootstrap.get_indexed_doc_count.assert_any_call(index_dir=run["index_dir"])
        # Only the stop button; no button starts or offers a rebuild.
        assert [key for key in keys if str(key).startswith("rebuild_")] == ["rebuild_stop"]
        assert module.st.chat_input.call_args.kwargs["disabled"] is True
        assert module.st.session_state.history == []
        module.st.rerun.assert_not_called()


def test_a_running_rebuild_offers_a_stop_button(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    run = _rebuild_run(tmp_path)
    module.st.button.reset_mock()
    module.st.button.return_value = False

    with mock.patch.object(module.rebuild_client, "get_run", return_value=run):
        module.render_rebuild_run()

    stop = next(call for call in module.st.button.call_args_list if call.kwargs.get("key") == "rebuild_stop")
    assert stop.args[0] == module.REBUILD_STOP_LABEL
    assert stop.kwargs["help"] == module.REBUILD_STOP_HELP
    assert stop.kwargs.get("disabled") is not True
    module.st.caption.assert_any_call(module.REBUILD_STOP_NOTE)


def test_clicking_stop_asks_the_rebuild_to_stop_and_draws_the_progress_again(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    run = _rebuild_run(tmp_path)
    module.st.button.reset_mock()
    module.st.button.side_effect = lambda *args, **kwargs: kwargs.get("key") == "rebuild_stop"
    request_stop = mock.Mock()

    with mock.patch.object(module.rebuild_client, "get_run", return_value=run):
        with mock.patch.object(module.rebuild_client, "request_stop", request_stop):
            module.render_rebuild_run()

    request_stop.assert_called_once_with(run)
    module.st.button.side_effect = None


def test_a_stopping_rebuild_says_so_and_disables_the_stop_button(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    run = _rebuild_run(tmp_path) | {"stopping": True}
    module.st.button.reset_mock()
    module.st.button.return_value = False
    module.st.warning.reset_mock()

    with mock.patch.object(module.rebuild_client, "get_run", return_value=run):
        module.render_rebuild_run()

    module.st.warning.assert_any_call(module.REBUILD_STOPPING_NOTE)
    stop = next(call for call in module.st.button.call_args_list if call.kwargs.get("key") == "rebuild_stop")
    assert stop.kwargs["disabled"] is True


def test_a_stopped_rebuild_says_the_index_can_still_be_used_and_how_many_pdfs_it_holds(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.success.reset_mock()
    module.st.error.reset_mock()

    module.render_rebuild_result(
        {"ok": False, "stopped": True, "log_path": "x", "lines": [], "index": {"usable": True, "indexed": 7}}
    )

    module.st.success.assert_called_once_with(module.REBUILD_STOPPED_USABLE_NOTICE.format(count=7))
    module.st.error.assert_not_called()


def test_a_rebuild_stopped_before_any_pdf_was_saved_says_nothing_is_indexed_yet(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.success.reset_mock()
    module.st.error.reset_mock()

    module.render_rebuild_result(
        {"ok": False, "stopped": True, "log_path": "x", "lines": [], "index": {"usable": True, "indexed": 0}}
    )

    module.st.success.assert_called_once_with(module.REBUILD_STOPPED_EMPTY_NOTICE)
    module.st.error.assert_not_called()


def test_a_stopped_rebuild_whose_index_cannot_be_read_says_so(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.success.reset_mock()
    module.st.error.reset_mock()

    module.render_rebuild_result(
        {"ok": False, "stopped": True, "log_path": "index/rebuild.log", "lines": [], "index": {"usable": False, "indexed": 0}}
    )

    module.st.error.assert_called_once_with(f"{module.REBUILD_STOPPED_UNUSABLE_NOTICE} See `index/rebuild.log`.")
    module.st.success.assert_not_called()


def test_a_rebuild_that_failed_says_whether_the_index_can_still_be_used(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.error.reset_mock()
    module.st.info.reset_mock()

    module.render_rebuild_result(
        {"ok": False, "log_path": "x", "lines": [], "index": {"usable": True, "indexed": 3}}
    )

    module.st.info.assert_called_once_with(module.REBUILD_INDEX_STATE_USABLE.format(count=3))


def test_a_rebuild_that_failed_and_left_an_unreadable_index_says_so(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.info.reset_mock()

    module.render_rebuild_result(
        {"ok": False, "log_path": "x", "lines": [], "index": {"usable": False, "indexed": 0}}
    )

    module.st.error.assert_any_call(module.REBUILD_INDEX_STATE_UNUSABLE)


def test_rebuild_progress_counts_failed_pdfs_as_done(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    run = _rebuild_run(tmp_path, pdf_count=4)
    module.st.progress.reset_mock()
    module.bootstrap.get_failed_files.return_value = ["bad.pdf"]

    with mock.patch.object(module.rebuild_client, "get_run", return_value=run):
        module.render_rebuild_run()

    assert run["done"] == 2
    assert module.st.progress.call_args.args[0] == 0.5


def test_a_finished_rebuild_draws_the_whole_page_again(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    module.st.progress.reset_mock()
    module.st.rerun.reset_mock()

    with mock.patch.multiple(
        module.rebuild_client,
        get_run=mock.Mock(return_value=_rebuild_run(tmp_path)),
        collect_result=mock.Mock(return_value={"ok": True}),
    ):
        module.render_rebuild_run()

    module.st.rerun.assert_called_once_with()
    module.st.progress.assert_not_called()


def test_a_rebuild_that_another_session_saw_finish_draws_the_whole_page_again(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.progress.reset_mock()
    module.st.rerun.reset_mock()

    module.render_rebuild_run()

    module.st.rerun.assert_called_once_with()
    module.st.progress.assert_not_called()


def test_a_finished_rebuild_is_reported_with_the_end_of_its_log():
    result = {"ok": True, "log_path": "index/rebuild.log", "lines": ["Indexed after run: 3", "Failed PDFs: 0"]}

    with imported_pdf_research_assistant(rebuild_patches={"take_result": mock.Mock(return_value=result)}) as module:
        module.st.success.assert_any_call(module.REBUILD_FINISHED_NOTICE)
        module.st.expander.assert_any_call(module.REBUILD_LOG_TITLE)
        module.st.code.assert_any_call("Indexed after run: 3\nFailed PDFs: 0")
        module.st.error.assert_not_called()


def test_a_rebuild_that_stopped_unexpectedly_is_reported_as_an_error(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.expander.reset_mock()

    module.render_rebuild_result({"ok": False, "log_path": "index/rebuild.log", "lines": []})

    module.st.error.assert_called_once_with(f"{module.REBUILD_FAILED_NOTICE} See `index/rebuild.log`.")
    module.st.expander.assert_not_called()


def test_no_rebuild_result_shows_nothing(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.success.reset_mock()

    module.render_rebuild_result(None)

    module.st.success.assert_not_called()
    module.st.error.assert_not_called()


def test_the_sidebar_offers_no_rebuild_while_the_page_is_busy(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_index_settings(module, busy=True)

    assert "rebuild_offer_sidebar" not in _button_keys(module)


def test_the_sidebar_rebuild_button_asks_for_confirmation(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_index_settings(module, clicked={"rebuild_offer_sidebar"})

    assert module.st.session_state.rebuild_pending is True
    module.st.rerun.assert_called_once_with()


def test_index_settings_cannot_be_saved_while_a_rebuild_runs(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    with mock.patch.object(module.rebuild_client, "get_run", return_value=_rebuild_run(tmp_path)):
        patched = draw_index_settings(module, chosen={"Chunk size (characters)": 3000}, submitted=True)

    patched["save_value"].assert_not_called()
    module.st.caption.assert_any_call(module.REBUILD_LOCKED_NOTE)
    assert _button_keys(module) == []


# Named research presets in the sidebar.

OLLAMA_MODEL = {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3"}
QUICK = {"answer": {"quality_preset": "Fast"}}
DEEP = {"answer": {"quality_preset": "High quality"}, "index": {"chunk_chars": 3000}}
FILE_BOXES = ("research_preset_export_model", "research_preset_import_replace")


def draw_research_presets(
    module,
    *,
    presets=None,
    saved=None,
    model=None,
    clicked=(),
    submitted=False,
    typed="",
    checks=(True, True),
    picked=None,
    new_name=None,
    rebuilding=False,
    media_available=True,
    save_result=("Typed", False),
    rename_result=None,
    rename_error=None,
    exported=None,
    ticked=None,
    uploaded=None,
    import_result=([], [], []),
    import_error=None,
):
    """Draw the research presets panel, with the user clicking buttons by key and typing into the boxes.

    The save form's two boxes are ticked in turn from checks; any other box by its key, from ticked.
    """
    st = module.st
    for name in ("rerun", "success", "error", "caption", "markdown", "warning", "subheader", "download_button"):
        setattr(st, name, mock.Mock())
    st.button = mock.Mock(side_effect=lambda _label, key=None, **_kwargs: key in clicked)
    st.form_submit_button = mock.Mock(return_value=submitted)
    st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: picked or options[index])
    ticks = iter(checks)
    st.checkbox = mock.Mock(
        side_effect=lambda *_args, key=None, **_kwargs: (ticked or {}).get(key, False) if key in FILE_BOXES else next(ticks)
    )
    st.multiselect = mock.Mock(
        side_effect=lambda _label, options, default=None, **_kwargs: list(default) if exported is None else exported
    )
    st.file_uploader = mock.Mock(
        return_value=None if uploaded is None else SimpleNamespace(getvalue=lambda: uploaded)
    )

    def type_into(label, value="", **_kwargs):
        if label == module.RESEARCH_PRESET_NAME_LABEL:
            return typed
        return value if new_name is None else new_name

    st.text_input = mock.Mock(side_effect=type_into)
    rename = mock.Mock(return_value=rename_result, side_effect=rename_error)
    with (
        mock.patch.object(module.research_presets, "load", return_value=dict(presets or {})),
        mock.patch.object(module.saved_settings, "load", return_value=dict(saved or {})),
        mock.patch.object(module.local_llm, "load_config", return_value=model),
        mock.patch.object(module.research_presets, "pick_preset") as pick,
        mock.patch.object(module.research_presets, "save_preset", return_value=save_result) as save,
        mock.patch.object(module.research_presets, "rename_preset", rename),
        mock.patch.object(module.research_presets, "delete_preset") as delete,
        mock.patch.object(
            module.research_presets, "import_presets", return_value=import_result, side_effect=import_error
        ) as import_presets,
        mock.patch.object(module.rebuild_client, "get_run", return_value={"pid": 1} if rebuilding else None),
        mock.patch.object(module, "media_reading_available", return_value=media_available),
    ):
        module.render_research_presets(module.st.sidebar)
        return {"pick": pick, "save": save, "rename": rename, "delete": delete, "import": import_presets}


def _markdown_lines(module):
    return [call.args[0] for call in module.st.markdown.call_args_list]


def test_research_presets_say_when_none_is_saved_and_offer_only_the_save_form(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module)

    module.st.subheader.assert_called_once_with(module.RESEARCH_PRESETS_TITLE)
    assert _markdown_lines(module) == [
        module.NO_RESEARCH_PRESETS_NOTE,
        module.RESEARCH_PRESET_SAVE_TITLE,
        module.RESEARCH_PRESET_IMPORT_TITLE,
    ]
    module.st.expander.assert_any_call(module.RESEARCH_PRESETS_PANEL)
    assert [call.args[0] for call in module.st.caption.call_args_list] == [
        module.RESEARCH_PRESETS_NOTE,
        module.RESEARCH_PRESET_FILES_NOTE,
    ]
    module.st.selectbox.assert_not_called()
    module.st.form.assert_any_call("research_preset_form")
    assert _button_keys(module) == []


def test_the_preset_in_use_is_named_and_the_list_starts_at_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Deep": DEEP, "Quick": QUICK}, saved={"quality_preset": "Fast"})

    assert _markdown_lines(module)[0] == "**Research preset:** Quick"
    call = module.st.selectbox.call_args
    assert call.args == (module.RESEARCH_PRESET_PICK_LABEL, ["Deep", "Quick"])
    assert call.kwargs["index"] == 1
    assert call.kwargs["key"] == "research_preset_pick_0"
    module.st.caption.assert_any_call(module.RESEARCH_PRESET_IN_USE_NOTE)
    # It is already in use, so there is nothing to use.
    assert _button_keys(module) == ["research_preset_rename", "research_preset_delete"]


def test_every_matching_preset_is_named(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"A*": QUICK, "B": QUICK}, saved={"quality_preset": "Fast"})

    assert _markdown_lines(module)[0] == r"**Research preset:** A\*, B"


def test_settings_that_match_no_preset_say_so_and_the_list_starts_at_the_first(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Deep": DEEP, "Quick": QUICK}, saved={"evidence_k": 3})

    assert _markdown_lines(module)[0] == module.NO_ACTIVE_RESEARCH_PRESET_NOTE
    assert module.st.selectbox.call_args.kwargs["index"] == 0
    assert "research_preset_use" in _button_keys(module)


def test_the_list_starts_at_the_preset_just_saved_renamed_or_used(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_selected = "Quick"
    module.st.session_state.research_preset_round = 2

    draw_research_presets(module, presets={"Deep": DEEP, "Quick": QUICK})

    call = module.st.selectbox.call_args
    assert call.kwargs["index"] == 1
    assert call.kwargs["key"] == "research_preset_pick_2"


def test_the_picked_preset_shows_what_it_holds(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Deep": DEEP}, model=OLLAMA_MODEL)

    lines = _markdown_lines(module)
    assert lines[1:-3] == module.research_presets.describe_preset(DEEP)
    assert lines[-3:] == [
        module.RESEARCH_PRESET_SAVE_TITLE,
        module.RESEARCH_PRESET_EXPORT_TITLE,
        module.RESEARCH_PRESET_IMPORT_TITLE,
    ]


def test_using_a_preset_puts_its_settings_in_place_and_redraws_every_control(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(
        module, presets={"Deep": DEEP, "Quick": QUICK}, picked="Quick", clicked={"research_preset_use"}
    )

    patched["pick"].assert_called_once_with(QUICK)
    state = module.st.session_state
    assert state.answer_settings_round == 1
    assert state.index_settings_round == 1
    assert state.research_preset_round == 1
    assert state.research_preset_selected == "Quick"
    assert state.research_preset_notice == "Now using the preset “Quick”."
    module.st.rerun.assert_called_once_with()


def test_a_preset_that_changes_the_index_settings_says_the_pdfs_must_be_indexed_again(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Deep": DEEP})

    module.st.caption.assert_any_call(module.RESEARCH_PRESET_REBUILD_NOTE)
    assert "research_preset_use" in _button_keys(module)


def test_a_preset_whose_index_settings_are_in_use_says_nothing_about_indexing(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Deep": DEEP}, saved={"chunk_chars": 3000})

    assert mock.call(module.RESEARCH_PRESET_REBUILD_NOTE) not in module.st.caption.call_args_list


def test_a_preset_that_changes_the_index_cannot_be_used_while_a_rebuild_runs(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(module, presets={"Deep": DEEP}, rebuilding=True, clicked={"research_preset_use"})

    module.st.caption.assert_any_call(module.RESEARCH_PRESET_LOCKED_NOTE)
    assert "research_preset_use" not in _button_keys(module)
    patched["pick"].assert_not_called()


def test_a_preset_that_leaves_the_index_alone_can_be_used_while_a_rebuild_runs(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(module, presets={"Quick": QUICK}, rebuilding=True, clicked={"research_preset_use"})

    patched["pick"].assert_called_once_with(QUICK)


def test_a_preset_with_images_and_tables_is_refused_without_the_package_to_read_them(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    media = {"answer": {}, "index": {"multimodal": "on_without_enrichment"}}

    patched = draw_research_presets(
        module, presets={"Media": media}, media_available=False, clicked={"research_preset_use"}
    )

    module.st.error.assert_called_once_with(module.MEDIA_PACKAGE_NOTE)
    patched["pick"].assert_not_called()
    module.st.rerun.assert_not_called()


def test_a_preset_with_images_and_tables_off_needs_no_package(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    plain = {"answer": {}, "index": {"multimodal": "off"}}

    patched = draw_research_presets(
        module, presets={"Plain": plain}, media_available=False, clicked={"research_preset_use"}
    )

    patched["pick"].assert_called_once_with(plain)


def test_renaming_a_preset_saves_the_new_name_and_selects_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(
        module, presets={"Quick": QUICK}, new_name="Fast look", rename_result="Fast look", clicked={"research_preset_rename"}
    )

    patched["rename"].assert_called_once_with("Quick", "Fast look")
    text_call = next(
        c for c in module.st.text_input.call_args_list if c.args[0] == module.RESEARCH_PRESET_RENAME_LABEL
    )
    assert text_call.kwargs["value"] == "Quick"
    assert text_call.kwargs["key"] == "research_preset_new_name_Quick_0"
    assert module.st.session_state.research_preset_selected == "Fast look"
    assert module.st.session_state.research_preset_notice == "Renamed the preset to “Fast look”."
    module.st.rerun.assert_called_once_with()


def test_renaming_a_preset_to_its_own_name_changes_nothing(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Quick": QUICK}, rename_result="Quick", clicked={"research_preset_rename"})

    module.st.rerun.assert_not_called()
    assert "research_preset_notice" not in module.st.session_state


def test_a_refused_rename_says_why(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(
        module,
        presets={"Quick": QUICK},
        new_name="Deep",
        rename_error=ValueError("There is already a preset called “Deep_1”."),
        clicked={"research_preset_rename"},
    )

    module.st.error.assert_called_once_with(r"There is already a preset called “Deep\_1”\.")
    module.st.rerun.assert_not_called()


def test_deleting_a_preset_asks_once_more_first(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(module, presets={"Quick": QUICK}, clicked={"research_preset_delete"})

    assert module.st.session_state.research_preset_to_delete == "Quick"
    patched["delete"].assert_not_called()
    module.st.rerun.assert_called_once_with()


def test_a_preset_waiting_to_be_deleted_offers_to_delete_or_keep_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_to_delete = "Quick"

    patched = draw_research_presets(module, presets={"Quick": QUICK})

    module.st.warning.assert_called_once_with("Delete the preset “Quick”? This cannot be undone.")
    assert _button_keys(module)[-2:] == ["research_preset_delete_confirm", "research_preset_delete_cancel"]
    patched["delete"].assert_not_called()


def test_confirming_the_delete_removes_the_preset(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_to_delete = "Quick"

    patched = draw_research_presets(module, presets={"Quick": QUICK}, clicked={"research_preset_delete_confirm"})

    patched["delete"].assert_called_once_with("Quick")
    state = module.st.session_state
    assert state.research_preset_to_delete is None
    assert state.research_preset_selected is None
    assert state.research_preset_notice == "Deleted the preset “Quick”."
    module.st.rerun.assert_called_once_with()


def test_keeping_the_preset_cancels_the_delete(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_to_delete = "Quick"

    patched = draw_research_presets(module, presets={"Quick": QUICK}, clicked={"research_preset_delete_cancel"})

    patched["delete"].assert_not_called()
    assert module.st.session_state.research_preset_to_delete is None
    module.st.rerun.assert_called_once_with()


def test_a_delete_waiting_for_another_preset_is_not_shown(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_to_delete = "Deep"

    draw_research_presets(module, presets={"Deep": DEEP, "Quick": QUICK}, picked="Quick")

    module.st.warning.assert_not_called()
    assert _button_keys(module)[-1] == "research_preset_delete"


def test_the_save_form_offers_a_name_and_which_parts_to_include(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module)

    name_call = module.st.text_input.call_args
    assert name_call.args == (module.RESEARCH_PRESET_NAME_LABEL,)
    assert name_call.kwargs["max_chars"] == 60
    assert name_call.kwargs["key"] == "research_preset_name_0"
    labels = [(c.args[0], c.kwargs["value"]) for c in module.st.checkbox.call_args_list]
    assert labels == [(module.RESEARCH_PRESET_INDEX_LABEL, True), (module.RESEARCH_PRESET_MODEL_LABEL, True)]
    module.st.form_submit_button.assert_called_once_with(module.RESEARCH_PRESET_SAVE_LABEL)


def test_saving_a_preset_keeps_the_settings_in_use_under_the_typed_name(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    saved = {"quality_preset": "Fast", "chunk_chars": 3000, "rate_profile": "Low limits"}

    patched = draw_research_presets(
        module, saved=saved, model=OLLAMA_MODEL, submitted=True, typed="Quick", save_result=("Quick", False)
    )

    patched["save"].assert_called_once_with(
        "Quick", {"answer": {"quality_preset": "Fast"}, "index": {"chunk_chars": 3000}, "model": OLLAMA_MODEL}
    )
    state = module.st.session_state
    assert state.research_preset_selected == "Quick"
    assert state.research_preset_round == 1
    assert state.research_preset_notice == "Saved the preset “Quick”."
    module.st.rerun.assert_called_once_with()


def test_saving_a_preset_can_leave_out_the_index_and_model(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(
        module, saved={"quality_preset": "Fast", "chunk_chars": 3000}, checks=(False, False), submitted=True, typed="Q"
    )

    patched["save"].assert_called_once_with("Q", {"answer": {"quality_preset": "Fast"}})


def test_saving_under_a_saved_name_says_the_preset_was_replaced(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Quick": QUICK}, submitted=True, typed="quick", save_result=("quick", True))

    assert module.st.session_state.research_preset_notice == "The preset “quick” now holds the settings in use."


def test_a_preset_that_cannot_be_saved_says_why(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    with mock.patch.object(module.research_presets, "save_preset", side_effect=ValueError("Type a name for the preset.")):
        module.st.rerun = mock.Mock()
        module.st.error = mock.Mock()
        module.st.form_submit_button = mock.Mock(return_value=True)
        module.save_research_preset("", True, True, {}, None)

    module.st.error.assert_called_once_with(f"{module.RESEARCH_PRESET_FAILED_NOTE} Type a name for the preset\\.")
    module.st.rerun.assert_not_called()


def test_the_research_preset_notice_shows_once(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_notice = "Saved the preset “Quick”."

    draw_research_presets(module, presets={"Quick": QUICK})

    module.st.success.assert_called_once_with("Saved the preset “Quick”.")
    assert "research_preset_notice" not in module.st.session_state


def test_a_template_change_shows_in_the_research_preset_line_in_the_same_run(pdf_research_assistant_module, tmp_path):
    """The template control saves without drawing the page again, so the presets must be read after it."""
    module = pdf_research_assistant_module
    st = module.st
    presets_path = tmp_path / "research_presets.json"
    presets_path.write_text(json.dumps({"Notes": {"answer": {"answer_template": "Study notes"}}}), encoding="utf-8")
    st.markdown = mock.Mock()
    st.selectbox = mock.Mock(
        side_effect=lambda label, options, index=0, **_kwargs: "Study notes" if label == "Answer template" else options[index]
    )
    st.session_state.update(_empty_state())
    with (
        mock.patch.object(module.research_presets, "PRESETS_PATH", presets_path),
        mock.patch.object(module.saved_settings, "SETTINGS_PATH", tmp_path / "saved_settings.json"),
    ):
        module.render_sidebar({}, _fake_settings())

    assert module.saved_settings.load(tmp_path / "saved_settings.json") == {"answer_template": "Study notes"}
    assert "**Research preset:** Notes" in [call.args[0] for call in st.markdown.call_args_list]


def test_a_preset_saved_from_the_app_is_restored_when_it_is_used(pdf_research_assistant_module, tmp_path):
    """Save the settings in use as a preset, change them, then use the preset: every setting comes back."""
    module = pdf_research_assistant_module
    settings_path = tmp_path / "saved_settings.json"
    model_path = tmp_path / "local_llm.json"
    st = module.st
    st.rerun = mock.Mock()
    with (
        mock.patch.object(module.saved_settings, "SETTINGS_PATH", settings_path),
        mock.patch.object(module.local_llm, "CONFIG_PATH", model_path),
        mock.patch.object(module.research_presets, "PRESETS_PATH", tmp_path / "research_presets.json"),
    ):
        for key, value in {"quality_preset": "Fast", "grounding": "Strictly grounded", "overlap": 500}.items():
            module.saved_settings.save_value(key, value)
        module.local_llm.save_config(OLLAMA_MODEL)
        module.save_research_preset("Quick", True, True, module.saved_settings.load(), module.local_llm.load_config())
        kept = module.saved_settings.load()

        module.saved_settings.replace_all({"evidence_k": 20, "chunk_chars": 3000})
        module.local_llm.clear_config()
        module.use_research_preset("Quick", module.research_presets.load()["Quick"])

        assert module.saved_settings.load() == kept
        assert module.local_llm.load_config() == OLLAMA_MODEL
        assert module.research_presets.find_active(
            module.research_presets.load(), module.saved_settings.load(), module.local_llm.load_config()
        ) == ["Quick"]


def test_the_research_preset_panel_keeps_its_own_values_apart_from_its_controls(pdf_research_assistant_module):
    """Streamlit owns the value under a control's key, so a value the panel keeps must never share one."""
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_to_delete = "Quick"
    module.st.session_state.research_preset_notice = "Saved."

    draw_research_presets(module, presets={"Quick": QUICK, "Deep": DEEP}, clicked={"research_preset_delete_confirm"})
    widget_keys = _research_preset_widget_keys(module)
    module.st.session_state.research_preset_file_notice = "Imported."
    draw_research_presets(
        module,
        presets={"Quick": QUICK, "Local": LOCAL},
        uploaded=_presets_file({"quick": QUICK}),
        clicked={"research_preset_import"},
    )
    widget_keys |= _research_preset_widget_keys(module)

    kept = {key for key in module.st.session_state if key.startswith("research_preset_")}
    assert {
        "research_preset_to_delete",
        "research_preset_selected",
        "research_preset_round",
        "research_preset_file_notice",
    } <= kept
    assert not kept & widget_keys


# Importing and exporting research presets.

LOCAL = {"answer": {"evidence_k": 3}, "model": OLLAMA_MODEL}


def _presets_file(presets):
    return research_presets.export_text(presets, list(presets), include_model_details=True).encode("utf-8")


def _download(module):
    return module.st.download_button.call_args


def test_export_offers_every_preset_and_downloads_them_as_a_presets_file(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Deep": DEEP, "Quick": QUICK})

    module.st.expander.assert_any_call(module.RESEARCH_PRESET_FILES_PANEL)
    pick = module.st.multiselect.call_args
    assert pick.args == (module.RESEARCH_PRESET_EXPORT_PICK_LABEL, ["Deep", "Quick"])
    assert pick.kwargs["default"] == ["Deep", "Quick"]
    assert pick.kwargs["key"] == "research_preset_export_pick_0"
    call = _download(module)
    assert call.args[0] == module.RESEARCH_PRESET_EXPORT_LABEL
    assert json.loads(call.args[1])["presets"] == {"Deep": DEEP, "Quick": QUICK}
    assert call.kwargs["file_name"] == "research-presets.json"
    assert call.kwargs["mime"] == "application/json"
    assert call.kwargs["on_click"] == "ignore"
    # No preset uses a local model, so there is no address to ask about.
    assert [c.kwargs["key"] for c in module.st.checkbox.call_args_list if c.kwargs.get("key") in FILE_BOXES] == []


def test_export_holds_only_the_presets_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Deep": DEEP, "Quick": QUICK}, exported=["Quick"])

    assert list(json.loads(_download(module).args[1])["presets"]) == ["Quick"]


def test_export_with_nothing_picked_says_so_and_offers_no_download(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Quick": QUICK}, exported=[])

    module.st.caption.assert_any_call(module.RESEARCH_PRESET_EXPORT_NONE_NOTE)
    module.st.download_button.assert_not_called()


def test_export_leaves_out_local_model_addresses_unless_asked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Local": LOCAL, "Quick": QUICK})

    box = next(c for c in module.st.checkbox.call_args_list if c.kwargs.get("key") == "research_preset_export_model")
    assert box.args == (module.RESEARCH_PRESET_EXPORT_MODEL_LABEL,)
    assert box.kwargs["value"] is False
    assert box.kwargs["help"] == module.RESEARCH_PRESET_EXPORT_MODEL_HELP
    text = _download(module).args[1]
    assert "localhost" not in text
    assert json.loads(text)["presets"]["Local"] == {"answer": {"evidence_k": 3}}


def test_export_includes_local_model_addresses_when_ticked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, presets={"Local": LOCAL}, ticked={"research_preset_export_model": True})

    assert json.loads(_download(module).args[1])["presets"]["Local"] == LOCAL


def test_with_no_presets_saved_only_import_is_offered(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module)

    module.st.multiselect.assert_not_called()
    module.st.download_button.assert_not_called()
    upload = module.st.file_uploader.call_args
    assert upload.args == (module.RESEARCH_PRESET_IMPORT_FILE_LABEL,)
    assert upload.kwargs["type"] == ["json"]
    # In megabytes: a little over the most the check reads.
    assert upload.kwargs["max_upload_size"] * 1024 * 1024 >= research_presets.MAX_IMPORT_BYTES
    assert upload.kwargs["key"] == "research_preset_import_file_0"


def test_a_file_that_cannot_be_imported_lists_its_problems_and_offers_no_import(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, uploaded=b"not json")

    module.st.error.assert_called_once_with(
        f"{module.RESEARCH_PRESET_IMPORT_REFUSED_NOTE}\n\n- "
        + module.escape_markdown("The file is not a presets file: it is not JSON text.")
    )
    assert "research_preset_import" not in _button_keys(module)


def test_a_checked_file_shows_its_presets_and_imports_them_when_asked(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(
        module,
        presets={"Deep": DEEP},
        uploaded=_presets_file({"Quick": QUICK, "Notes*": QUICK}),
        clicked={"research_preset_import"},
        import_result=(["Quick", "Notes*"], [], []),
    )

    assert r"The file holds 2 presets: Quick, Notes\*" in _markdown_lines(module)
    patched["import"].assert_called_once_with({"Quick": QUICK, "Notes*": QUICK}, False)
    state = module.st.session_state
    assert state.research_preset_file_notice == r"Imported “Quick”, “Notes\*”."
    assert state.research_preset_selected == "Quick"
    assert state.research_preset_round == 1
    assert "research_preset_notice" not in state
    module.st.rerun.assert_called_once_with()


def test_a_file_with_one_preset_says_so(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(module, uploaded=_presets_file({"Quick": QUICK}))

    assert "The file holds 1 preset: Quick" in _markdown_lines(module)
    assert "research_preset_import" in _button_keys(module)
    patched["import"].assert_not_called()


def test_importing_names_already_saved_asks_whether_to_replace_them(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(
        module,
        presets={"Quick": QUICK},
        uploaded=_presets_file({"quick": DEEP, "New": QUICK}),
        clicked={"research_preset_import"},
        import_result=(["New"], [], ["quick"]),
    )

    box = next(c for c in module.st.checkbox.call_args_list if c.kwargs.get("key") == "research_preset_import_replace")
    assert box.args == (module.RESEARCH_PRESET_IMPORT_REPLACE_LABEL,)
    assert box.kwargs["value"] is False
    assert "quick" in box.kwargs["help"]
    patched["import"].assert_called_once_with({"quick": DEEP, "New": QUICK}, False)
    assert module.st.session_state.research_preset_file_notice == (
        "Imported “New”. Skipped “quick”, because a saved preset has the same name."
    )


def test_importing_can_replace_saved_presets_with_the_same_name(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    patched = draw_research_presets(
        module,
        presets={"Quick": QUICK},
        uploaded=_presets_file({"quick": DEEP}),
        ticked={"research_preset_import_replace": True},
        clicked={"research_preset_import"},
        import_result=([], ["quick"], []),
    )

    patched["import"].assert_called_once_with({"quick": DEEP}, True)
    assert module.st.session_state.research_preset_file_notice == "Replaced “quick”."
    assert module.st.session_state.research_preset_selected == "quick"


def test_an_import_that_skips_every_preset_keeps_the_preset_picked(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_selected = "Deep"

    draw_research_presets(
        module,
        presets={"Quick": QUICK, "Deep": DEEP},
        uploaded=_presets_file({"quick": DEEP}),
        clicked={"research_preset_import"},
        import_result=([], [], ["quick"]),
    )

    assert module.st.session_state.research_preset_selected == "Deep"


def test_importing_a_preset_with_a_local_model_says_to_check_its_address(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(module, uploaded=_presets_file({"Local": LOCAL, "Quick": QUICK}))

    module.st.caption.assert_any_call(module.RESEARCH_PRESET_IMPORT_MODEL_NOTE.format(count=1))


def test_an_import_that_cannot_be_saved_says_why(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    draw_research_presets(
        module,
        uploaded=_presets_file({"Quick": QUICK}),
        clicked={"research_preset_import"},
        import_error=OSError("disk full"),
    )

    module.st.error.assert_called_once_with(f"{module.RESEARCH_PRESET_IMPORT_FAILED_NOTE} disk full")
    module.st.rerun.assert_not_called()


def test_the_import_notice_shows_once_in_the_import_and_export_panel(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.research_preset_file_notice = "Imported “Quick”."

    draw_research_presets(module, presets={"Quick": QUICK})

    module.st.success.assert_called_once_with("Imported “Quick”.")
    assert "research_preset_file_notice" not in module.st.session_state


def test_describe_import_names_what_was_added_replaced_and_skipped(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    assert module.describe_import(["A"], ["B", "C"], ["D"]) == (
        "Imported “A”. Replaced “B”, “C”. Skipped “D”, because a saved preset has the same name."
    )
    assert module.describe_import([], [], []) == ""


def test_presets_exported_from_the_app_are_imported_into_the_normal_preset_list(pdf_research_assistant_module, tmp_path):
    """Export from one presets file, import into another: the presets can then be picked and used as usual."""
    module = pdf_research_assistant_module
    st = module.st
    for name in ("rerun", "success", "error", "caption", "markdown", "warning", "subheader", "download_button"):
        setattr(st, name, mock.Mock())
    st.button = mock.Mock(side_effect=lambda _label, key=None, **_kwargs: key == "research_preset_import")
    st.form_submit_button = mock.Mock(return_value=False)
    st.checkbox = mock.Mock(side_effect=lambda _label, value=False, **_kwargs: value)
    st.multiselect = mock.Mock(side_effect=lambda _label, options, default=None, **_kwargs: list(default))
    st.file_uploader = mock.Mock(return_value=None)
    source, target = tmp_path / "here.json", tmp_path / "there.json"
    source.write_text(json.dumps({"Deep": DEEP, "Local": LOCAL}), encoding="utf-8")
    with (
        mock.patch.object(module.saved_settings, "SETTINGS_PATH", tmp_path / "saved_settings.json"),
        mock.patch.object(module.local_llm, "CONFIG_PATH", tmp_path / "local_llm.json"),
    ):
        with mock.patch.object(module.research_presets, "PRESETS_PATH", source):
            module.render_research_presets(st.sidebar)
        exported = _download(module).args[1].encode("utf-8")

        st.file_uploader = mock.Mock(return_value=SimpleNamespace(getvalue=lambda: exported))
        with mock.patch.object(module.research_presets, "PRESETS_PATH", target):
            module.render_research_presets(st.sidebar)
            st.file_uploader = mock.Mock(return_value=None)
            st.button = mock.Mock(return_value=False)
            st.selectbox = mock.Mock(side_effect=lambda _label, options, index=0, **_kwargs: options[index])
            module.render_research_presets(st.sidebar)

    assert module.research_presets.load(target) == {"Deep": DEEP, "Local": {"answer": {"evidence_k": 3}}}
    assert st.selectbox.call_args.args == (module.RESEARCH_PRESET_PICK_LABEL, ["Deep", "Local"])


def _research_preset_widget_keys(module):
    controls = [
        module.st.button,
        module.st.selectbox,
        module.st.text_input,
        module.st.checkbox,
        module.st.multiselect,
        module.st.file_uploader,
        module.st.download_button,
    ]
    return {call.kwargs.get("key") for control in controls for call in control.call_args_list}


FOUND_ANSWER = {
    "id": "a1",
    "question": "Why is the sky blue?",
    "content": "Rayleigh scattering.",
    "cost": 0.0123,
    "asked_at": "2026-11-24T14:05:00+00:00",
    "contexts": [{"name": "Source", "context": "Quoted text"}],
    "settings": {"index": "pqa_index_x", "values": []},
}


@pytest.mark.parametrize("asked_at", [None, "", "not a time", 5])
def test_describe_asked_at_gives_none_when_the_time_is_not_known(pdf_research_assistant_module, asked_at):
    assert pdf_research_assistant_module.describe_asked_at(asked_at) is None


def test_describe_asked_at_gives_the_day_and_time(pdf_research_assistant_module):
    assert pdf_research_assistant_module.describe_asked_at("2026-11-24T14:05:09+00:00") == "24 Nov 2026, 14:05"


def test_describe_found_answer_says_when_on_which_index_and_at_what_cost(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    line = module.describe_found_answer(FOUND_ANSWER, {"pqa_index_x": "Thesis"})

    assert line == "Asked 24 Nov 2026, 14:05 · Index: **Thesis** (`pqa_index_x`) · Cost: $0.0123"


def test_describe_found_answer_gives_only_the_cost_when_nothing_else_is_known(pdf_research_assistant_module):
    answer = FOUND_ANSWER | {"asked_at": None, "settings": None, "cost": None}

    assert pdf_research_assistant_module.describe_found_answer(answer, {}) == "Cost: $0.0000"


def test_app_formats_costs_with_the_shared_helper_the_cli_uses(pdf_research_assistant_module):
    from pdf_research_assistant import costs

    assert pdf_research_assistant_module.format_cost is costs.format_cost


def test_describe_found_answer_shows_a_tiny_cost_as_less_than_a_hundredth_of_a_cent(pdf_research_assistant_module):
    answer = FOUND_ANSWER | {"asked_at": None, "settings": None, "cost": 0.00000018}

    assert pdf_research_assistant_module.describe_found_answer(answer, {}) == "Cost: less than $0.0001"


def test_describe_search_result_says_when_nothing_matches(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    note = module.describe_search_result({"text": "sky*", "answers": [], "searched": 1})

    assert note == f"No previous answer matches **{module.escape_markdown('sky*')}**. Searched 1 kept answer."


def test_describe_search_result_counts_one_match(pdf_research_assistant_module):
    note = pdf_research_assistant_module.describe_search_result({"text": "sky", "answers": [FOUND_ANSWER], "searched": 3})

    assert note == "1 previous answer matches **sky**. Searched 3 kept answers."


def test_describe_search_result_counts_several_matches(pdf_research_assistant_module):
    result = {"text": "sky", "answers": [FOUND_ANSWER] * 2, "searched": 3}

    assert pdf_research_assistant_module.describe_search_result(result).startswith("2 previous answers match **sky**")


def test_describe_search_result_says_only_the_best_matches_are_shown(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    result = {"text": "sky", "answers": [FOUND_ANSWER] * module.saved_answers.MAX_RESULTS, "searched": 40}

    assert module.describe_search_result(result).endswith(
        f"Searched 40 kept answers. Only the best {module.saved_answers.MAX_RESULTS} are shown."
    )


def test_search_previous_answers_asks_for_words_when_none_were_typed(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    with mock.patch.object(module.saved_answers, "search") as search:
        module.search_previous_answers("   ")

    search.assert_not_called()
    assert module.st.session_state.answer_search_found == {"note": module.ANSWER_SEARCH_EMPTY_NOTE}


def test_search_previous_answers_searches_the_index_folder_and_keeps_what_it_found(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    found = {"text": "sky", "answers": [], "searched": 0}

    with mock.patch.object(module.saved_answers, "search", return_value=found) as search:
        module.search_previous_answers("  sky ")

    search.assert_called_once_with(module.bootstrap.INDEX_DIR, "sky")
    assert module.st.session_state.answer_search_found == found


def test_search_previous_answers_explains_a_search_paperqa_cannot_read(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    with mock.patch.object(module.saved_answers, "search", side_effect=ValueError("Syntax Error: AND")):
        module.search_previous_answers("AND")

    assert module.st.session_state.answer_search_found == {"note": module.ANSWER_SEARCH_UNREADABLE_NOTE}


def test_search_previous_answers_keeps_why_the_search_failed(pdf_research_assistant_module):
    module = pdf_research_assistant_module

    with mock.patch.object(module.saved_answers, "search", side_effect=OSError("locked")):
        module.search_previous_answers("sky")

    assert module.st.session_state.answer_search_found == {"error": "OSError: locked"}


def test_render_found_answer_shows_the_question_when_where_the_answer_its_passages_and_settings(
    pdf_research_assistant_module,
):
    module = pdf_research_assistant_module
    st = module.st
    for shown in (st.markdown, st.caption):
        shown.reset_mock()
    answer = FOUND_ANSWER | {"question": "Why *blue*?", "content": "Costs $5 to $10."}

    with mock.patch.object(module, "render_source_passages") as passages, mock.patch.object(
        module, "render_answer_settings"
    ) as answer_settings:
        module.render_found_answer(answer, {})

    assert [call.args[0] for call in st.markdown.call_args_list] == [
        f"**{module.escape_markdown('Why *blue*?')}**",
        module.escape_dollars("Costs $5 to $10."),
    ]
    st.caption.assert_called_once_with(module.describe_found_answer(answer, {}))
    passages.assert_called_once_with(FOUND_ANSWER["contexts"], key="found_a1")
    answer_settings.assert_called_once_with(FOUND_ANSWER["settings"])


def test_render_found_answer_explains_an_answer_kept_by_paperqa_itself(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.caption.reset_mock()
    answer = FOUND_ANSWER | {"asked_at": None, "contexts": [], "settings": None}

    module.render_found_answer(answer, {})

    assert module.st.caption.call_args_list[-1] == mock.call(module.NO_ANSWER_DETAILS_NOTE)


def test_render_search_results_shows_why_the_search_failed(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.error.reset_mock()

    module.render_search_results({"error": "ValueError: bad"})

    module.st.error.assert_called_once_with(f"{module.ANSWER_SEARCH_FAILED_NOTE} {module.escape_markdown('ValueError: bad')}")


def test_render_search_results_shows_a_note_instead_of_answers(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.caption.reset_mock()

    with mock.patch.object(module, "render_found_answer") as render_found_answer:
        module.render_search_results({"note": module.ANSWER_SEARCH_EMPTY_NOTE})

    module.st.caption.assert_called_once_with(module.ANSWER_SEARCH_EMPTY_NOTE)
    render_found_answer.assert_not_called()


def test_render_search_results_shows_each_answer_found_with_the_index_names(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    found = {"text": "sky", "answers": [FOUND_ANSWER, FOUND_ANSWER | {"id": "a2"}], "searched": 2}
    module.index_names.load.return_value = {"pqa_index_x": "Thesis"}
    module.st.markdown.reset_mock()

    with mock.patch.object(module, "render_found_answer") as render_found_answer:
        module.render_search_results(found)

    module.st.markdown.assert_called_once_with(module.describe_search_result(found))
    module.index_names.load.assert_called_with(module.bootstrap.INDEX_DIR)
    assert render_found_answer.call_args_list == [
        mock.call(FOUND_ANSWER, {"pqa_index_x": "Thesis"}),
        mock.call(FOUND_ANSWER | {"id": "a2"}, {"pqa_index_x": "Thesis"}),
    ]


def test_render_search_results_leaves_the_results_up_until_cleared(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.rerun.reset_mock()
    module.st.session_state.answer_search_found = {"note": "x"}

    module.render_search_results({"note": "x"})

    assert module.st.session_state.answer_search_found == {"note": "x"}
    module.st.rerun.assert_not_called()


def test_clearing_the_search_results_puts_them_away():
    with imported_pdf_research_assistant(clicked_keys={"answer_search_clear"}) as module:
        module.st.rerun.reset_mock()
        module.st.session_state.answer_search_found = {"note": "x"}

        module.render_search_results({"note": "x"})

        assert module.st.session_state.answer_search_found is None
        module.st.rerun.assert_called_once_with()


def test_render_answer_search_draws_the_box_and_no_results_before_a_search(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    st = module.st
    st.text_input.reset_mock()
    st.session_state.pop("answer_search_found", None)

    with mock.patch.object(module, "search_previous_answers") as search, mock.patch.object(
        module, "render_search_results"
    ) as results:
        module.render_answer_search()

    st.text_input.assert_called_once_with(
        module.ANSWER_SEARCH_LABEL, key="answer_search_text", help=module.ANSWER_SEARCH_HELP
    )
    search.assert_not_called()
    results.assert_not_called()


def test_render_answer_search_searches_when_the_button_is_pressed_and_shows_what_it_found(
    pdf_research_assistant_module,
):
    module = pdf_research_assistant_module
    st = module.st
    st.text_input = mock.Mock(return_value="sky")
    st.form_submit_button = mock.Mock(return_value=True)
    found = {"text": "sky", "answers": [], "searched": 0}

    def search(_text):
        st.session_state.answer_search_found = found

    with mock.patch.object(module, "search_previous_answers", side_effect=search) as search_mock, mock.patch.object(
        module, "render_search_results"
    ) as results:
        module.render_answer_search()

    search_mock.assert_called_once_with("sky")
    results.assert_called_once_with(found)


def test_render_answer_search_keeps_showing_the_last_results_on_later_runs(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.answer_search_found = {"note": "x"}

    with mock.patch.object(module, "render_search_results") as results:
        module.render_answer_search()

    results.assert_called_once_with({"note": "x"})


def test_page_draws_the_answer_search_above_the_chat(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    steps = mock.Mock()

    with mock.patch.object(module, "render_answer_search", steps.search), mock.patch.object(
        module, "render_history", steps.history
    ):
        module.main()

    assert [name for name, *_ in steps.mock_calls] == ["search", "history"]


def test_render_history_says_when_an_answer_could_not_be_kept(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.button = mock.Mock(return_value=False)
    module.st.caption.reset_mock()
    item = {"role": "assistant", "content": "A", "cost": 0.0, "contexts": [], "id": 1, "save_error": "OSError: full"}

    module.render_history([item])

    assert module.st.caption.call_args_list[-1] == mock.call(
        f"{module.ANSWER_NOT_SAVED_NOTE} {module.escape_markdown('OSError: full')}"
    )


def test_submit_question_records_why_the_answer_could_not_be_kept(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.update({"history": [], "total_cost": 0.0, "query_count": 0})

    with mock.patch.object(module, "answer_question", return_value=("A", 0.0, [], None, "OSError: full")):
        module.submit_question("Q", {})

    assert module.st.session_state.history[-1]["save_error"] == "OSError: full"


def test_submit_question_records_no_save_problem_when_the_answer_was_kept(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    module.st.session_state.update({"history": [], "total_cost": 0.0, "query_count": 0})

    with mock.patch.object(module, "answer_question", return_value=("A", 0.0, [], None, None)):
        module.submit_question("Q", {})

    assert "save_error" not in module.st.session_state.history[-1]


# The settings that decide which PDFs are searched: the paper list and subfolders.


def _scope_files(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def draw_scope_settings(
    module,
    tmp_path,
    *,
    files=("a.pdf", "b.pdf", "sub/c.pdf"),
    listed=frozenset(),
    saved_scope=("manifest", True),
    indexed=frozenset(),
    paper_list_found=True,
    settings=None,
    chosen=None,
    submitted=False,
    clicked=(),
    pending=None,
    rebuilding=False,
):
    """Draw the scope settings over a real folder of PDFs, with the user choosing values and perhaps clicking.

    chosen maps a control's label to the value the user leaves in it.
    """
    st = module.st
    chosen = chosen or {}
    _scope_files(tmp_path, *files)
    for shown in (st.caption, st.expander, st.rerun, st.error, st.info, st.success, st.warning, st.markdown):
        shown.reset_mock()
    st.session_state.pop("scope_pending", None)
    if pending:
        st.session_state.scope_pending = pending
    st.selectbox = mock.Mock(side_effect=lambda label, options, index=0, **_kwargs: chosen.get(label, options[index]))
    st.checkbox = mock.Mock(side_effect=lambda label, value=False, **_kwargs: chosen.get(label, value))
    st.form = mock.Mock(return_value=_ContextManagerStub())
    st.form_submit_button = mock.Mock(return_value=submitted)
    st.button = mock.Mock(side_effect=lambda _label, key=None, **_kwargs: key in clicked)
    run = {"process": mock.Mock()} if rebuilding else None
    with mock.patch.multiple(
        module.bootstrap,
        PAPER_DIR=tmp_path,
        MANIFEST_PATH=tmp_path / "manifest.csv",
        saved_scope=mock.Mock(return_value=saved_scope),
        manifest_exists=mock.Mock(return_value=paper_list_found),
        load_allowed_manifest_paths=mock.Mock(return_value=set(listed)),
        get_index_entries=mock.Mock(return_value=set(indexed)),
    ), mock.patch.object(module.rebuild_client, "get_run", return_value=run), mock.patch.multiple(
        module.saved_settings, save_value=mock.DEFAULT
    ) as patched:
        module.render_scope_settings(settings or _scope_fake_settings(saved_scope, paper_list_found))
        return patched


def _scope_fake_settings(saved_scope=("manifest", True), paper_list_found=True):
    settings = _fake_settings()
    settings.agent.index.manifest_file = Path("manifest.csv") if saved_scope[0] == "manifest" and paper_list_found else None
    settings.agent.index.recurse_subdirectories = saved_scope[1]
    return settings


def test_the_scope_settings_sit_in_their_own_panel_and_form_so_nothing_changes_until_saved(
    pdf_research_assistant_module, tmp_path
):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(module, tmp_path)

    module.st.expander.assert_called_once_with(module.SCOPE_SETTINGS_PANEL)
    module.st.form.assert_called_once_with("scope_settings_form")
    module.st.form_submit_button.assert_called_once_with(module.SCOPE_SETTINGS_SAVE_LABEL, disabled=False)
    module.st.caption.assert_any_call(module.SCOPE_SETTINGS_NOTE)
    patched["save_value"].assert_not_called()
    module.st.rerun.assert_not_called()


def test_the_scope_controls_start_at_the_values_in_use(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    get_setting = module.settings_groups.get_setting

    draw_scope_settings(module, tmp_path, saved_scope=("all", False))

    paper_list = module.st.selectbox.call_args
    assert paper_list.args[0] == "Paper list"
    assert paper_list.args[1] == ["manifest", "all"]
    assert paper_list.kwargs["index"] == 1
    assert paper_list.kwargs["key"] == "scope_setting_manifest_scope_0"
    assert paper_list.kwargs["help"] == get_setting("manifest_scope").help
    assert paper_list.kwargs["disabled"] is False
    assert [paper_list.kwargs["format_func"](value) for value in paper_list.args[1]] == [
        "Only the PDFs in the paper list",
        "All PDFs in the folder",
    ]
    subfolders = module.st.checkbox.call_args
    assert subfolders.args[0] == "Search subfolders"
    assert subfolders.kwargs == {
        "value": False,
        "help": get_setting("recurse_subdirectories").help,
        "key": "scope_setting_recurse_subdirectories_0",
    }


def test_the_paper_list_control_is_switched_off_when_there_is_no_paper_list(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    draw_scope_settings(module, tmp_path, paper_list_found=False)

    assert module.st.selectbox.call_args.kwargs["disabled"] is True
    module.st.caption.assert_any_call(module.SCOPE_NO_PAPER_LIST_NOTE)


def test_the_no_paper_list_note_is_left_out_when_there_is_a_paper_list(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    draw_scope_settings(module, tmp_path)

    assert mock.call(module.SCOPE_NO_PAPER_LIST_NOTE) not in module.st.caption.call_args_list


def test_saving_scope_without_a_change_saves_nothing(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(module, tmp_path, submitted=True, indexed={"a.pdf"})

    patched["save_value"].assert_not_called()
    module.st.info.assert_called_once_with(module.INDEX_SETTINGS_UNCHANGED_NOTE)
    assert "scope_pending" not in module.st.session_state or not module.st.session_state.scope_pending


def test_turning_subfolders_off_waits_for_confirmation_and_says_how_many_pdfs_leave_the_index(
    pdf_research_assistant_module, tmp_path
):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(
        module,
        tmp_path,
        chosen={"Search subfolders": False},
        submitted=True,
        indexed={"a.pdf", "b.pdf", "sub/c.pdf"},
    )

    patched["save_value"].assert_not_called()
    assert module.st.session_state.scope_pending == {
        "values": {"recurse_subdirectories": False},
        "preview": {"added": [], "removed": ["sub/c.pdf"], "already_indexed": []},
    }
    module.st.rerun.assert_called_once_with()


def test_turning_subfolders_on_says_how_many_pdfs_would_be_added(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(
        module,
        tmp_path,
        saved_scope=("manifest", False),
        chosen={"Search subfolders": True},
        submitted=True,
        indexed={"a.pdf", "b.pdf"},
    )

    patched["save_value"].assert_not_called()
    assert module.st.session_state.scope_pending["preview"] == {
        "added": ["sub/c.pdf"],
        "removed": [],
        "already_indexed": [],
    }


def test_leaving_the_paper_list_out_adds_the_unlisted_pdfs(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    draw_scope_settings(
        module,
        tmp_path,
        listed={"a.pdf"},
        chosen={"Paper list": "all"},
        submitted=True,
        indexed={"a.pdf"},
    )

    assert module.st.session_state.scope_pending == {
        "values": {"manifest_scope": "all"},
        "preview": {"added": ["b.pdf", "sub/c.pdf"], "removed": [], "already_indexed": []},
    }


def test_using_the_paper_list_removes_the_unlisted_pdfs_from_the_index(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    draw_scope_settings(
        module,
        tmp_path,
        listed={"a.pdf"},
        saved_scope=("all", True),
        chosen={"Paper list": "manifest"},
        submitted=True,
        indexed={"a.pdf", "b.pdf", "sub/c.pdf"},
    )

    assert module.st.session_state.scope_pending["preview"] == {
        "added": [],
        "removed": ["b.pdf", "sub/c.pdf"],
        "already_indexed": [],
    }


def test_a_scope_change_that_adds_or_removes_nothing_is_saved_at_once(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(
        module,
        tmp_path,
        files=("a.pdf", "b.pdf"),
        chosen={"Search subfolders": False},
        submitted=True,
        indexed={"a.pdf", "b.pdf"},
    )

    patched["save_value"].assert_called_once_with("recurse_subdirectories", False)
    assert module.st.session_state.scope_settings_notice == module.SCOPE_NO_CHANGE_NOTICE
    assert module.st.session_state.scope_pending is None
    module.st.rerun.assert_called_once_with()


def test_pdfs_coming_into_scope_that_are_already_indexed_need_no_confirmation(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(
        module,
        tmp_path,
        saved_scope=("manifest", False),
        chosen={"Search subfolders": True},
        submitted=True,
        indexed={"a.pdf", "b.pdf", "sub/c.pdf"},
    )

    patched["save_value"].assert_called_once_with("recurse_subdirectories", True)
    assert module.st.session_state.scope_settings_notice == module.SCOPE_NO_CHANGE_NOTICE


def test_a_scope_change_is_saved_at_once_when_the_index_holds_no_pdfs_yet(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(module, tmp_path, chosen={"Search subfolders": False}, submitted=True)

    patched["save_value"].assert_called_once_with("recurse_subdirectories", False)
    assert module.st.session_state.scope_settings_notice == module.SCOPE_EMPTY_INDEX_NOTICE


def test_a_saved_scope_change_gives_the_controls_fresh_keys(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    module.st.session_state.pop("scope_settings_round", None)

    draw_scope_settings(module, tmp_path, chosen={"Search subfolders": False}, submitted=True)

    assert module.st.session_state.scope_settings_round == 1


def test_subfolders_cannot_be_turned_off_when_the_paper_list_names_only_pdfs_in_them(
    pdf_research_assistant_module, tmp_path
):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(
        module,
        tmp_path,
        listed={"sub/c.pdf"},
        chosen={"Search subfolders": False},
        submitted=True,
        indexed={"sub/c.pdf"},
    )

    patched["save_value"].assert_not_called()
    module.st.error.assert_called_once_with(module.SCOPE_NOTHING_LEFT_NOTE)
    module.st.rerun.assert_not_called()


def test_a_scope_change_waiting_for_confirmation_shows_the_warning_with_the_counts(
    pdf_research_assistant_module, tmp_path
):
    module = pdf_research_assistant_module
    pending = {
        "values": {"recurse_subdirectories": False},
        "preview": {"added": ["x.pdf"], "removed": ["sub/c.pdf"], "already_indexed": []},
    }

    patched = draw_scope_settings(module, tmp_path, pending=pending)

    (warning,) = [call.args[0] for call in module.st.warning.call_args_list]
    assert "This changes the index you are using. It does not start a new one." in warning
    assert "`active-shard`" in warning
    assert "1 PDF would be added to the index and read, which costs money: `x.pdf`" in warning
    assert "1 PDF would be removed from the index: `sub/c.pdf`" in warning
    assert "Nothing is saved until you confirm." in warning
    patched["save_value"].assert_not_called()


def test_the_scope_warning_names_the_index_the_user_gave_it(pdf_research_assistant_module):
    module = pdf_research_assistant_module
    preview = {"added": [], "removed": ["a.pdf"], "already_indexed": []}

    text = module.describe_scope_change(preview, Path("index/pqa_index_1"), "My papers")

    assert "**My papers** (`pqa_index_1`)" in text


def test_confirming_a_scope_change_saves_every_value_and_draws_the_page_again(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    pending = {
        "values": {"manifest_scope": "all", "recurse_subdirectories": False},
        "preview": {"added": [], "removed": ["a.pdf"], "already_indexed": []},
    }

    patched = draw_scope_settings(module, tmp_path, pending=pending, clicked={"scope_confirm"})

    assert patched["save_value"].call_args_list == [
        mock.call("manifest_scope", "all"),
        mock.call("recurse_subdirectories", False),
    ]
    assert module.st.session_state.scope_pending is None
    assert module.st.session_state.scope_settings_notice == module.SCOPE_SAVED_NOTICE
    module.st.rerun.assert_called_once_with()


def test_cancelling_a_scope_change_saves_nothing_and_resets_the_controls(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    module.st.session_state.pop("scope_settings_round", None)
    pending = {"values": {"manifest_scope": "all"}, "preview": {"added": ["a.pdf"], "removed": [], "already_indexed": []}}

    patched = draw_scope_settings(module, tmp_path, pending=pending, clicked={"scope_cancel"})

    patched["save_value"].assert_not_called()
    assert module.st.session_state.scope_pending is None
    assert module.st.session_state.scope_settings_round == 1
    module.st.rerun.assert_called_once_with()


def test_the_scope_settings_cannot_be_saved_while_a_rebuild_runs(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module

    patched = draw_scope_settings(
        module, tmp_path, chosen={"Search subfolders": False}, submitted=True, indexed={"sub/c.pdf"}, rebuilding=True
    )

    patched["save_value"].assert_not_called()
    module.st.form_submit_button.assert_called_once_with(module.SCOPE_SETTINGS_SAVE_LABEL, disabled=True)
    module.st.caption.assert_any_call(module.REBUILD_LOCKED_NOTE)
    module.st.rerun.assert_not_called()


def test_a_scope_notice_is_shown_once(pdf_research_assistant_module, tmp_path):
    module = pdf_research_assistant_module
    module.st.session_state.scope_settings_notice = module.SCOPE_SAVED_NOTICE

    draw_scope_settings(module, tmp_path)
    module.st.success.assert_called_once_with(module.SCOPE_SAVED_NOTICE)
    module.st.success.reset_mock()
    draw_scope_settings(module, tmp_path)

    module.st.success.assert_not_called()
