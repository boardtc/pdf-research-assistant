import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture
def query_papers_module():
    sys.modules.pop("pdf_research_assistant.cli", None)
    module = importlib.import_module("pdf_research_assistant.cli")
    yield module
    sys.modules.pop("pdf_research_assistant.cli", None)


def test_main_returns_zero_when_user_quits_immediately(query_papers_module):
    # Mock input so the CLI loop follows the immediate-exit branch.
    with mock.patch("builtins.input", return_value="quit"):
        exit_code = query_papers_module.main()

    assert exit_code == 0


def test_main_prints_subprocess_error_message_when_helper_fails_without_stdout(query_papers_module):
    result = SimpleNamespace(returncode=1, stdout="   ", stderr="helper failed")

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can drive the non-zero-returncode branch without spawning a helper.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert on the CLI error output.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nQuery failed."), mock.call("helper failed")]


def test_main_prints_generic_failure_only_when_helper_fails_without_stdout_or_stderr(query_papers_module):
    result = SimpleNamespace(returncode=1, stdout="   ", stderr="   ")

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can drive the empty-error-output branch.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert that only the generic failure line is shown.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nQuery failed.")]


def test_main_prints_stderr_when_stdout_is_not_json_and_stderr_is_present(query_papers_module):
    result = SimpleNamespace(returncode=0, stdout="not json", stderr="better error")

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can reach the JSON decode error branch.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert that stderr takes precedence after decode failure.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nQuery failed."), mock.call("better error")]


def test_main_prints_stdout_when_stdout_is_not_json_and_stderr_is_empty(query_papers_module):
    result = SimpleNamespace(returncode=0, stdout="plain text failure", stderr="   ")

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can reach the stdout-fallback branch after decode failure.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert on the displayed fallback output.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nQuery failed."), mock.call("plain text failure")]


def test_main_prints_generic_failure_when_stdout_is_empty_json_decode_fails_and_stderr_is_empty(query_papers_module):
    result = SimpleNamespace(returncode=0, stdout="", stderr="   ")

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can reach the decode-error branch with no fallback message text.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert that only the generic failure line is shown.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nQuery failed.")]


def test_main_prints_answer_when_payload_is_successful(query_papers_module):
    result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": True, "answer": "Answer text"}),
        stderr="",
    )

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can drive the successful payload branch.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert on the displayed answer.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nAnswer text")]


def test_main_prints_answer_when_nonzero_returncode_still_includes_successful_json_payload(query_papers_module):
    result = SimpleNamespace(
        returncode=1,
        stdout=json.dumps({"ok": True, "answer": "Recovered answer"}),
        stderr="helper complained",
    )

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can show that stdout payload wins when it is present and valid JSON.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert on the displayed recovered answer.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nRecovered answer")]


def test_main_prints_payload_error_type_and_message_when_payload_is_not_ok(query_papers_module):
    result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": False, "error_type": "ValueError", "error": "bad payload"}),
        stderr="",
    )

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can drive the application-level error branch.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert on the rendered error summary.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nValueError: bad payload")]


def test_main_defaults_error_type_and_message_when_error_payload_fields_are_missing(query_papers_module):
    result = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": False}),
        stderr="",
    )

    # Mock input so the loop processes one question and then exits cleanly.
    with mock.patch("builtins.input", side_effect=["question", "quit"]):
        # Mock subprocess.run so the test can drive the fallback error formatting branch.
        with mock.patch.object(query_papers_module.subprocess, "run", return_value=result):
            # Mock print so the test can assert on the default error summary.
            with mock.patch("builtins.print") as print_mock:
                exit_code = query_papers_module.main()

    assert exit_code == 0
    assert print_mock.call_args_list == [mock.call("\nError: Unknown error")]


def _plan(**overrides):
    values = {
        "paper_count": 2,
        "status_counts": {"loadable": 2},
        "chunk_count": 5,
        "cost_usd": {"low": 0.01, "typical": 0.02, "high": 0.03},
        "estimated_minutes": 1.0,
    }
    return values | overrides


def _printed(print_mock):
    return "\n".join(str(call.args[0]) if call.args else "" for call in print_mock.call_args_list)


def _run_exhaustive_cli(module, inputs, plan=None, summary=None, plan_error=None, poll_values=(0,), finished=(2,)):
    client = module.exhaustive_client
    process = mock.Mock()
    process.poll.side_effect = list(poll_values)
    plan_mock = mock.Mock(side_effect=plan_error) if plan_error else mock.Mock(return_value=plan or _plan())
    with mock.patch.object(sys, "argv", ["pdf-research", "--exhaustive"]), \
            mock.patch("builtins.input", side_effect=inputs), \
            mock.patch.object(client, "request_plan", plan_mock), \
            mock.patch.object(client, "new_run_dir", return_value=Path("run-dir")), \
            mock.patch.object(client, "start_run", return_value=process) as start_run, \
            mock.patch.object(client, "count_finished_papers", side_effect=list(finished)), \
            mock.patch.object(client, "read_summary", return_value=summary), \
            mock.patch.object(client, "describe_crash", return_value="It crashed."), \
            mock.patch.object(module.subprocess, "run") as query_run, \
            mock.patch.object(module.time, "sleep"), \
            mock.patch("builtins.print") as print_mock:
        exit_code = module.main()
    return SimpleNamespace(exit_code=exit_code, start_run=start_run, query_run=query_run, output=_printed(print_mock))


def _finished_summary():
    return {
        "paper_count": 2,
        "status_counts": {"relevant": 1},
        "failed_chunks": 0,
        "cost_usd": 0.5,
        "cost_may_be_understated": False,
        "runtime_seconds": 60,
        "stopped_early": False,
        "stop_reason": None,
    }


def test_main_uses_normal_query_mode_by_default(query_papers_module):
    result = SimpleNamespace(returncode=0, stdout='{"ok": true, "answer": "Answer"}', stderr="")

    with mock.patch.object(sys, "argv", ["pdf-research"]), \
            mock.patch("builtins.input", side_effect=["question", "quit"]), \
            mock.patch.object(query_papers_module.subprocess, "run", return_value=result) as run, \
            mock.patch.object(query_papers_module.exhaustive_client, "request_plan") as request_plan, \
            mock.patch("builtins.print"):
        query_papers_module.main()

    run.assert_called_once()
    request_plan.assert_not_called()


def test_exhaustive_shows_the_plan_before_asking_to_confirm(query_papers_module):
    outcome = _run_exhaustive_cli(query_papers_module, ["q", "no", "quit"])

    assert "Papers in scope: 2" in outcome.output
    assert "Estimated cost: $0.01 to $0.03, typically $0.02" in outcome.output


@pytest.mark.parametrize("answer", ["no", "", "y", "maybe"])
def test_exhaustive_makes_no_run_unless_the_user_types_yes(query_papers_module, answer):
    outcome = _run_exhaustive_cli(query_papers_module, ["q", answer, "quit"])

    outcome.start_run.assert_not_called()
    assert "No API calls were made" in outcome.output


def test_exhaustive_runs_after_yes_and_reports_actual_cost(query_papers_module):
    outcome = _run_exhaustive_cli(query_papers_module, ["q", "YES", "quit"], summary=_finished_summary())

    outcome.start_run.assert_called_once_with("q", Path("run-dir"))
    outcome.query_run.assert_not_called()
    assert "Actual cost: $0.5000" in outcome.output
    assert outcome.exit_code == 0


def test_exhaustive_shows_progress_only_when_it_changes(query_papers_module):
    outcome = _run_exhaustive_cli(
        query_papers_module,
        ["q", "yes", "quit"],
        summary=_finished_summary(),
        poll_values=(None, None, None, 0),
        finished=(0, 1, 1),
    )

    assert outcome.output.count("1 of 2 papers finished") == 1
    assert outcome.output.count("0 of 2 papers finished") == 1


def test_exhaustive_refuses_to_run_when_no_paper_can_be_searched(query_papers_module):
    plan = _plan(status_counts={"not_indexed": 2}, chunk_count=0)

    outcome = _run_exhaustive_cli(query_papers_module, ["q", "quit"], plan=plan)

    outcome.start_run.assert_not_called()
    assert "pdf-research-rebuild" in outcome.output


def test_exhaustive_reports_a_planning_failure_and_keeps_going(query_papers_module):
    outcome = _run_exhaustive_cli(query_papers_module, ["q", "quit"], plan_error=RuntimeError("bad index"))

    outcome.start_run.assert_not_called()
    assert "bad index" in outcome.output
    assert outcome.exit_code == 0


def test_exhaustive_says_so_when_the_run_crashes(query_papers_module):
    outcome = _run_exhaustive_cli(query_papers_module, ["q", "yes", "quit"], summary=None)

    assert "It crashed." in outcome.output
