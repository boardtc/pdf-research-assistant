import importlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _plan(**overrides):
    values = {
        "paper_count": 3,
        "status_counts": {"loadable": 2, "not_indexed": 1},
        "chunk_count": 40,
        "cost_usd": {"low": 0.01, "typical": 0.02, "high": 0.03},
        "estimated_minutes": 4.2,
    }
    return values | overrides


def _summary(**overrides):
    values = {
        "paper_count": 3,
        "status_counts": {"relevant": 2},
        "failed_chunks": 0,
        "cost_usd": 0.1234,
        "cost_may_be_understated": False,
        "runtime_seconds": 180,
        "stopped_early": False,
        "stop_reason": None,
    }
    return values | overrides


@pytest.fixture
def client():
    sys.modules.pop("pdf_research_assistant.exhaustive_client", None)
    module = importlib.import_module("pdf_research_assistant.exhaustive_client")
    yield module
    sys.modules.pop("pdf_research_assistant.exhaustive_client", None)


@pytest.fixture
def workspace_tmp_path():
    # Create a unique per-test directory under tests/.tmp/ so filesystem tests stay isolated.
    root = PROJECT_ROOT / "tests" / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(uuid.uuid4())
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_importing_the_client_does_not_load_paperqa():
    code = (
        "import sys; import pdf_research_assistant.exhaustive_client;"
        "sys.exit(1 if 'paperqa' in sys.modules else 0)"
    )

    result = __import__("subprocess").run(
        [sys.executable, "-c", code],
        env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        check=False,
    )

    assert result.returncode == 0


def test_request_plan_returns_the_plan_from_the_helper(client):
    result = SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "plan": _plan()}), stderr="")

    with mock.patch.object(client.subprocess, "run", return_value=result) as run:
        plan = client.request_plan("q")

    assert plan == _plan()
    command = run.call_args.args[0]
    assert command[1:] == [str(client.HELPER), "plan", "q"]


def test_request_plan_raises_stderr_when_helper_prints_nothing(client):
    result = SimpleNamespace(returncode=1, stdout="  ", stderr="helper failed")

    with mock.patch.object(client.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="helper failed"):
            client.request_plan("q")


def test_request_plan_raises_when_helper_output_is_not_json(client):
    result = SimpleNamespace(returncode=0, stdout="not json", stderr="")

    with mock.patch.object(client.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="not json"):
            client.request_plan("q")


def test_request_plan_raises_the_helper_error_message(client):
    payload = {"ok": False, "error_type": "ValueError", "error": "bad index"}
    result = SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr="")

    with mock.patch.object(client.subprocess, "run", return_value=result):
        with pytest.raises(RuntimeError, match="ValueError: bad index"):
            client.request_plan("q")


def test_can_run_is_true_when_a_paper_is_loadable(client):
    assert client.can_run(_plan()) is True


def test_can_run_is_false_when_no_paper_is_loadable(client):
    assert client.can_run(_plan(status_counts={"not_indexed": 3})) is False


def test_describe_plan_shows_counts_cost_and_time(client):
    text = "\n".join(client.describe_plan(_plan()))

    assert "Papers in scope: 3" in text
    assert "Ready to search: 2 (40 chunks)" in text
    assert "Cannot be searched: 1" in text
    assert "Estimated cost: $0.01 to $0.03, typically $0.02" in text
    assert "Estimated time: about 4 minutes" in text


def test_describe_plan_shows_small_costs_instead_of_rounding_to_zero(client):
    plan = _plan(cost_usd={"low": 0.0021, "typical": 0.0028, "high": 0.0035})

    assert "Estimated cost: $0.0021 to $0.0035, typically $0.0028" in client.describe_plan(plan)


def test_describe_plan_omits_the_unsearchable_line_when_every_paper_is_ready(client):
    plan = _plan(status_counts={"loadable": 3})

    assert not any("Cannot be searched" in line for line in client.describe_plan(plan))


def test_describe_plan_says_when_cost_is_unknown(client):
    assert "Estimated cost: unknown" in client.describe_plan(_plan(cost_usd=None))


def test_describe_plan_rounds_short_runs_up_to_a_minute(client):
    assert "Estimated time: about 1 minute" in client.describe_plan(_plan(estimated_minutes=0.2))


def test_new_run_dir_creates_a_separate_folder_each_time(client, workspace_tmp_path):
    with mock.patch.object(client, "RUNS_DIR", workspace_tmp_path / "runs"):
        first, second = client.new_run_dir(), client.new_run_dir()

    assert first != second
    assert first.is_dir() and second.is_dir()


def test_start_run_confirms_the_helper_and_logs_to_the_run_folder(client, workspace_tmp_path):
    with mock.patch.object(client.subprocess, "Popen") as popen:
        process = client.start_run("q", workspace_tmp_path)

    assert process is popen.return_value
    command = popen.call_args.args[0]
    assert command[1:] == [str(client.HELPER), "run", "q", "--output-dir", str(workspace_tmp_path), "--confirmed"]
    assert (workspace_tmp_path / "run.log").exists()


def test_count_finished_papers_is_zero_before_any_result(client, workspace_tmp_path):
    assert client.count_finished_papers(workspace_tmp_path) == 0


def test_count_finished_papers_counts_saved_results(client, workspace_tmp_path):
    (workspace_tmp_path / "results.jsonl").write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

    assert client.count_finished_papers(workspace_tmp_path) == 2


def test_read_summary_returns_saved_summary(client, workspace_tmp_path):
    (workspace_tmp_path / "summary.json").write_text('{"cost_usd": 1}', encoding="utf-8")

    assert client.read_summary(workspace_tmp_path) == {"cost_usd": 1}


def test_read_summary_is_none_when_missing(client, workspace_tmp_path):
    assert client.read_summary(workspace_tmp_path) is None


def test_read_summary_is_none_when_corrupt(client, workspace_tmp_path):
    (workspace_tmp_path / "summary.json").write_text("{", encoding="utf-8")

    assert client.read_summary(workspace_tmp_path) is None


def test_describe_result_shows_outcome_cost_time_and_report(client, workspace_tmp_path):
    text = "\n".join(client.describe_result(_summary(), workspace_tmp_path))

    assert "2 of 3 papers have relevant evidence" in text
    assert "Actual cost: $0.1234" in text
    assert "Runtime: 3.0 minutes" in text
    assert str(workspace_tmp_path / "report.md") in text


def test_describe_result_leads_with_an_early_stop(client, workspace_tmp_path):
    lines = client.describe_result(_summary(stopped_early=True, stop_reason="Stopped."), workspace_tmp_path)

    assert lines[0] == "This run did not finish. Stopped."


def test_describe_result_warns_about_failed_chunks(client, workspace_tmp_path):
    lines = client.describe_result(_summary(failed_chunks=2), workspace_tmp_path)

    assert lines[0] == "2 chunks could not be checked. See the report."


def test_describe_result_notes_when_cost_may_be_low(client, workspace_tmp_path):
    text = "\n".join(client.describe_result(_summary(cost_may_be_understated=True), workspace_tmp_path))

    assert "may be slightly higher" in text


def test_describe_result_handles_no_relevant_papers(client, workspace_tmp_path):
    text = "\n".join(client.describe_result(_summary(status_counts={}), workspace_tmp_path))

    assert "0 of 3 papers have relevant evidence" in text


def test_read_saved_cost_adds_up_finished_papers(client, workspace_tmp_path):
    (workspace_tmp_path / "results.jsonl").write_text('{"cost_usd": 0.25}\n{"cost_usd": 0.5}\n', encoding="utf-8")

    assert client.read_saved_cost(workspace_tmp_path) == 0.75


def test_read_saved_cost_is_zero_before_any_result(client, workspace_tmp_path):
    assert client.read_saved_cost(workspace_tmp_path) == 0.0


def test_describe_crash_says_what_was_saved_and_where(client, workspace_tmp_path):
    (workspace_tmp_path / "results.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

    text = client.describe_crash(workspace_tmp_path)

    assert "stopped unexpectedly" in text
    assert "1 papers were saved" in text
    assert str(workspace_tmp_path) in text
