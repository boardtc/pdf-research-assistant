import json
import subprocess
import sys
from unittest import mock

import pytest

from pdf_research_assistant import query_client

# Kept before the fixture replaces it, for the one test that starts a real (tiny) process.
REAL_POPEN = subprocess.Popen


@pytest.fixture(autouse=True)
def no_spare_process():
    """Start every test with no spare process, and never start a real one."""
    with mock.patch.object(query_client, "_spare", None), mock.patch.object(query_client.subprocess, "Popen") as popen:
        yield popen


def _process(running=True, returncode=0, stdout='{"ok": true}', stderr=""):
    process = mock.Mock()
    process.poll.return_value = None if running else returncode
    process.returncode = returncode
    process.args = ["python", "query_once.py", query_client.WAIT_FLAG]
    process.communicate.return_value = (stdout, stderr)
    return process


def test_start_process_starts_the_query_helper_waiting_for_a_question(no_spare_process):
    query_client.start_process()

    no_spare_process.assert_called_once_with(
        [sys.executable, str(query_client.HELPER), query_client.WAIT_FLAG],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_the_helper_is_the_one_shot_query_script():
    assert query_client.HELPER.name == "query_once.py"
    assert query_client.HELPER.exists()


def test_prepare_starts_a_spare_process_when_there_is_none(no_spare_process):
    query_client.prepare()

    no_spare_process.assert_called_once()
    assert query_client._spare is no_spare_process.return_value


def test_prepare_leaves_a_spare_that_is_still_waiting(no_spare_process):
    waiting = _process()
    query_client._spare = waiting

    query_client.prepare()

    no_spare_process.assert_not_called()
    assert query_client._spare is waiting


def test_prepare_replaces_a_spare_that_has_stopped(no_spare_process):
    query_client._spare = _process(running=False, returncode=1)

    query_client.prepare()

    no_spare_process.assert_called_once()
    assert query_client._spare is no_spare_process.return_value


def test_run_hands_the_question_to_the_spare_and_leaves_no_spare_behind(no_spare_process):
    spare = _process(stdout='{"ok": true, "answer": "A"}', stderr="log")
    query_client._spare = spare

    result = query_client.run("What is new?", {"evidence_k": 3})

    no_spare_process.assert_not_called()
    spare.communicate.assert_called_once_with(
        json.dumps({"question": "What is new?", "one_off": {"evidence_k": 3}}) + "\n"
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, '{"ok": true, "answer": "A"}', "log")
    assert result.args == spare.args
    # Each process answers one question, so the next question needs a new one.
    assert query_client._spare is None


def test_run_sends_no_one_off_settings_when_there_are_none(no_spare_process):
    spare = _process()
    query_client._spare = spare

    query_client.run("What is new?")

    spare.communicate.assert_called_once_with(json.dumps({"question": "What is new?", "one_off": None}) + "\n")


def test_run_starts_a_process_when_there_is_no_spare(no_spare_process):
    started = _process(stdout="{}")
    no_spare_process.return_value = started

    result = query_client.run("What is new?")

    no_spare_process.assert_called_once()
    started.communicate.assert_called_once()
    assert result.stdout == "{}"


def test_run_starts_a_new_process_when_the_spare_has_stopped(no_spare_process):
    stopped = _process(running=False, returncode=1)
    started = _process(stdout="{}")
    query_client._spare = stopped
    no_spare_process.return_value = started

    result = query_client.run("What is new?")

    stopped.communicate.assert_not_called()
    started.communicate.assert_called_once()
    assert result.stdout == "{}"


def test_run_passes_on_a_failed_question_unchanged(no_spare_process):
    query_client._spare = _process(returncode=1, stdout="", stderr="Traceback ...")

    result = query_client.run("What is new?")

    assert (result.returncode, result.stdout, result.stderr) == (1, "", "Traceback ...")


ECHO_HELPER = """
import json, sys
request = json.loads(sys.stdin.readline())
print(json.dumps({"ok": True, "answer": request["question"], "one_off": request["one_off"]}))
"""


def test_a_real_spare_process_answers_the_question_it_is_sent(no_spare_process, tmp_path):
    helper = tmp_path / "echo_helper.py"
    helper.write_text(ECHO_HELPER, encoding="utf-8")
    no_spare_process.side_effect = REAL_POPEN

    with mock.patch.object(query_client, "HELPER", helper):
        query_client.prepare()
        result = query_client.run("Why is the sky blue – and what’s Rayleigh?", {"evidence_k": 3})

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "ok": True,
        "answer": "Why is the sky blue – and what’s Rayleigh?",
        "one_off": {"evidence_k": 3},
    }
