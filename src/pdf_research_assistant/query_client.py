"""Hands questions to a query process that has already loaded PaperQA.

Each question still runs in a process of its own (query_once.py), so nothing carries over from one question to
the next, and settings saved in the sidebar are read when the question arrives. What changes is when that
process starts: the app and the CLI start the next one ahead of time, and it loads PaperQA and LiteLLM (about
10 seconds) while it waits for a question on stdin. If the spare process has stopped, a new one is started for
the question instead, which is as slow as before but otherwise the same.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

HELPER = Path(__file__).with_name("query_once.py")
# Tells query_once.py to read its question from stdin instead of the command line.
WAIT_FLAG = "--wait"

# Several browser sessions share one app process, so only one of them may take the spare at a time.
_lock = threading.Lock()
_spare: subprocess.Popen | None = None


def start_process() -> subprocess.Popen:
    """Start a query process that loads PaperQA and then waits for one question."""
    return subprocess.Popen(
        [sys.executable, str(HELPER), WAIT_FLAG],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def prepare() -> None:
    """Start a spare query process unless one is already waiting, so the next question starts quickly."""
    global _spare
    with _lock:
        if _spare is None or _spare.poll() is not None:
            _spare = start_process()


def take_process() -> subprocess.Popen:
    """Take the spare query process for a question, or start one if there is none still waiting."""
    global _spare
    with _lock:
        process, _spare = _spare, None
    if process is None or process.poll() is not None:
        return start_process()
    return process


def run(question: str, one_off: dict | None = None) -> subprocess.CompletedProcess:
    """Ask one question, with any settings changed for this question only, and wait for the process to finish.

    The result holds the process's exit code and output, just as running query_once.py directly would.
    """
    process = take_process()
    request = json.dumps({"question": question, "one_off": one_off})
    stdout, stderr = process.communicate(request + "\n")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
