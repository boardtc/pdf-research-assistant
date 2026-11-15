"""Starting and following an index rebuild from the app.

The rebuild runs rebuild.py in its own process, so it builds its settings afresh, saved sidebar choices
included, exactly as `pdf-research-rebuild` does from a terminal. This module must stay free of PaperQA imports.

One index is shared by everyone using the app, so the running rebuild is kept here, in the server process,
and not in one browser session. A reloaded page or a second browser tab then still sees it, and cannot start
a second rebuild beside it.
"""

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

HELPER = Path(__file__).with_name("rebuild.py")
LOG_NAME = "rebuild.log"
LOG_TAIL_LINES = 30
PDF_SUFFIX = ".pdf"

_state: dict = {"run": None, "result": None}
_start_lock = threading.Lock()


def count_pdfs(paper_dir: Path) -> int:
    """Return how many PDFs sit under a folder, subfolders included."""
    return sum(1 for path in Path(paper_dir).rglob("*") if path.suffix.lower() == PDF_SUFFIX)


def log_path(index_dir: Path) -> Path:
    """Return where a rebuild writes its output: beside the index folders, replaced by each rebuild."""
    return Path(index_dir).parent / LOG_NAME


def get_run() -> dict | None:
    """Return the rebuild that is running, or None."""
    return _state["run"]


def start(index_dir: Path, pdf_count: int) -> dict:
    """Start a rebuild, which costs money. Callers must get the user's confirmation first.

    Only one rebuild runs at a time: while one is running it is returned and nothing new starts.
    """
    with _start_lock:
        if _state["run"] is not None:
            return _state["run"]
        log = log_path(index_dir)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                # Unbuffered, so the log can be read while the rebuild runs.
                [sys.executable, "-u", str(HELPER)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                # File names in the output may hold characters the console encoding cannot write.
                env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            )
        _state["result"] = None
        _state["run"] = {
            "process": process,
            "index_dir": Path(index_dir),
            "log_path": log,
            "pdf_count": pdf_count,
            "done": 0,
            "started": datetime.now(),
        }
        return _state["run"]


def note_progress(run: dict, done: int) -> int:
    """Record how many PDFs the index now accounts for, and return the highest count seen so far.

    The index's file list cannot be read at the moment the rebuild is writing it, which looks like 0.
    """
    run["done"] = min(max(run["done"], done), run["pdf_count"])
    return run["done"]


def progress_fraction(run: dict, done: int) -> float:
    """Return the share of PDFs done, from 0 to 1."""
    return done / run["pdf_count"] if run["pdf_count"] else 0.0


def describe_progress(run: dict, done: int) -> str:
    return f"{done} of {run['pdf_count']} PDFs done · started at {run['started']:%H:%M}"


def read_log_tail(path: Path) -> list[str]:
    """Return the last lines a rebuild wrote, or none when there is no log to read."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-LOG_TAIL_LINES:]


def collect_result() -> dict | None:
    """End the running rebuild if its process has finished, and return its result; otherwise None."""
    run = _state["run"]
    if run is None or run["process"].poll() is None:
        return None
    _state["run"] = None
    _state["result"] = {
        "ok": run["process"].poll() == 0,
        "log_path": str(run["log_path"]),
        "lines": read_log_tail(run["log_path"]),
    }
    return _state["result"]


def take_result() -> dict | None:
    """Return the last finished rebuild's result once, so it is shown once."""
    result, _state["result"] = _state["result"], None
    return result
