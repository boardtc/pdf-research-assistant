"""Starting and following an index rebuild from the app.

The rebuild runs rebuild.py in its own process, so it builds its settings afresh, saved sidebar choices
included, exactly as `pdf-research-rebuild` does from a terminal. This module must stay free of PaperQA imports.

One index is shared by everyone using the app, so the running rebuild is kept here, in the server process,
and not in one browser session. A reloaded page or a second browser tab then still sees it, and cannot start
a second rebuild beside it.
"""

import json
import os
import pickle
import subprocess
import sys
import threading
import zlib
from datetime import datetime
from pathlib import Path

from pdf_research_assistant.costs import format_cost

HELPER = Path(__file__).with_name("rebuild.py")
LOG_NAME = "rebuild.log"
LOG_TAIL_LINES = 30
PDF_SUFFIX = ".pdf"
STOP_NAME = "rebuild.stop"
# The rebuild process looks for this file between PDFs; rebuild.py reads the same names.
STOP_ENV = "PDF_RESEARCH_STOP_FILE"
STOPPED_EXIT_CODE = 3
# The rebuild writes what it really cost to this file, by kind of call; rebuild.py reads the same names.
COST_NAME = "rebuild.cost.json"
COST_ENV = "PDF_RESEARCH_COST_FILE"
# Embeddings of the passages, descriptions of images and tables, and reading the citation of a PDF the paper
# list does not name.
COST_KINDS = ("embeddings", "enrichment", "citations")
# PaperQA's marker in files.zip for a PDF it could not index (bootstrap.py repeats it for the app).
FAILED_DOCUMENT_ADD_ID = "ERROR"

_state: dict = {"run": None, "result": None}
_start_lock = threading.Lock()


def count_pdfs(paper_dir: Path, recurse: bool = True) -> int:
    """Return how many PDFs sit in a folder, in its subfolders too unless recurse is off."""
    paths = Path(paper_dir).rglob("*") if recurse else Path(paper_dir).glob("*")
    return sum(1 for path in paths if path.suffix.lower() == PDF_SUFFIX)


def log_path(index_dir: Path) -> Path:
    """Return where a rebuild writes its output: beside the index folders, replaced by each rebuild."""
    return Path(index_dir).parent / LOG_NAME


def stop_path(index_dir: Path) -> Path:
    """Return the file whose presence asks a running rebuild to stop: beside the log."""
    return Path(index_dir).parent / STOP_NAME


def cost_path(index_dir: Path) -> Path:
    """Return where a rebuild writes what it cost: beside the log, replaced by each rebuild."""
    return Path(index_dir).parent / COST_NAME


def empty_costs() -> dict[str, float]:
    return dict.fromkeys(COST_KINDS, 0.0)


def read_cost(path: Path) -> dict[str, float] | None:
    """Return what a rebuild cost by kind of call, or None when it wrote no usable record, as after a crash."""
    try:
        costs = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(costs, dict) or not all(isinstance(costs.get(kind), int | float) for kind in COST_KINDS):
        return None
    return {kind: float(costs[kind]) for kind in COST_KINDS}


def describe_cost(costs: dict[str, float]) -> str:
    """Say what a rebuild really cost, in all and for embeddings and enrichment, with citations when they cost anything."""
    parts = [f"embeddings {format_cost(costs['embeddings'])}", f"describing images and tables {format_cost(costs['enrichment'])}"]
    if costs["citations"]:
        parts.append(f"reading citations {format_cost(costs['citations'])}")
    return f"Real cost: {format_cost(sum(costs.values()))} in all: {', '.join(parts)}."


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
        # A request left by an earlier rebuild must not stop this one before it begins.
        stop_path(index_dir).unlink(missing_ok=True)
        cost_path(index_dir).unlink(missing_ok=True)
        with open(log, "w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                # Unbuffered, so the log can be read while the rebuild runs.
                [sys.executable, "-u", str(HELPER)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                # File names in the output may hold characters the console encoding cannot write.
                env=os.environ
                | {
                    "PYTHONIOENCODING": "utf-8",
                    STOP_ENV: str(stop_path(index_dir)),
                    COST_ENV: str(cost_path(index_dir)),
                },
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


def request_stop(run: dict) -> None:
    """Ask a running rebuild to stop at its next safe point: no new PDF is started, those under way are saved.

    Nothing is killed, so the index is never cut off in the middle of a write.
    """
    stop_path(run["index_dir"]).write_text("stop", encoding="utf-8")
    run["stopping"] = True


def is_stopping(run: dict) -> bool:
    return bool(run.get("stopping"))


def read_index_state(index_dir: Path) -> dict:
    """Say whether PaperQA's file list in an index can be read, and how many PDFs it holds.

    PaperQA rewrites files.zip in place, so a rebuild cut off at the wrong moment could leave it half written.
    """
    file_list = Path(index_dir) / "files.zip"
    if not file_list.exists():
        # Nothing was saved yet, so there is nothing to be damaged.
        return {"usable": True, "indexed": 0}
    try:
        files = pickle.loads(zlib.decompress(file_list.read_bytes()))
    except Exception:
        return {"usable": False, "indexed": 0}
    if not isinstance(files, dict):
        return {"usable": False, "indexed": 0}
    return {"usable": True, "indexed": sum(1 for status in files.values() if status != FAILED_DOCUMENT_ADD_ID)}


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
    stop_path(run["index_dir"]).unlink(missing_ok=True)
    returncode = run["process"].poll()
    _state["result"] = {
        "ok": returncode == 0,
        "stopped": returncode == STOPPED_EXIT_CODE,
        "log_path": str(run["log_path"]),
        "lines": read_log_tail(run["log_path"]),
        "index_dir": run["index_dir"],
        "index": read_index_state(run["index_dir"]),
        "cost": read_cost(cost_path(run["index_dir"])),
    }
    return _state["result"]


def take_result() -> dict | None:
    """Return the last finished rebuild's result once, so it is shown once."""
    result, _state["result"] = _state["result"], None
    return result
