"""Shared helpers for starting and following exhaustive runs from the app and CLI.

Runs happen in a fresh helper process, so this module must stay free of PaperQA imports.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "exhaustive_runs"
HELPER = Path(__file__).with_name("exhaustive.py")
NOTHING_TO_SEARCH_MESSAGE = "No papers can be searched yet. Run pdf-research-rebuild first."
COST_NOTE = "The real figure may be slightly higher, because failed requests can be billed."


def request_plan(question: str) -> dict:
    """Ask the helper for the pre-run plan, which makes no API calls."""
    result = subprocess.run(
        [sys.executable, str(HELPER), "plan", question],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or "Exhaustive helper failed.")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Invalid helper response.") from exc
    if not payload.get("ok"):
        raise RuntimeError(f"{payload.get('error_type', 'Error')}: {payload.get('error', 'Unknown error')}")
    return payload["plan"]


def can_run(plan: dict) -> bool:
    """Return whether at least one paper can be searched."""
    return plan["status_counts"].get("loadable", 0) > 0


def format_usd(amount: float) -> str:
    """Format a cost, keeping small amounts from rounding to $0.00."""
    return f"${amount:.2f}" if amount >= 0.01 else f"${amount:.4f}"


def describe_plan(plan: dict) -> list[str]:
    """Describe the run size so the user can decide whether to pay for it."""
    ready = plan["status_counts"].get("loadable", 0)
    lines = [f"Papers in scope: {plan['paper_count']}", f"Ready to search: {ready} ({plan['chunk_count']} chunks)"]
    if ready < plan["paper_count"]:
        lines.append(f"Cannot be searched: {plan['paper_count'] - ready}. These will be listed in the report.")
    cost = plan["cost_usd"]
    if cost:
        low, high, typical = (format_usd(cost[label]) for label in ("low", "high", "typical"))
        lines.append(f"Estimated cost: {low} to {high}, typically {typical}")
    else:
        lines.append("Estimated cost: unknown")
    minutes = max(round(plan["estimated_minutes"]), 1)
    lines.append(f"Estimated time: about {minutes} minute{'' if minutes == 1 else 's'}")
    return lines


def new_run_dir() -> Path:
    """Create a separate output folder for one run."""
    path = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path.mkdir(parents=True)
    return path


def start_run(question: str, output_dir: Path) -> subprocess.Popen:
    """Start a paid run. Callers must get the user's confirmation first."""
    with open(output_dir / "run.log", "w", encoding="utf-8") as log:
        return subprocess.Popen(
            [sys.executable, str(HELPER), "run", question, "--output-dir", str(output_dir), "--confirmed"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def count_finished_papers(output_dir: Path) -> int:
    """Return how many papers the run has saved so far."""
    results = output_dir / "results.jsonl"
    if not results.exists():
        return 0
    return len(results.read_text(encoding="utf-8").splitlines())


def read_saved_cost(output_dir: Path) -> float:
    """Return the cost of the papers saved so far, for a run that left no summary."""
    results = output_dir / "results.jsonl"
    if not results.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in results.read_text(encoding="utf-8").splitlines())


def read_summary(output_dir: Path) -> dict | None:
    """Return the finished run's summary, or None if the run did not complete."""
    try:
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def describe_result(summary: dict, output_dir: Path) -> list[str]:
    """Describe a finished run, leading with anything that makes it incomplete."""
    lines = []
    if summary["stopped_early"]:
        lines.append(f"This run did not finish. {summary['stop_reason']}")
    if summary["failed_chunks"]:
        lines.append(f"{summary['failed_chunks']} chunks could not be checked. See the report.")
    relevant = summary["status_counts"].get("relevant", 0)
    cost_note = f" {COST_NOTE}" if summary["cost_may_be_understated"] else ""
    lines.extend(
        [
            f"{relevant} of {summary['paper_count']} papers have relevant evidence.",
            f"Actual cost: ${summary['cost_usd']:.4f}.{cost_note}",
            f"Runtime: {summary['runtime_seconds'] / 60:.1f} minutes",
            f"Report: {output_dir / 'report.md'}",
        ]
    )
    return lines


def describe_crash(output_dir: Path) -> str:
    """Describe a run that ended without a summary."""
    finished = count_finished_papers(output_dir)
    return f"The exhaustive search stopped unexpectedly. {finished} papers were saved in {output_dir}. See run.log there."
