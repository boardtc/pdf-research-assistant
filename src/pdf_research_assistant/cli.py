import json
import subprocess
import sys
import time
from pathlib import Path

from pdf_research_assistant import exhaustive_client

PROGRESS_POLL_SECONDS = 2


def run_exhaustive_question(question: str) -> None:
    """Show the plan, get confirmation, then follow a paid run to the end."""
    try:
        plan = exhaustive_client.request_plan(question)
    except RuntimeError as exc:
        print(f"\nCould not plan the search: {exc}")
        return
    print()
    for line in exhaustive_client.describe_plan(plan):
        print(line)
    if not exhaustive_client.can_run(plan):
        print(exhaustive_client.NOTHING_TO_SEARCH_MESSAGE)
        return
    if input("\nType yes to run this paid search: ").strip().lower() != "yes":
        print("Cancelled. No API calls were made.")
        return

    output_dir = exhaustive_client.new_run_dir()
    process = exhaustive_client.start_run(question, output_dir)
    shown = None
    while process.poll() is None:
        finished = exhaustive_client.count_finished_papers(output_dir)
        if finished != shown:
            print(f"{finished} of {plan['paper_count']} papers finished")
            shown = finished
        time.sleep(PROGRESS_POLL_SECONDS)

    summary = exhaustive_client.read_summary(output_dir)
    if summary is None:
        print(f"\n{exhaustive_client.describe_crash(output_dir)}")
        return
    print()
    for line in exhaustive_client.describe_result(summary, output_dir):
        print(line)


def main() -> int:
    """Run the interactive CLI loop that forwards questions to the one-shot query helper."""
    exhaustive_mode = "--exhaustive" in sys.argv[1:]
    while True:
        q = input("\nQuestion (or quit): ").strip()
        if q.lower() == "quit":
            return 0
        if exhaustive_mode:
            run_exhaustive_question(q)
            continue
        helper = Path(__file__).with_name("query_once.py")
        result = subprocess.run(
            [sys.executable, str(helper), q],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 and not result.stdout.strip():
            print("\nQuery failed.")
            if result.stderr.strip():
                print(result.stderr.strip())
            continue
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            print("\nQuery failed.")
            if result.stderr.strip():
                print(result.stderr.strip())
            elif result.stdout.strip():
                print(result.stdout.strip())
            continue
        if payload.get("ok"):
            print("\n" + payload["answer"])
        else:
            print(f"\n{payload.get('error_type', 'Error')}: {payload.get('error', 'Unknown error')}")


if __name__ == "__main__":
    raise SystemExit(main())

