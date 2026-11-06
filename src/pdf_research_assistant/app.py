import json
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import streamlit as st

from pdf_research_assistant import bootstrap, exhaustive_client

os.environ.setdefault("PQA_INDEX_DONT_CACHE_INDEXES", "true")

QUERY_MODE = "Query"
EXHAUSTIVE_MODE = "Exhaustive"
PROGRESS_POLL_SECONDS = 2


def flatten_exception_messages(exc, depth=0):
    indent = "  " * depth
    lines = [f"{indent}{type(exc).__name__}: {exc}"]
    children = getattr(exc, "exceptions", None)
    if children:
        for child in children:
            lines.extend(flatten_exception_messages(child, depth + 1))
    return lines


def copy_to_clipboard(text: str) -> None:
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Set-Clipboard -Value ([Console]::In.ReadToEnd())",
        ],
        input=text,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def render_copy_button(text: str, key: str) -> None:
    if st.button("Copy answer", key=key):
        try:
            copy_to_clipboard(text)
        except Exception as exc:
            st.error(f"Copy failed: {exc}")
        else:
            st.toast("Answer copied")


def render_source_passages(contexts):
    if not contexts:
        return
    with st.expander("Show source passages"):
        for ctx in contexts:
            st.subheader(ctx.get("name", "Source"))
            st.write(ctx.get("context", ""))
            raw_text = ctx.get("raw_text", "")
            if raw_text:
                st.caption("Raw chunk text")
                st.write(raw_text)


def run_query_subprocess(question: str) -> tuple[str, float, list[dict]]:
    helper = Path(__file__).with_name("query_once.py")
    result = subprocess.run(
        [sys.executable, str(helper), question],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or "Query subprocess failed.")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Invalid query response.") from exc
    if not payload.get("ok"):
        raise RuntimeError(f"{payload.get('error_type', 'Error')}: {payload.get('error', 'Unknown error')}")
    return payload["answer"], payload["cost"], payload.get("contexts", [])


@st.cache_resource
def get_persistent_app_state() -> dict:
    return {
        "history": [],
        "total_cost": 0.0,
        "query_count": 0,
    }


def sync_persistent_state(app_state: dict) -> None:
    app_state["history"] = deepcopy(st.session_state.history)
    app_state["total_cost"] = st.session_state.total_cost
    app_state["query_count"] = st.session_state.query_count


def escape_dollars(text: str) -> str:
    """Stop Streamlit Markdown reading text between two dollar signs as a maths formula."""
    return text.replace("$", r"\$")


def render_exhaustive_report(report_path: str) -> None:
    path = Path(report_path)
    if not path.exists():
        st.warning(f"Report not found: {path}")
        return
    report = path.read_text(encoding="utf-8")
    st.download_button("Download report", report, file_name="exhaustive-report.md", key=f"report_{path.parent.name}")
    with st.expander("Show full report"):
        # The report escapes text from PDFs and the model; only its own collapsible sections are HTML.
        st.markdown(escape_dollars(report), unsafe_allow_html=True)


def plan_exhaustive_question(question: str) -> None:
    with st.spinner("Checking which papers can be searched..."):
        try:
            plan = exhaustive_client.request_plan(question)
        except RuntimeError as exc:
            st.error(f"Could not plan the search: {exc}")
            return
    st.session_state.exhaustive_pending = {"question": question, "plan": plan}
    st.rerun()


def start_exhaustive_run(pending: dict, app_state: dict) -> None:
    output_dir = exhaustive_client.new_run_dir()
    process = exhaustive_client.start_run(pending["question"], output_dir)
    st.session_state.exhaustive_run = {
        "question": pending["question"],
        "output_dir": str(output_dir),
        "process": process,
        "paper_count": pending["plan"]["paper_count"],
    }
    st.session_state.exhaustive_pending = None
    st.session_state.history.append({"role": "user", "content": pending["question"]})
    sync_persistent_state(app_state)
    st.rerun()


def render_exhaustive_pending(pending: dict, app_state: dict) -> None:
    plan = pending["plan"]
    with st.chat_message("assistant"):
        st.markdown(escape_dollars(f"**Exhaustive search:** {pending['question']}"))
        st.markdown(escape_dollars("\n".join(f"- {line}" for line in exhaustive_client.describe_plan(plan))))
        if not exhaustive_client.can_run(plan):
            st.error(exhaustive_client.NOTHING_TO_SEARCH_MESSAGE)
            if st.button("Dismiss", key="dismiss_exhaustive"):
                st.session_state.exhaustive_pending = None
                st.rerun()
            return
        if st.button("Run paid search", key="confirm_exhaustive"):
            start_exhaustive_run(pending, app_state)
            return
        if st.button("Cancel", key="cancel_exhaustive"):
            st.session_state.exhaustive_pending = None
            st.rerun()


def finish_exhaustive_run(run: dict, app_state: dict) -> None:
    output_dir = Path(run["output_dir"])
    summary = exhaustive_client.read_summary(output_dir)
    item = {"role": "assistant", "id": len(st.session_state.history), "contexts": []}
    if summary is None:
        item |= {"content": exhaustive_client.describe_crash(output_dir), "cost": exhaustive_client.read_saved_cost(output_dir)}
    else:
        item |= {
            "content": escape_dollars("\n\n".join(exhaustive_client.describe_result(summary, output_dir))),
            "cost": summary["cost_usd"],
            "report_path": str(output_dir / "report.md"),
        }
    st.session_state.history.append(item)
    st.session_state.total_cost += item["cost"]
    st.session_state.query_count += 1
    st.session_state.exhaustive_run = None
    sync_persistent_state(app_state)
    st.rerun()


@st.fragment(run_every=PROGRESS_POLL_SECONDS)
def render_exhaustive_run(run: dict, app_state: dict) -> None:
    # A fragment refreshes alone, so the page finishes and old content such as the Run button clears.
    if run["process"].poll() is not None:
        finish_exhaustive_run(run, app_state)
        return
    finished = exhaustive_client.count_finished_papers(Path(run["output_dir"]))
    total = run["paper_count"]
    with st.chat_message("assistant"):
        st.progress(min(finished / max(total, 1), 1.0), text=f"{finished} of {total} papers finished")


st.title("PDF Research Assistant")

allowed_paths = bootstrap.get_allowed_paths()
use_manifest_scope = bool(allowed_paths)
manifest_count = len(allowed_paths)
active_index_dir = bootstrap.get_active_index_dir()
indexed_count = bootstrap.get_indexed_doc_count(index_dir=active_index_dir)
failed_files = bootstrap.get_failed_files(index_dir=active_index_dir)

if indexed_count == 0 and use_manifest_scope:
    st.warning(
        f"Index not found. The first question will rebuild the index for {manifest_count} manifest PDFs, "
        "which may take a long time."
    )
    st.info("A rebuild starts only after you submit a question below.")
elif use_manifest_scope and indexed_count < manifest_count:
    st.warning(
        f"Index is incomplete: {indexed_count} of {manifest_count} manifest PDFs appear indexed. "
        "The next question should continue or trigger indexing work."
    )
elif indexed_count == 0:
    st.warning("Index not found. The first question will build the PDF index, which may take a long time.")
    st.info("No manifest.csv was found, so all PDFs under PAPER_DIR are in scope.")
else:
    if use_manifest_scope:
        st.caption(f"Index available for {indexed_count} manifest PDFs.")
    else:
        st.caption(f"Index available for {indexed_count} PDFs under PAPER_DIR.")

if failed_files:
    st.error(f"{len(failed_files)} PDF(s) failed to index.")
    with st.expander("Show failed PDFs"):
        for file_location in failed_files:
            st.code(file_location)

app_state = get_persistent_app_state()
if "history" not in st.session_state:
    st.session_state.history = deepcopy(app_state["history"])
if "total_cost" not in st.session_state:
    st.session_state.total_cost = app_state["total_cost"]
if "query_count" not in st.session_state:
    st.session_state.query_count = app_state["query_count"]

with st.sidebar:
    search_mode = st.radio(
        "Search mode",
        [QUERY_MODE, EXHAUSTIVE_MODE],
        index=0,
        help="Exhaustive checks every paper in scope. It takes minutes, costs more, and asks you to confirm first.",
    )
    st.header("Usage")
    st.metric("Queries this session", st.session_state.query_count)
    st.metric("Total cost this session", f"${st.session_state.total_cost:.4f}")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.total_cost = 0.0
        st.session_state.query_count = 0
        sync_persistent_state(app_state)
        st.rerun()

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant" and "cost" in item:
            render_copy_button(item["content"], key=f"copy_history_{item.get('id', 0)}")
            render_source_passages(item.get("contexts", []))
            if "report_path" in item:
                render_exhaustive_report(item["report_path"])
            st.caption(f"Cost: ${item['cost']:.4f}")

exhaustive_pending = st.session_state.get("exhaustive_pending")
exhaustive_run = st.session_state.get("exhaustive_run")
question = st.chat_input("Ask a question about your PDFs...", disabled=bool(exhaustive_pending or exhaustive_run))

if exhaustive_run:
    render_exhaustive_run(exhaustive_run, app_state)
elif exhaustive_pending:
    render_exhaustive_pending(exhaustive_pending, app_state)
elif question and search_mode == EXHAUSTIVE_MODE:
    plan_exhaustive_question(question)
elif question:
    st.session_state.history.append({"role": "user", "content": question})
    sync_persistent_state(app_state)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs..."):
            try:
                answer, cost, contexts = run_query_subprocess(question)
            except Exception as exc:
                answer = (
                    "Indexing/search failed.\n\n"
                    "Top-level error:\n"
                    f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                    "Underlying error(s):\n"
                    f"```\n{chr(10).join(flatten_exception_messages(exc))}\n```\n\n"
                    "This usually means one document or a parsing/indexing step failed."
                )
                cost = 0.0
                contexts = []
                st.error(answer)
                st.caption("Cost: $0.0000")
                st.code("".join(traceback.format_exception(exc)))
            else:
                st.markdown(answer)
                render_copy_button(answer, key=f"copy_current_{len(st.session_state.history)}")
                render_source_passages(contexts)
                st.caption(f"Cost: ${cost:.4f}")

    st.session_state.history.append(
        {
            "role": "assistant",
            "id": len(st.session_state.history),
            "content": answer,
            "cost": cost,
            "contexts": contexts,
        }
    )
    st.session_state.total_cost += cost
    st.session_state.query_count += 1
    sync_persistent_state(app_state)
    st.rerun()

