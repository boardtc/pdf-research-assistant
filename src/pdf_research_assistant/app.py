import base64
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import streamlit as st

from pdf_research_assistant import (
    bootstrap,
    estimate,
    exhaustive_client,
    index_record,
    local_llm,
    rate_profiles,
    rebuild_client,
    saved_settings,
    settings_groups,
)

os.environ.setdefault("PQA_INDEX_DONT_CACHE_INDEXES", "true")

QUERY_MODE = "Query"
EXHAUSTIVE_MODE = "Exhaustive"
PROGRESS_POLL_SECONDS = 2
INSTANT_SETTINGS_TITLE = "Applies to your next question"
INSTANT_SETTINGS_NOTE = (
    "Change these whenever you like. They take effect on your next question, and the index is not rebuilt."
)
REINDEX_SETTINGS_TITLE = "Needs an index rebuild"
REINDEX_SETTINGS_NOTE = (
    "These decide how your PDFs are read and stored in the index. Changing any of them means the app has to "
    "index every PDF again, which can take a long time and costs money for embeddings. Nothing changes until "
    "you click **Save index settings**. The ones without a control are fixed in `.env` or in the app itself."
)
REBUILD_BADGE = ":orange-badge[Needs rebuild]"
INDEX_CHANGES_NOTE = (
    "To keep using that index, put these settings back as they were, under **Library and index settings** in "
    "the sidebar. Otherwise the app has to index every PDF again with the new settings, which can take a long "
    "time and may cost money for embeddings. Click **Rebuild the index** below to start that."
)
INDEX_SETTING_KEYS = [setting.key for setting in settings_groups.REINDEX_SETTINGS if setting.write is not None]
INDEX_SETTINGS_SAVE_LABEL = "Save index settings"
INDEX_SETTINGS_RESET_LABEL = "Go back to the default index settings"
INDEX_SETTINGS_SAVED_NOTICE = (
    "Index settings saved. PaperQA now looks for an index built with them, so your PDFs have to be indexed "
    "again before they can be searched. Click **Rebuild the index** to do that."
)
REBUILD_LABEL = "Rebuild the index"
REBUILD_HELP = (
    "Indexes your PDFs with the library and index settings saved here. PDFs already in the index are left "
    "alone; new ones, and ones that failed before, are indexed. You are shown what it involves and asked "
    "to confirm before anything starts."
)
REBUILD_CONFIRM_LABEL = "Start paid rebuild"
REBUILD_TIME_NOTE = (
    "A large library can take hours. Questions are switched off until the rebuild finishes, and it carries on "
    "if you close this page."
)
REBUILD_RUNNING_NOTE = "The index is being rebuilt. Questions and index settings are switched off until it finishes."
REBUILD_QUESTION_NOTE = "Your question was not sent, because the index is being rebuilt. Ask it again afterwards."
REBUILD_LOCKED_NOTE = "These cannot be changed while the index is being rebuilt."
REBUILD_FINISHED_NOTICE = "The index rebuild has finished. This page shows what the index holds now."
REBUILD_FAILED_NOTICE = "The index rebuild stopped unexpectedly, so the index may be incomplete."
REBUILD_LOG_TITLE = "Show the end of the rebuild log"
INDEX_SETTINGS_UNCHANGED_NOTE = "These are the settings already in use, so nothing was saved."
MEDIA_PACKAGE_NOTE = (
    "Images and tables cannot be read yet, because the `pypdfium2` package is not installed. Install it with "
    "`pip install paper-qa-pypdf[media]`, restart the app and save again. Nothing was changed."
)
MEDIA_GUIDE = (
    "**When to turn images and tables on:** for PDFs where the answers are in charts, diagrams, tables or "
    "slides rather than in the text. Leave it off for papers and books that are mostly text: it makes "
    "indexing slower and every question that finds a picture cost more. Without whole-page pictures, each "
    "picture stored in the PDF (photos, scanned pages, pasted-in charts) and each table with ruled lines is "
    "cut out on its own. Charts and diagrams drawn with lines and shapes, and tables without lines, are "
    "missed. Whole-page pictures catch everything on a page, at the cost of a larger picture for every page."
)
SOURCE_SETTING_KEYS = ["evidence_k", "answer_max_sources"]
SOURCE_SETTINGS_NOTE = "Exhaustive search ignores these two numbers, because it checks every passage in scope."
SAVED_SETTING_KEYS = [setting.key for setting in settings_groups.INSTANT_SETTINGS if setting.write is not None]
SETTINGS_RESET_LABEL = "Go back to the default settings"
NO_PROFILE_NOTE = "The speed limits in use match none of the profiles. Pick one to use its limits."
LOCAL_PROFILE_NOTE = (
    "Answers come from a local model, which has no account limits. Only the number of requests sent at once "
    "matters: a lower profile sends fewer, which a small local server may handle better."
)
ONE_OFF_TITLE = "Change for the next question only"
ONE_OFF_TOGGLE_LABEL = "Use different settings for my next question"
ONE_OFF_NOTE = (
    "Try other values on one question without changing your usual settings above. They are used for your "
    "next question only, marked under that answer, and then switched off again. "
    "Exhaustive search ignores them."
)
NO_CHUNK_TEXT_NOTE = "No chunk text came back with this passage."
NO_MEDIA_NOTE = (
    "No images or tables are stored with this chunk. They are only kept when the index is built with "
    "the Images and tables setting turned on."
)
IMAGE_FAILED_NOTE = "This image could not be shown."
SCORE_NOTE = (
    "Relevance score: the 0 to 10 rating the summary model gave each passage for your question while "
    "summarising it, not a measure of how closely the search matched. Scores tend to bunch at 0 or "
    "8 to 9, so treat them as a rough guide."
)
ANSWER_SETTINGS_TITLE = "Show settings used"
ESTIMATE_LABEL = "**Next question:**"
ESTIMATE_DETAILS_TITLE = "What this estimate is based on"
EXHAUSTIVE_ESTIMATE_NOTE = (
    "Exhaustive search reads every passage in scope, so it costs and takes far more than a question. "
    "After you ask, it shows its own estimate and waits for you to confirm."
)
LOCAL_LLM_GUIDE = (
    "Use a model running on your own computer to write answers, instead of OpenAI.\n\n"
    "1. Install and start [Ollama](https://ollama.com), or another local server such as LM Studio.\n"
    "2. Download a model that supports tool calling, for example `ollama pull llama3.2`.\n"
    "3. Fill in the details below and click **Check connection**.\n"
    "4. When the check says it is ready, click **Use this local model**.\n\n"
    "**How this differs from OpenAI**\n\n"
    "- The local model takes over every model step in PaperQA: it reads each passage, writes the answer "
    "and plans the searches.\n"
    "- Finding passages in your papers still uses OpenAI embeddings, so the OpenAI key is still needed "
    "and your index is not rebuilt.\n"
    "- The answer model has no cost, so the **Next question** estimate gives no cost figure for it.\n"
    "- Local models are usually slower than OpenAI, and small ones may give weaker answers or lose track "
    "of PaperQA's instructions.\n"
    "- **Speed limits** then only sets how many requests are sent at once; OpenAI's account limits "
    "no longer apply."
)
LOCAL_LLM_PROVIDER_HELP = (
    "The program that runs the model on your computer. Pick **Ollama** if you use Ollama. Pick "
    "**OpenAI-compatible server** for LM Studio, llama.cpp or any other server that offers the OpenAI API. "
    "PaperQA talks to the two kinds differently, so this must match your server."
)
LOCAL_LLM_ADDRESS_HELP = (
    "Where your local server is listening. The box starts with the usual address for the server type: "
    "Ollama uses http://localhost:11434 and LM Studio uses http://localhost:1234/v1. Change it only if "
    "you changed the port or the server runs on another computer. An OpenAI-compatible address usually "
    "ends in /v1."
)
LOCAL_LLM_MODEL_HELP = (
    "The model's name exactly as your server lists it, for example `llama3.2`. With Ollama, run "
    "`ollama list` to see your models. The model must already be downloaded and should support tool "
    "calling, because PaperQA uses tools to search your papers. Bigger models give better answers but "
    "run more slowly."
)
LOCAL_LLM_CHECK_HELP = (
    "Asks the server which models it has and says whether these details will work. Nothing is saved "
    "and no question is sent."
)
LOCAL_LLM_SAVE_HELP = (
    "Checks the connection again and, if it works, saves these details. The app, the command line and "
    "exhaustive search then use this model for answers. Your index is not rebuilt."
)
LOCAL_LLM_CLEAR_HELP = "Forgets the local model details, so answers come from OpenAI again."


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


def describe_score(ctx: dict) -> str | None:
    """Return a passage's relevance score as text, or None when PaperQA set none.

    PaperQA uses -1 for a score that was never set; answers saved before scores were kept have none.
    """
    score = ctx.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or score < 0:
        return None
    return f"{score} out of 10"


def describe_chunk_facts(ctx: dict) -> list[tuple[str, str]]:
    """Return the labelled facts shown at the top of the chunk view, in order.

    Anything else known about a chunk is added here as one more pair.
    """
    facts = [("Source", ctx.get("name", "Source"))]
    if "cited" in ctx:
        facts.append(("Cited in the answer", "yes" if ctx["cited"] else "no"))
    score = describe_score(ctx)
    if score:
        facts.append(("Relevance score", score))
    return facts


def describe_media(media: dict) -> str:
    label = (media.get("kind") or "Image").capitalize()
    if media.get("page") is not None:
        label += f", page {media['page']}"
    return label


def render_media_image(media: dict) -> None:
    """Show a chunk's image from its link or its stored bytes, or say that it could not be shown."""
    try:
        st.image(media.get("url") or base64.b64decode(media["image"], validate=True))
    except Exception:
        st.caption(IMAGE_FAILED_NOTE)


def render_chunk_media(media_items: list[dict]) -> None:
    if not media_items:
        st.caption(NO_MEDIA_NOTE)
        return
    for media in media_items:
        st.markdown(f"**{describe_media(media)}**")
        if media.get("url") or media.get("image"):
            render_media_image(media)
        if media.get("text"):
            st.markdown(escape_dollars(media["text"]))
        if media.get("description"):
            st.caption(media["description"])


@st.dialog("Chunk details", width="large")
def show_chunk_details(ctx: dict) -> None:
    """Show one retrieved chunk as the query returned it: its facts, then its text, then its media."""
    st.caption(" · ".join(f"{label}: {value}" for label, value in describe_chunk_facts(ctx)))
    if describe_score(ctx):
        st.caption(SCORE_NOTE)
    st.subheader("Chunk text")
    if ctx.get("raw_text"):
        # Plain text, so nothing in the PDF text is read as Markdown or maths.
        st.text(ctx["raw_text"])
    else:
        st.caption(NO_CHUNK_TEXT_NOTE)
    st.subheader("Images and tables")
    render_chunk_media(ctx.get("media") or [])


def render_source_passages(contexts, key: str = "passages"):
    """List each passage's summary and relevance score with a button that opens the chunk behind it."""
    if not contexts:
        return
    with st.expander("Show source passages"):
        if any(describe_score(ctx) for ctx in contexts):
            st.caption(SCORE_NOTE)
        for position, ctx in enumerate(contexts):
            st.subheader(ctx.get("name", "Source"))
            score = describe_score(ctx)
            if score:
                st.caption(f"Relevance score: {score}")
            st.write(ctx.get("context", ""))
            if st.button("Open chunk", key=f"open_chunk_{key}_{position}"):
                show_chunk_details(ctx)


def render_answer_settings(recorded: dict | None) -> None:
    """List the index and settings a query used, as the query process recorded them.

    Answers saved without a record, such as exhaustive-search answers, show nothing.
    """
    if not recorded:
        return
    with st.expander(ANSWER_SETTINGS_TITLE):
        st.markdown(settings_groups.describe_line("Index", recorded["index"]))
        for item in recorded["values"]:
            st.markdown(settings_groups.describe_recorded_setting(item))


def run_query_subprocess(question: str, one_off: dict | None = None) -> tuple[str, float, list[dict], dict | None]:
    """Run one question in its own process, with any settings changed for this question only."""
    helper = Path(__file__).with_name("query_once.py")
    command = [sys.executable, str(helper), question]
    if one_off:
        command.append(json.dumps(one_off))
    result = subprocess.run(
        command,
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
    return payload["answer"], payload["cost"], payload.get("contexts", []), payload.get("settings")


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


def select_index_status(indexed_count: int, manifest_count: int) -> list[tuple[str, str]]:
    """Return the index banner as (level, message) pairs; a manifest count of 0 means no manifest."""
    use_manifest_scope = manifest_count > 0
    if indexed_count == 0 and use_manifest_scope:
        return [
            (
                "warning",
                f"Index not found. The first question will rebuild the index for {manifest_count} manifest PDFs, "
                "which may take a long time.",
            ),
            ("info", f"A rebuild starts only when you click **{REBUILD_LABEL}** or submit a question below."),
        ]
    if use_manifest_scope and indexed_count < manifest_count:
        return [
            (
                "warning",
                f"Index is incomplete: {indexed_count} of {manifest_count} manifest PDFs appear indexed. "
                "The next question should continue or trigger indexing work.",
            )
        ]
    if indexed_count == 0:
        return [
            ("warning", "Index not found. The first question will build the PDF index, which may take a long time."),
            ("info", "No manifest.csv was found, so all PDFs under PAPER_DIR are in scope."),
        ]
    if use_manifest_scope:
        return [("caption", f"Index available for {indexed_count} manifest PDFs.")]
    return [("caption", f"Index available for {indexed_count} PDFs under PAPER_DIR.")]


def render_index_status(status: list[tuple[str, str]]) -> None:
    for level, message in status:
        getattr(st, level)(message)


def describe_active_index(
    active_index_dir: Path, manifest_path: Path, manifest_count: int, paper_dir: Path
) -> dict[str, str | None]:
    """Describe the index the query path uses; a manifest count of 0 means no paper list is in use.

    "name" is the folder name for now, so a friendly name can replace it later without changing the layout.
    """
    use_manifest_scope = manifest_count > 0
    return {
        "name": active_index_dir.name,
        "folder": str(active_index_dir),
        "paper_list": (
            f"`{manifest_path.name}` ({manifest_count} PDFs)"
            if use_manifest_scope
            else "none, so all PDFs under PAPER_DIR are in scope"
        ),
        "paper_list_path": str(manifest_path) if use_manifest_scope else None,
        "paper_dir": str(paper_dir),
    }


def describe_index_changes(changes: dict) -> str:
    """Return the warning for library and index settings that changed since the last index was built."""
    parts = [
        "**Your library and index settings no longer match the index.** They have changed since the index "
        f"`{changes['index']}` was built, so the app is now looking for an index built with the new settings, "
        "and there is none yet."
    ]
    if changes["settings"]:
        parts.append(
            "\n".join(f"- {item['label']}: was `{item['was']}`, now `{item['now']}`" for item in changes["settings"])
        )
    parts.append(INDEX_CHANGES_NOTE)
    return "\n\n".join(parts)


def render_index_changes(changes: dict | None) -> None:
    if changes:
        st.warning(describe_index_changes(changes))


def render_active_index(details: dict[str, str | None]) -> None:
    st.caption(f"Active index: `{details['name']}` · Paper list: {details['paper_list']}")
    with st.expander("Active index details"):
        st.markdown(f"**Index folder:** `{details['folder']}`")
        if details["paper_list_path"]:
            st.markdown(f"**Paper list:** `{details['paper_list_path']}`")
        st.markdown(f"**PDF folder:** `{details['paper_dir']}`")


def index_needs_rebuild(indexed_count: int, manifest_count: int, failed_files: list[str]) -> bool:
    """Return whether the page can tell that the index is out of date; a manifest count of 0 means no manifest.

    An empty index covers changed index settings too, because PaperQA then looks for a new, empty index.
    Without a manifest the page cannot tell that PDFs were added, so the sidebar always offers a rebuild.
    """
    return indexed_count == 0 or indexed_count < manifest_count or bool(failed_files)


def describe_rebuild_plan(settings, active_index_dir: Path, pdf_count: int, indexed_count: int) -> list[str]:
    """Describe what a rebuild will do and what is paid for, so the user can decide whether to start it."""
    scope = f"PDFs in scope: {pdf_count}"
    if indexed_count:
        scope += f". {indexed_count} are already in this index and are not indexed again"
    lines = [
        scope,
        f"Index: `{active_index_dir.name}`, built with the library and index settings saved in the sidebar",
        f"Each PDF that is indexed is read, split into passages and sent to the embedding model "
        f"`{settings.embedding}`, which costs money. The app cannot estimate how much before reading the PDFs",
    ]
    media = settings_groups.current_choice(settings_groups.get_setting("multimodal"), settings)
    if media == "on_with_enrichment":
        lines.append(
            "Images and tables are on with enrichment, so the model "
            f"`{settings.parsing.enrichment_llm}` is also paid to describe every image and table"
        )
    lines.append(REBUILD_TIME_NOTE)
    return lines


def request_rebuild() -> None:
    """Ask the user to confirm a rebuild. Nothing starts here."""
    st.session_state.rebuild_pending = True
    st.rerun()


def render_rebuild_pending(settings, active_index_dir: Path, manifest_count: int, indexed_count: int) -> None:
    """Show what the rebuild involves and start it only when the user confirms."""
    pdf_count = manifest_count or rebuild_client.count_pdfs(bootstrap.PAPER_DIR)
    st.markdown(f"**{REBUILD_LABEL}**")
    st.markdown(
        "\n".join(f"- {line}" for line in describe_rebuild_plan(settings, active_index_dir, pdf_count, indexed_count))
    )
    if st.button(REBUILD_CONFIRM_LABEL, key="rebuild_confirm"):
        rebuild_client.start(active_index_dir, pdf_count)
        st.session_state.rebuild_pending = False
        st.rerun()
        return
    if st.button("Cancel", key="rebuild_cancel"):
        st.session_state.rebuild_pending = False
        st.rerun()


@st.fragment(run_every=PROGRESS_POLL_SECONDS)
def render_rebuild_run() -> None:
    """Show how far the rebuild has got, and draw the whole page again once it has finished."""
    run = rebuild_client.get_run()
    # No run means another browser session already saw it finish.
    if run is None or rebuild_client.collect_result():
        st.rerun()
        return
    # PaperQA saves the index's file list after each PDF, whether it was indexed or failed.
    done = rebuild_client.note_progress(
        run,
        bootstrap.get_indexed_doc_count(index_dir=run["index_dir"])
        + len(bootstrap.get_failed_files(index_dir=run["index_dir"])),
    )
    st.info(REBUILD_RUNNING_NOTE)
    st.progress(rebuild_client.progress_fraction(run, done), text=rebuild_client.describe_progress(run, done))
    st.caption(f"Details are written to `{run['log_path']}`.")


def render_rebuild_result(result: dict | None) -> None:
    """Say how the last rebuild ended, with the end of its log."""
    if not result:
        return
    if result["ok"]:
        st.success(REBUILD_FINISHED_NOTICE)
    else:
        st.error(f"{REBUILD_FAILED_NOTICE} See `{result['log_path']}`.")
    if result["lines"]:
        with st.expander(REBUILD_LOG_TITLE):
            st.code("\n".join(result["lines"]))


def render_rebuild(
    settings, active_index_dir: Path, manifest_count: int, indexed_count: int, out_of_date: bool, busy: bool
) -> None:
    """Show the rebuild's result, progress or confirm step, or offer a rebuild when the index is out of date."""
    render_rebuild_result(rebuild_client.take_result())
    if rebuild_client.get_run():
        render_rebuild_run()
    elif st.session_state.get("rebuild_pending"):
        render_rebuild_pending(settings, active_index_dir, manifest_count, indexed_count)
    elif out_of_date and not busy and st.button(REBUILD_LABEL, key="rebuild_offer_main", help=REBUILD_HELP):
        request_rebuild()


def render_failed_files(failed_files: list[str]) -> None:
    if not failed_files:
        return
    st.error(f"{len(failed_files)} PDF(s) failed to index.")
    with st.expander("Show failed PDFs"):
        for file_location in failed_files:
            st.code(file_location)


def init_session_state(app_state: dict) -> None:
    """Start a new browser session from the saved state, leaving a session that already has values alone."""
    if "history" not in st.session_state:
        st.session_state.history = deepcopy(app_state["history"])
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = app_state["total_cost"]
    if "query_count" not in st.session_state:
        st.session_state.query_count = app_state["query_count"]


def setting_widget_key(setting_key: str) -> str:
    """Return a sidebar control's key, which changes each time the saved settings go back to the defaults."""
    return f"setting_{setting_key}_{st.session_state.get('answer_settings_round', 0)}"


def draw_whole_number(setting: settings_groups.Setting, current: int, high: int, key: str) -> int:
    """Draw a number box for a setting, starting at the current value brought inside its limits.

    The key keeps it the same widget when its starting value changes, so quick clicks are not lost.
    Streamlit puts the box back to its starting value if the number in it goes above a lowered maximum.
    """
    low = setting.check.low
    return st.number_input(
        setting.label,
        min_value=low,
        max_value=high,
        value=min(max(current, low), high),
        step=1,
        help=setting.help,
        key=key,
    )


def pick_whole_number(setting: settings_groups.Setting, current: int, high: int) -> int:
    """Draw a number box for a setting, and save the chosen number when it differs from the one in use."""
    chosen = draw_whole_number(setting, current, high, setting_widget_key(setting.key))
    if chosen != current:
        saved_settings.save_value(setting.key, chosen)
    return chosen


def render_source_settings(settings) -> None:
    """Let the user choose how many passages are gathered and how many the answer may cite.

    The numbers start at the values the next question will use. A changed number is saved at once, so the
    query process picks it up when it builds its settings.
    """
    gathered_setting, sources_setting = (settings_groups.get_setting(key) for key in SOURCE_SETTING_KEYS)
    gathered = pick_whole_number(gathered_setting, gathered_setting.read(settings), gathered_setting.check.high)
    # PaperQA cannot cite more passages than it gathered.
    pick_whole_number(sources_setting, sources_setting.read(settings), min(sources_setting.check.high, gathered))
    st.caption(SOURCE_SETTINGS_NOTE)


def describe_answer_length(phrase: str) -> str:
    """Return an answer length as its name and phrase, or just the phrase if it is not one of the choices."""
    name = settings_groups.get_setting("answer_length").check.name_for(phrase)
    return f"{name} ({phrase})" if name else phrase


def draw_answer_length(setting: settings_groups.Setting, current: str, key: str) -> str:
    """Draw the answer length choices, starting at the current length.

    A length that is not one of the choices, such as a changed PaperQA default, is offered as well, so
    drawing the control never changes the setting by itself.
    """
    options = list(setting.check.values)
    if current not in options:
        options.append(current)
    return st.selectbox(
        setting.label,
        options,
        index=options.index(current),
        format_func=describe_answer_length,
        help=setting.help,
        key=key,
    )


def render_answer_length(settings) -> None:
    """Let the user choose how long the answer is asked to be, saving a new choice at once."""
    setting = settings_groups.get_setting("answer_length")
    current = setting.read(settings)
    chosen = draw_answer_length(setting, current, setting_widget_key(setting.key))
    if chosen != current:
        saved_settings.save_value(setting.key, chosen)


def render_rate_profile(settings) -> None:
    """Let the user choose how fast PaperQA may send requests, saving a new choice at once.

    Limits that match no profile are offered as they are, so drawing the control never changes them.
    """
    setting = settings_groups.get_setting("rate_profile")
    current = setting.read(settings)
    options = list(setting.check.values)
    if current not in options:
        options.append(current)
    chosen = st.selectbox(
        setting.label,
        options,
        index=options.index(current),
        help=setting.help,
        key=setting_widget_key(setting.key),
    )
    if chosen != current:
        saved_settings.save_value(setting.key, chosen)
        current = chosen
    if current == rate_profiles.NO_PROFILE:
        st.caption(NO_PROFILE_NOTE)
    else:
        st.caption(rate_profiles.describe_profile(rate_profiles.get_profile(current)))
    if local_llm.load_config():
        st.caption(LOCAL_PROFILE_NOTE)


def render_settings_reset() -> None:
    """Offer to remove every saved sidebar setting, once at least one is saved."""
    saved = saved_settings.load()
    if any(key in saved for key in SAVED_SETTING_KEYS) and st.button(SETTINGS_RESET_LABEL, key="answer_settings_reset"):
        saved_settings.forget(SAVED_SETTING_KEYS)
        # The browser keeps the old values in controls it has already drawn, so draw new ones.
        st.session_state.answer_settings_round = st.session_state.get("answer_settings_round", 0) + 1
        st.rerun()


def one_off_widget_key(name: str, usual: object = "") -> str:
    """Return a one-off control's key.

    It changes after each question, so the controls start again switched off, and when the usual value
    changes, so an untouched box never holds on to an old usual value.
    """
    return f"one_off_{name}_{st.session_state.get('one_off_round', 0)}" + (f"_{usual}" if usual != "" else "")


def render_one_off_settings(settings) -> dict:
    """Let the user change settings for the next question only; return the changes, by setting key.

    Nothing here is saved. Only values that differ from the usual ones are returned, so the query process
    can mark exactly those as changed for this question.
    """
    with st.expander(ONE_OFF_TITLE):
        st.caption(ONE_OFF_NOTE)
        if not st.toggle(ONE_OFF_TOGGLE_LABEL, key=one_off_widget_key("enabled")):
            return {}
        gathered_setting, sources_setting = (settings_groups.get_setting(key) for key in SOURCE_SETTING_KEYS)
        length_setting = settings_groups.get_setting("answer_length")
        usual = {setting.key: setting.read(settings) for setting in (gathered_setting, sources_setting, length_setting)}
        gathered = draw_whole_number(
            gathered_setting,
            usual["evidence_k"],
            gathered_setting.check.high,
            one_off_widget_key("evidence_k", usual["evidence_k"]),
        )
        # PaperQA cannot cite more passages than it gathered.
        sources = draw_whole_number(
            sources_setting,
            usual["answer_max_sources"],
            min(sources_setting.check.high, gathered),
            one_off_widget_key("answer_max_sources", usual["answer_max_sources"]),
        )
        length = draw_answer_length(
            length_setting, usual["answer_length"], one_off_widget_key("answer_length", usual["answer_length"])
        )
    chosen = {"evidence_k": gathered, "answer_max_sources": sources, "answer_length": length}
    return {key: value for key, value in chosen.items() if value != usual[key]}


def render_instant_settings(settings) -> tuple[str, dict]:
    """Show the settings that apply to the next question; return the search mode and any one-off changes."""
    with st.sidebar:
        st.subheader(INSTANT_SETTINGS_TITLE)
        st.caption(INSTANT_SETTINGS_NOTE)
        search_mode = st.radio(
            "Search mode",
            [QUERY_MODE, EXHAUSTIVE_MODE],
            index=0,
            help="Exhaustive checks every paper in scope. It takes minutes, costs more, and asks you to confirm first.",
        )
        render_source_settings(settings)
        render_answer_length(settings)
        render_rate_profile(settings)
        render_settings_reset()
        one_off = render_one_off_settings(settings)
    render_local_llm_setup()
    return search_mode, one_off


def index_widget_key(setting_key: str) -> str:
    """Return an index control's key, which changes each time the index settings go back to the defaults."""
    return f"index_setting_{setting_key}_{st.session_state.get('index_settings_round', 0)}"


def draw_index_setting(setting: settings_groups.Setting, settings) -> object:
    """Draw the control for an index setting, starting at the value in use, and return the value left in it.

    The kind of control follows the setting's check: a number box, a list of choices, or a tick box.
    A value in use that the list does not offer is offered as well, so drawing never changes a setting.
    """
    current = settings_groups.current_choice(setting, settings)
    label = f"{setting.label} {REBUILD_BADGE}"
    key = index_widget_key(setting.key)
    check = setting.check
    if isinstance(check, settings_groups.WholeNumber):
        return st.number_input(
            label,
            min_value=check.low,
            max_value=check.high,
            value=min(max(current, check.low), check.high),
            step=check.step,
            help=setting.help,
            key=key,
        )
    if isinstance(check, settings_groups.OneOf):
        options = list(check.values)
        if current not in options:
            options.append(current)
        return st.selectbox(
            label,
            options,
            index=options.index(current),
            format_func=lambda value: check.name_for(value) or str(value),
            help=setting.help,
            key=key,
        )
    return st.checkbox(label, value=bool(current), help=setting.help, key=key)


def media_reading_available() -> bool:
    """Return whether the package PaperQA's PDF reader needs for images and tables is installed."""
    return importlib.util.find_spec("pypdfium2") is not None


def save_index_settings(chosen: dict, settings) -> None:
    """Save the index settings that differ from the ones in use, then draw the page again.

    Saving changes the index PaperQA looks for, so the next page load shows that the index needs building.
    """
    changed = {
        key: value
        for key, value in chosen.items()
        if value != settings_groups.current_choice(settings_groups.get_setting(key), settings)
    }
    if not changed:
        st.info(INDEX_SETTINGS_UNCHANGED_NOTE)
        return
    if changed.get("multimodal", settings_groups.MEDIA_OFF) != settings_groups.MEDIA_OFF and not media_reading_available():
        st.error(MEDIA_PACKAGE_NOTE)
        return
    for key, value in changed.items():
        saved_settings.save_value(key, value)
    st.session_state.index_settings_notice = INDEX_SETTINGS_SAVED_NOTICE
    st.rerun()


def render_index_settings_reset() -> None:
    """Offer to remove every saved index setting, once at least one is saved."""
    saved = saved_settings.load()
    if any(key in saved for key in INDEX_SETTING_KEYS) and st.button(
        INDEX_SETTINGS_RESET_LABEL, key="index_settings_reset"
    ):
        saved_settings.forget(INDEX_SETTING_KEYS)
        # The browser keeps the old values in controls it has already drawn, so draw new ones.
        st.session_state.index_settings_round = st.session_state.get("index_settings_round", 0) + 1
        st.rerun()


def render_changed_note(setting: settings_groups.Setting, changed: dict, index_changes: dict | None) -> None:
    """Say what a setting was when the last index was built, if it has changed since."""
    if setting.key in changed:
        st.caption(f"Changed. The index `{index_changes['index']}` was built with `{changed[setting.key]}`.")


def render_reindex_settings(settings, index_changes: dict | None = None, busy: bool = False) -> None:
    """Show each setting that cannot change without rebuilding the index, with a control where it can change.

    The controls sit in one form, so several can be changed and saved together; nothing is saved before then.
    index_changes comes from index_record.find_changes; each changed setting also says what it was.
    busy means a rebuild or an exhaustive search is under way or waiting to be confirmed, so no rebuild is
    offered. While a rebuild runs the settings cannot be saved either, because it is using them.
    """
    rebuilding = rebuild_client.get_run() is not None
    changed = {item["key"]: item["was"] for item in (index_changes or {}).get("settings", [])}
    with st.sidebar:
        st.subheader(REINDEX_SETTINGS_TITLE)
        st.caption(REINDEX_SETTINGS_NOTE)
        with st.expander("Library and index settings"):
            notice = st.session_state.pop("index_settings_notice", None)
            if notice:
                st.success(notice)
            chosen = {}
            with st.form("index_settings_form"):
                for setting in settings_groups.REINDEX_SETTINGS:
                    if setting.write is not None:
                        chosen[setting.key] = draw_index_setting(setting, settings)
                        render_changed_note(setting, changed, index_changes)
                        if setting.key == "multimodal":
                            st.caption(MEDIA_GUIDE)
                submitted = st.form_submit_button(INDEX_SETTINGS_SAVE_LABEL, disabled=rebuilding)
            if rebuilding:
                st.caption(REBUILD_LOCKED_NOTE)
            else:
                if submitted:
                    save_index_settings(chosen, settings)
                render_index_settings_reset()
                if not busy and st.button(REBUILD_LABEL, key="rebuild_offer_sidebar", help=REBUILD_HELP):
                    request_rebuild()
            # The settings with no control come last, below the ones that can be saved.
            for setting in settings_groups.REINDEX_SETTINGS:
                if setting.write is None:
                    st.markdown(f"{settings_groups.describe_setting(setting, settings)} {REBUILD_BADGE}")
                    st.caption(setting.help)
                    render_changed_note(setting, changed, index_changes)


def render_usage(app_state: dict) -> None:
    with st.sidebar:
        st.header("Usage")
        st.metric("Queries this session", st.session_state.query_count)
        st.metric("Total cost this session", f"${st.session_state.total_cost:.4f}")
        if st.button("Clear chat"):
            st.session_state.history = []
            st.session_state.total_cost = 0.0
            st.session_state.query_count = 0
            sync_persistent_state(app_state)
            st.rerun()


def render_sidebar(
    app_state: dict, settings, index_changes: dict | None = None, busy: bool = False
) -> tuple[str, dict]:
    """Show the sidebar, with settings in two groups; return the search mode and any one-off changes."""
    with st.sidebar:
        st.header("Settings")
    search_mode, one_off = render_instant_settings(settings)
    render_reindex_settings(settings, index_changes, busy)
    render_usage(app_state)
    return search_mode, one_off


def render_local_llm_setup() -> None:
    """Show the sidebar panel for checking, saving and removing local LLM settings."""
    saved = local_llm.load_config()
    providers = list(local_llm.PROVIDERS)
    with st.sidebar, st.expander("Local LLM setup"):
        st.caption(local_llm.describe_answer_model(saved, bootstrap.MODEL_NAME))
        notice = st.session_state.pop("local_llm_notice", None)
        if notice:
            st.success(notice)
        st.markdown(LOCAL_LLM_GUIDE)
        provider = st.selectbox(
            "Local server type",
            providers,
            index=providers.index(saved["provider"]) if saved else 0,
            format_func=lambda key: local_llm.PROVIDERS[key]["label"],
            key="local_llm_provider",
            help=LOCAL_LLM_PROVIDER_HELP,
        )
        # Each server type keeps its own address box, so picking another type shows that type's usual address.
        saved_address = saved["address"] if saved and saved["provider"] == provider else None
        address = st.text_input(
            "Server address",
            value=saved_address or local_llm.PROVIDERS[provider]["default_address"],
            key=f"local_llm_address_{provider}",
            help=LOCAL_LLM_ADDRESS_HELP,
        )
        model = st.text_input(
            "Model name", value=saved["model"] if saved else "", key="local_llm_model", help=LOCAL_LLM_MODEL_HELP
        )
        candidate = local_llm.clean_config(provider, address, model)

        if st.button("Check connection", key="local_llm_check", help=LOCAL_LLM_CHECK_HELP):
            ready, message = local_llm.check_connection(candidate)
            (st.success if ready else st.error)(message)
        if st.button("Use this local model", key="local_llm_save", help=LOCAL_LLM_SAVE_HELP):
            ready, message = local_llm.check_connection(candidate)
            if ready:
                local_llm.save_config(candidate)
                st.session_state.local_llm_notice = f"{message} Answers will now use this local model."
                st.rerun()
            else:
                st.error(f"{message} Nothing was changed.")
        if saved and st.button("Switch back to OpenAI", key="local_llm_clear", help=LOCAL_LLM_CLEAR_HELP):
            local_llm.clear_config()
            st.session_state.local_llm_notice = "Answers will now use OpenAI again."
            st.rerun()


def render_history(history: list[dict]) -> None:
    for item in history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and "cost" in item:
                render_copy_button(item["content"], key=f"copy_history_{item.get('id', 0)}")
                render_source_passages(item.get("contexts", []), key=f"history_{item.get('id', 0)}")
                render_answer_settings(item.get("settings"))
                if "report_path" in item:
                    render_exhaustive_report(item["report_path"])
                st.caption(f"Cost: ${item['cost']:.4f}")


def render_estimate(search_mode: str, one_off: dict) -> None:
    """Show a rough sense of what the next question will cost and how long it will take.

    The settings are built again here, after the sidebar may have saved a change, with any one-off values
    on top, just as the query process will build them, so the estimate follows every control at once.
    """
    if search_mode == EXHAUSTIVE_MODE:
        st.markdown(f"{ESTIMATE_LABEL} {estimate.badge(estimate.HIGHER_COST)}")
        st.caption(EXHAUSTIVE_ESTIMATE_NOTE)
        return
    settings = bootstrap.build_settings()
    settings_groups.apply_usable(settings, one_off)
    result = estimate.estimate_question(settings, local=bool(local_llm.load_config()))
    st.markdown(f"{ESTIMATE_LABEL} {estimate.badge(estimate.label_for(result))} {estimate.summarize(result)}")
    with st.expander(ESTIMATE_DETAILS_TITLE):
        for line in estimate.describe_estimate(result):
            st.caption(line)


def describe_query_failure(exc: BaseException) -> str:
    return (
        "Indexing/search failed.\n\n"
        "Top-level error:\n"
        f"```\n{type(exc).__name__}: {exc}\n```\n\n"
        "Underlying error(s):\n"
        f"```\n{chr(10).join(flatten_exception_messages(exc))}\n```\n\n"
        "This usually means one document or a parsing/indexing step failed."
    )


def answer_question(question: str, one_off: dict | None = None) -> tuple[str, float, list[dict], dict | None]:
    """Run one query and show its answer, or its failure; return what to record in the history."""
    with st.spinner("Searching PDFs..."):
        try:
            answer, cost, contexts, recorded_settings = run_query_subprocess(question, one_off)
        except Exception as exc:
            answer = describe_query_failure(exc)
            st.error(answer)
            st.caption("Cost: $0.0000")
            st.code("".join(traceback.format_exception(exc)))
            return answer, 0.0, [], None
        st.markdown(answer)
        render_copy_button(answer, key=f"copy_current_{len(st.session_state.history)}")
        render_source_passages(contexts, key=f"current_{len(st.session_state.history)}")
        render_answer_settings(recorded_settings)
        st.caption(f"Cost: ${cost:.4f}")
        return answer, cost, contexts, recorded_settings


def submit_question(question: str, app_state: dict, one_off: dict | None = None) -> None:
    """Record a Query-mode question, answer it, and add the answer and its cost to the session.

    one_off holds settings changed for this question only. They are used once: afterwards the one-off
    controls start again, switched off.
    """
    st.session_state.history.append({"role": "user", "content": question})
    sync_persistent_state(app_state)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer, cost, contexts, recorded_settings = answer_question(question, one_off)
    st.session_state.one_off_round = st.session_state.get("one_off_round", 0) + 1

    st.session_state.history.append(
        {
            "role": "assistant",
            "id": len(st.session_state.history),
            "content": answer,
            "cost": cost,
            "contexts": contexts,
            "settings": recorded_settings,
        }
    )
    st.session_state.total_cost += cost
    st.session_state.query_count += 1
    sync_persistent_state(app_state)
    st.rerun()


def main() -> None:
    """Draw the page. Streamlit reruns this file from the top on every interaction."""
    st.title("PDF Research Assistant")

    manifest_count = len(bootstrap.get_allowed_paths())
    settings = bootstrap.build_settings()
    active_index_dir = bootstrap.get_active_index_dir(settings)
    indexed_count = bootstrap.get_indexed_doc_count(index_dir=active_index_dir)
    # An index holding PDFs was built with the settings in use, so it becomes the one to compare against.
    # An empty one may mean a library or index setting changed since the last index was built.
    if indexed_count:
        index_record.remember(settings, active_index_dir)
        index_changes = None
    else:
        index_changes = index_record.find_changes(settings, active_index_dir)
    failed_files = bootstrap.get_failed_files(index_dir=active_index_dir)
    exhaustive_pending = st.session_state.get("exhaustive_pending")
    exhaustive_run = st.session_state.get("exhaustive_run")
    # A rebuild and a question both write to the index, so only one of them goes ahead at a time.
    rebuild_busy = bool(rebuild_client.get_run() or st.session_state.get("rebuild_pending"))
    busy = bool(exhaustive_pending or exhaustive_run or rebuild_busy)
    render_index_changes(index_changes)
    render_index_status(select_index_status(indexed_count, manifest_count))
    render_rebuild(
        settings,
        active_index_dir,
        manifest_count,
        indexed_count,
        out_of_date=index_needs_rebuild(indexed_count, manifest_count, failed_files),
        busy=busy,
    )
    render_active_index(
        describe_active_index(active_index_dir, bootstrap.MANIFEST_PATH, manifest_count, bootstrap.PAPER_DIR)
    )
    render_failed_files(failed_files)

    app_state = get_persistent_app_state()
    init_session_state(app_state)
    search_mode, one_off = render_sidebar(app_state, settings, index_changes, busy)
    render_history(st.session_state.history)

    if not busy:
        render_estimate(search_mode, one_off)
    question = st.chat_input("Ask a question about your PDFs...", disabled=busy)

    if exhaustive_run:
        render_exhaustive_run(exhaustive_run, app_state)
    elif exhaustive_pending:
        render_exhaustive_pending(exhaustive_pending, app_state)
    elif question and rebuild_busy:
        # The chat box is switched off, so this is a question sent just as a rebuild began. It must not run.
        st.warning(REBUILD_QUESTION_NOTE)
    elif question and search_mode == EXHAUSTIVE_MODE:
        plan_exhaustive_question(question)
    elif question:
        submit_question(question, app_state, one_off)


main()
