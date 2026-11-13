# PDF Research Assistant

![Python 3.11%2B](https://img.shields.io/badge/python-3.11%2B-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![PaperQA2](https://img.shields.io/badge/PaperQA2-FutureHouse-orange)

A local PDF research assistant built with [PaperQA2](https://github.com/Future-House/paper-qa). It can use either a manifest-controlled document library or all PDFs under a chosen root folder.

Core indexing and querying work on Windows, macOS, and Linux. The Streamlit `Copy answer` clipboard button is currently Windows-only.

![PDF Research Assistant screenshot](images/streamlit_app.jpg)

> **Important**
> PaperQA2 uses retrieval-augmented generation (RAG). It returns real source passages and page references, but it can still overstate, paraphrase, or extrapolate beyond what the source explicitly says. Treat answers as a starting point for exploration, not as a citable summary. Always verify important claims against the source passages and the original PDF.

## Getting Started

### Requirements

- Python 3.11 or newer
- An OpenAI API key

Install the Python dependencies:

```bash
cd ~/gitrepos/pdf-research-assistant
python -m pip install -e .
```

<details>
<summary>More info: where do these commands come from?</summary>

Running `python -m pip install -e .` installs this project in editable mode.

That does two useful things:

- it makes the `pdf_research_assistant` package importable in your current Python environment
- it creates the `pdf-research` and `pdf-research-rebuild` console commands defined in `pyproject.toml`

These are three ways to run the rebuild flow:

- Installed console command:
  `pdf-research-rebuild`
- Module form:
  `python -m pdf_research_assistant.rebuild`
- Direct file path form:
  `python src/pdf_research_assistant/rebuild.py`

The first is the normal user-facing command. The second is a Python module form many developers prefer. The third is the old-school "run this file directly" form.

</details>

Create an OpenAI API key in the OpenAI dashboard:

- [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Setup

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` in `.env`.
3. Set `PAPER_DIR` in `.env` to the root folder containing your PDFs.
4. Optional: copy `manifest.example.csv` to `manifest.csv` if you want curated scope and metadata.
5. If you use `manifest.csv`, replace the example rows with paths relative to your chosen `PAPER_DIR`.

## Project Layout

This project uses a standard Python `src` layout.

- `src/pdf_research_assistant/` contains the application code
- `tests/` contains the test suite
- `pyproject.toml` defines dependencies and the `pdf-research` and `pdf-research-rebuild` commands

The `pdf_research_assistant` folder under `src` is the importable package, so code uses imports such as `from pdf_research_assistant.bootstrap import build_settings`.

## Usage

### Streamlit UI

```bash
cd ~/gitrepos/pdf-research-assistant
streamlit run src/pdf_research_assistant/app.py
```

Streamlit usually opens the app in your browser automatically and prints the local URL in the terminal. By default, it uses `http://localhost:8501` unless that port is already in use.

The Streamlit sidebar shows:

- query count for the current session
- total session cost
- a button to clear the chat

Each assistant response also includes:

- a `Copy answer` button that copies the full answer text to the clipboard on Windows
- a `Show source passages` expander with the retrieved evidence passages used for the answer

Clipboard support for the `Copy answer` button is currently implemented for Windows only.

On first use, there is no search index yet. The first query builds it, which can take a while for a large PDF library.

Each question runs in a fresh helper process so repeated questions in the same session start with clean query state.

### CLI

```bash
cd ~/gitrepos/pdf-research-assistant
pdf-research
```

Type a question at the prompt to search your indexed PDFs and return a cited answer with page references. Type `quit` to exit.

Like the Streamlit app, the CLI runs each question in a fresh helper process so repeated questions start with clean query state.

### Exhaustive Search

Normal queries are search-first: PaperQA picks candidate papers, ranks the top chunks, and caps the sources in the answer, so a relevant paper can be missed without warning. Exhaustive search checks every chunk of every paper in scope instead, which suits searching for ideas across a whole library.

Query mode stays the default. To use exhaustive search:

- In the Streamlit UI, choose `Exhaustive` under `Search mode` in the sidebar.
- In the CLI, run `pdf-research --exhaustive`.

Before each run you see the paper count, any papers that cannot be searched, the estimated cost, and the estimated time. Nothing is spent until you confirm. Afterwards you see the actual cost and runtime.

Each run saves its output in a new folder under `exhaustive_runs/`:

- `report.md` is the readable report: evidence grouped by paper with page references, strongest first
- `results.jsonl` holds each paper's result, including the full chunk text, and is saved as each paper finishes
- `summary.json` holds the run totals
- `run.log` holds any helper output

Every paper in scope appears in the report with a status:

| Status | Meaning |
| --- | --- |
| Relevant evidence found | At least one chunk was judged relevant |
| Possible misses only | Chunks were rejected, but they keep mentioning the topic, so check them |
| No relevant evidence | Every chunk was checked and rejected |
| Incomplete | Some chunks could not be checked, and none were judged relevant |
| Not processed | The run stopped early, usually after repeated failures |
| Failed to index, Not indexed, Unreadable | The paper could not be searched |

A failed request or rate limit is never reported as "no relevant evidence".

> **Important**
> Exhaustive search is more thorough than a normal query, but it can still misjudge relevance. A model scores each chunk, so it can reject something relevant or accept something that is not. Scanned pages and figures are not searched. Quotes are marked verified or not verified, and a quote marked not verified may still be genuine, so check it against the PDF. Treat the report as a list to review, not a conclusion.

Exhaustive search reads the saved index and never builds it, so run `pdf-research-rebuild` first. Its speed settings are in `src/pdf_research_assistant/exhaustive.py`: `MAX_CONCURRENT_REQUESTS` and `TOKENS_PER_MINUTE`. Lower them if your OpenAI account has a lower rate limit. You can see your limits at [platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits).

### Local LLM

By default, answers are written by OpenAI's `gpt-4o-mini`. To use a model running on your own computer instead:

1. Install and start [Ollama](https://ollama.com), or another local server that speaks the OpenAI API, such as LM Studio or llama.cpp.
2. Download a model that supports tool calling, for example `ollama pull llama3.2`.
3. In the app's sidebar, open **Local LLM setup**, choose the server type, and enter the server address and model name.
4. Click **Check connection**. The app asks the server which models it has and tells you whether your settings are ready.
5. Click **Use this local model**. Settings are only saved if the check passes.

**Switch back to OpenAI** in the same panel removes the saved settings.

The settings are saved in `local_llm.json` in the project folder, which Git ignores, and apply to the app, the CLI and exhaustive search.

> **Note**
> Only the model that reads passages and writes answers changes. Finding passages still uses OpenAI embeddings, so `OPENAI_API_KEY` is still needed, and switching does not rebuild your index. Local models are usually slower than OpenAI, and exhaustive search shows no cost estimate for them.

### Rebuild Index

```bash
cd ~/gitrepos/pdf-research-assistant
pdf-research-rebuild
```

Use this when:

- running the project for the first time and you want to build the index explicitly
- you add new PDFs and want to rebuild the index before querying again
- you want a terminal-only indexing run instead of letting the first query build the index

On a clean rebuild, it is normal to see `Manifest PDFs: <n>` and `Indexed before run: 0` before indexing starts.

See `windows-helper-commands.example.md` for optional PowerShell commands that help check index build progress and troubleshoot rebuild issues on Windows. If you want a version with your own local paths ready to copy and paste, create `windows-helper-commands.md` from it.

## Environment Variables

The app and CLI read settings from environment variables and load values from `.env` when present.

By default, `INDEX_DIR` and `MANIFEST_PATH` are resolved relative to the repository root, so the project can be moved without changing code. `PAPER_DIR` can point to any folder on your system using normal paths for your OS. If `manifest.csv` is present, it stores paths relative to that root folder. If `manifest.csv` is absent, the app indexes all PDFs under `PAPER_DIR`.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API key for PaperQA2 queries | unset |
| `PAPER_DIR` | Common root folder containing your PDFs | required |
| `INDEX_DIR` | Optional override for where the local PaperQA index is stored | `<repo-root>/index` |
| `MANIFEST_PATH` | Optional CSV manifest of allowed PDFs | `<repo-root>/manifest.csv` |
| `PDF_RESEARCH_ASSISTANT_SYNC_DIR` | Optional destination folder for post-push copies of `.env`, `manifest.csv`, and your private `windows-helper-commands.md` notes | unset |

See `.env.example` for the expected keys. Leave `INDEX_DIR` and `MANIFEST_PATH` unset to use the repo-root defaults.

If `PDF_RESEARCH_ASSISTANT_SYNC_DIR` is set, the tracked `post-push` hook copies `.env` and `manifest.csv` when present. It also copies `windows-helper-commands.md` if you created a private local version for your own use.

## Additional Notes

- The search index is stored in the folder set by `INDEX_DIR`.
- If `manifest.csv` exists, the app uses it to decide which PDFs are in scope and which metadata to use.
- If `manifest.csv` does not exist, the app indexes all PDFs under `PAPER_DIR`.
- `windows-helper-commands.example.md` is a safe-to-share template. If you want a version with your own local paths ready to copy and paste, create `windows-helper-commands.md` from it.
- The app automatically ignores the broken loopback proxy placeholder `127.0.0.1:9` if it appears in `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY`.
- `src/pdf_research_assistant/query_once.py` and `src/pdf_research_assistant/exhaustive.py` are internal helpers used by the app and CLI; they are not intended as separate user entry points.
- If you change `MANIFEST_PATH` to a narrower manifest while keeping the same `PAPER_DIR` and `INDEX_DIR`, the next rebuild or normal query removes the other papers from that index. Use a separate `INDEX_DIR` for a subset.
- If PaperQA reports that a PDF is empty but the file opens normally, it may be image-only and need OCR before it can be indexed.
- Answers cite specific pages from your PDFs when available.

## Feature Requests

Planned improvements are tracked in GitHub Issues.

The current implementation order is tracked in the pinned roadmap issue:
[Roadmap: Current Implementation Order](https://github.com/boardtc/pdf-research-assistant/issues/40)

If an issue matters to you, please add a thumbs-up reaction to the issue instead of commenting `+1`. That makes it easier to see which ideas are most useful to potential users.

- Browse open ideas: [GitHub Issues](https://github.com/boardtc/pdf-research-assistant/issues)
- Use a thumbs-up reaction to vote for an idea
- Add a comment if you have a specific use case or suggestion

## Adding New PDFs

1. Add the PDF under your configured `PAPER_DIR`.
2. If you are using a manifest, add a row to `manifest.csv`.
3. Rebuild the index with `pdf-research-rebuild` before querying again.
4. Start the UI or CLI and ask a question about your PDFs.
