# PDF Research Assistant

![Python 3.11%2B](https://img.shields.io/badge/python-3.11%2B-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![PaperQA2](https://img.shields.io/badge/PaperQA2-FutureHouse-orange)

A local PDF research assistant built with [PaperQA2](https://github.com/Future-House/paper-qa). It can use either a manifest-controlled document library or all PDFs under a chosen root folder. You ask questions in a Streamlit page or at the command line, and get cited answers with page references.

Core indexing and querying work on Windows, macOS, and Linux. The Streamlit `Copy answer` clipboard button is currently Windows-only.

![PDF Research Assistant screenshot](images/streamlit_app.jpg)

> **Important**
> PaperQA2 uses retrieval-augmented generation (RAG). It returns real source passages and page references, but it can still overstate, paraphrase, or extrapolate beyond what the source explicitly says. Treat answers as a starting point for exploration, not as a citable summary. Always verify important claims against the source passages and the original PDF.

More detail:

- [Settings and features reference](docs/settings.md): every setting, with its PaperQA name, values, default, and effect on cost and time.
- [Files and internals](docs/internals.md): the code layout, the files the app writes, environment variables and Windows notes.
- [TESTING.md](TESTING.md): the test suite.

## Getting started

You need Python 3.11 or newer and an OpenAI API key, which you can create at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

```bash
cd ~/gitrepos/pdf-research-assistant
python -m pip install -e .
```

This installs the project and the `pdf-research` and `pdf-research-rebuild` commands ([more about the commands](docs/internals.md#commands)). Then:

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` in `.env`.
3. Set `PAPER_DIR` in `.env` to the root folder containing your PDFs.
4. Optional: copy `manifest.example.csv` to `manifest.csv` if you want curated scope and metadata.
5. If you use `manifest.csv`, replace the example rows with paths relative to your chosen `PAPER_DIR`.

The other environment variables are in [Files and internals](docs/internals.md#environment-variables).

Start the app:

```bash
streamlit run src/pdf_research_assistant/app.py
```

Streamlit usually opens the app in your browser and prints the local URL, by default `http://localhost:8501` unless that port is already in use.

## A tour of the page

- **Under the title**: the active index and the paper list in use. **Active index details** lets you [name the index](docs/settings.md#index-name).
- **Sidebar, from the top**:
  - **Research presets**: named sets of settings ([reference](docs/settings.md#research-presets)).
  - **Applies to your next question**: the answer settings. Change them whenever you like; the index is not rebuilt ([reference](docs/settings.md#answer-settings)).
  - **Needs an index rebuild**: the library and index settings ([reference](docs/settings.md#library-and-index-settings)), then **Which PDFs are searched** ([reference](docs/settings.md#which-pdfs-are-searched)).
  - **Usage**: the questions and total cost this session, and **Clear chat**.
- **At the top of the page**: **Rebuild the index**, when the index is out of date ([reference](docs/settings.md#rebuild-the-index)).
- **Above the chat**: **Search previous answers** ([reference](docs/settings.md#search-previous-answers)).
- **Above the question box**: the `Next question:` cost and time estimate ([reference](docs/settings.md#the-next-question-estimate)).
- **Under each answer**: its source passages, the settings used and its cost ([reference](docs/settings.md#answers)).

## What you can do

### Ask a first question

1. Click **Rebuild the index**, check what will be paid for, and click **Start paid rebuild**. If you skip this, the first question builds the index, which can take a while for a large library ([reference](docs/settings.md#rebuild-the-index)).
2. Type a question in the box at the bottom of the page. The default settings suit a first question.
3. Open `Show source passages` and `Open chunk` under the answer to check what it was based on ([reference](docs/settings.md#answers)).

### Get quick, cheap answers

- [Quality preset](docs/settings.md#quality-preset): **Fast**, for about half the time of the default.
- [Passages gathered](docs/settings.md#passages-gathered): lower it, so fewer passages are read.
- [Answer length](docs/settings.md#answer-length): **Short**.
- [Images and tables](docs/settings.md#images-and-tables): leave it off unless you need pictures.

Watch the [`Next question:` estimate](docs/settings.md#the-next-question-estimate) change as you go.

### Get thorough answers

- [Quality preset](docs/settings.md#quality-preset): **High quality**, which reads twice as many passages.
- [Passages gathered](docs/settings.md#passages-gathered) and [Most sources in the answer](docs/settings.md#most-sources-in-the-answer): raise them to find and cite more evidence.
- [Answer length](docs/settings.md#answer-length): **Long**, for room to cover it.
- [Embedding model](docs/settings.md#embedding-model): `text-embedding-3-large` can find passages the small one misses. Needs a rebuild.

To try these on one question without changing your usual settings, use the [one-off settings](docs/settings.md#one-off-settings).

### Work with charts, tables and slides

- [Images and tables](docs/settings.md#images-and-tables): **On with enrichment**, so pictures are found by what they show, or **On without enrichment** to index them for less.
- [Whole-page pictures](docs/settings.md#whole-page-pictures): on, if the charts are drawn with lines and shapes or the tables have no ruled lines.
- [Speed limits](docs/settings.md#speed-limits): match your account, because pictures count heavily against the token limit.

Click **Save index settings**, then **Rebuild the index**. Questions cost more and take longer with pictures.

### Check every paper for an idea

A normal question can miss a relevant paper. [Exhaustive search](docs/settings.md#exhaustive-search) checks every chunk of every paper in scope instead.

- [Search mode](docs/settings.md#search-mode): **Exhaustive**.
- [Speed limits](docs/settings.md#speed-limits): the profile that fits your OpenAI account, which sets how long the run takes.
- [Which PDFs are searched](docs/settings.md#which-pdfs-are-searched): narrow the scope to cut the cost.

Check the estimate, click **Run paid search**, and review the report as a list to check, not a conclusion.

### Compare sources or find where they disagree

- [Answer template](docs/settings.md#answer-template): **Compare sources** for each source in turn, or **Contradictions only** for just the disagreements.
- [Most sources in the answer](docs/settings.md#most-sources-in-the-answer): raise it, so more sources can be compared.

### Keep answers strictly to the sources

- [Grounding](docs/settings.md#grounding): **Strictly grounded**. Expect "I cannot answer." more often.
- [Before-answer prompt](docs/settings.md#before-answer-prompt): leave it empty, because its reply is added to the passages.

For the opposite, background to explain terms, pick **Exploratory** grounding or set a before-answer prompt.

### Save a workflow and switch between workflows

- [Research presets](docs/settings.md#research-presets): set things up, then **Save preset** under a name. **Use this preset** puts it back later.
- [Import and export](docs/settings.md#import-and-export): to back presets up or move them to another computer.

### Find an old answer

- [Search previous answers](docs/settings.md#search-previous-answers): type a few words and click **Search**. It makes no API calls.

### Run answers on your own computer

- [Local LLM setup](docs/settings.md#local-llm-setup): point it at Ollama or another local server, **Check connection**, then **Use this local model**. You still need the OpenAI key for embeddings.
- [Quality preset](docs/settings.md#quality-preset): **Fast**, because small models often lose their way when they plan their own searches.
- [Speed limits](docs/settings.md#speed-limits): **Low limits**, so a slow computer gets fewer requests at once.

### Add PDFs and rebuild the index

1. Add the PDF under your configured `PAPER_DIR`.
2. If you are using a manifest, add a row to `manifest.csv`.
3. Rebuild the index with **Rebuild the index** in the app or `pdf-research-rebuild` before querying again ([reference](docs/settings.md#rebuild-the-index)).
4. Start the UI or CLI and ask a question about your PDFs.

To change which PDFs are searched without editing the manifest, use [Paper list](docs/settings.md#paper-list) and [Search subfolders](docs/settings.md#search-subfolders).

### Use the command line

- `pdf-research` asks questions at a prompt, with the settings saved in the sidebar ([reference](docs/settings.md#command-line)).
- `pdf-research --exhaustive` runs an [exhaustive search](docs/settings.md#exhaustive-search).
- `pdf-research-rebuild` builds the index ([reference](docs/settings.md#from-a-terminal)).

## Feature requests

Planned improvements and known bugs are tracked in [GitHub Issues](https://github.com/boardtc/pdf-research-assistant/issues). The features described here were built in the order set out in the [roadmap issue](https://github.com/boardtc/pdf-research-assistant/issues/40), which is now finished and closed.

If an issue matters to you, please add a thumbs-up reaction to the issue instead of commenting `+1`. That makes it easier to see which ideas are most useful to potential users.

- Browse open ideas: [GitHub Issues](https://github.com/boardtc/pdf-research-assistant/issues)
- Use a thumbs-up reaction to vote for an idea
- Add a comment if you have a specific use case or suggestion
