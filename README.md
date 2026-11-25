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
- `tests/` contains the test suite (see [TESTING.md](TESTING.md))
- `pyproject.toml` defines dependencies and the `pdf-research` and `pdf-research-rebuild` commands

The `pdf_research_assistant` folder under `src` is the importable package, so code uses imports such as `from pdf_research_assistant.bootstrap import build_settings`.

The modules in `src/pdf_research_assistant/`:

| Module | What it does |
| --- | --- |
| `app.py` | The Streamlit page |
| `cli.py` | The `pdf-research` command line |
| `rebuild.py` | The `pdf-research-rebuild` command, also run by the app's rebuild button |
| `rebuild_client.py` | Starts a rebuild from the app and follows its progress |
| `bootstrap.py` | Reads `.env`, finds the PDFs and the index, and builds the PaperQA settings everything else uses |
| `pdf_parser.py` | Reads PDFs, with a fallback for some PDFs whose fonts the usual reader cannot decode |
| `query_once.py` | Answers one question in its own process, for the app and the CLI |
| `exhaustive.py` | Exhaustive search: checks every chunk of every paper in scope, in its own process |
| `exhaustive_client.py` | Plans, starts and follows exhaustive runs for the app and the CLI |
| `settings_groups.py` | Lists every sidebar setting, which group it is in, and its allowed values |
| `saved_settings.py` | Saves the sidebar choices to `saved_settings.json` and applies them whenever settings are built |
| `quality_presets.py` | The quality presets |
| `answer_templates.py` | The answer templates |
| `grounding_modes.py` | The grounding modes |
| `prompt_hooks.py` | The before-answer prompt |
| `rate_profiles.py` | The speed limits profiles, and how long a request may wait for the rate limiter |
| `estimate.py` | The `Next question:` cost and time estimate |
| `research_presets.py` | Research presets, and their import and export |
| `local_llm.py` | The local LLM settings in `local_llm.json` |
| `index_names.py` | Friendly names for index folders |
| `index_record.py` | Remembers the settings of the last built index, for the changed-settings warning |
| `saved_answers.py` | Keeps each answer for **Search previous answers**, and searches them |

## Usage

### Streamlit UI

```bash
cd ~/gitrepos/pdf-research-assistant
streamlit run src/pdf_research_assistant/app.py
```

Streamlit usually opens the app in your browser automatically and prints the local URL in the terminal. By default, it uses `http://localhost:8501` unless that port is already in use.

On first use, there is no search index yet. Build it with **Rebuild the index** (see [Rebuild the index](#rebuild-the-index)) or `pdf-research-rebuild`; otherwise the first question builds it, which can take a while for a large PDF library.

Each question runs in a fresh helper process so repeated questions in the same session start with clean query state.

#### Naming the index

Under the title, the page names the active index and the paper list in use. PaperQA names each index folder after a hash of its library and index settings, such as `pqa_index_1a2b...`, so to tell indexes apart, open **Active index details**, type a name in **Index name** and click **Save name**. The page then shows that name instead of the folder name: in the active index line, in the warning about changed settings, in the rebuild confirm step and in each answer's `Show settings used`. The folder name stays beside it, and the full folder path stays in **Active index details**. Names are saved to `index_names.json` in `INDEX_DIR`, keyed by folder name, so a name survives a restart and stays with its index: change a library or index setting and the page shows the other index's name, or its folder if it has none; put the setting back and the old name returns. Save an empty name to go back to the folder name. Two indexes cannot share a name.

#### The sidebar

From the top, the sidebar has:

- **Research presets**: named sets of settings for a recurring piece of work (see [Research presets](#research-presets)).
- **Applies to your next question**: the search mode, the quality preset, the passages gathered and most sources in the answer, the answer length, the answer template, the grounding mode, the before-answer prompt, the speed limits, the one-off settings panel and the local LLM setup. Change these whenever you like; the index is not rebuilt.
- **Needs an index rebuild**: the library and index settings: the PDF folder, embedding model, PDF reader, chunk size and overlap, whole-page pictures, and images and tables. Each is marked `Needs rebuild`, because changing any of them means every PDF has to be indexed again.
- **Usage**: the number of questions and the total cost this session, and **Clear chat**.

The two settings groups are defined in `src/pdf_research_assistant/settings_groups.py`. The rebuild group matches the settings PaperQA uses to name an index.

### Answer settings

These settings are in the **Applies to your next question** group. A changed setting is saved straight away in `saved_settings.json` in the project folder, which Git ignores, and applies to the next question in the app and the CLI. **Go back to the default settings** removes every saved choice.

Exhaustive search ignores the quality preset and the passage, source, length, template, grounding and before-answer prompt settings, because it checks every passage in scope and writes no combined answer, but it follows the speed limits.

#### Quality preset

**Quality preset** is a quick way to set how thoroughly PaperQA searches and answers. Each preset follows one of PaperQA's own bundled presets (in `paperqa/configs`) or its defaults, and sets five PaperQA settings: `answer.evidence_k`, `answer.answer_max_sources`, `answer.answer_length`, `answer.evidence_summary_length` (how long each passage's summary is) and `agent.agent_type` (whether the model plans each search or PaperQA runs fixed steps):

| Preset | Passages | Sources | Answer | Passage summary | Search steps | Follows |
|---|---|---|---|---|---|---|
| **Fast** | 5 | 3 | 50 to 100 words | 25 to 50 words | Fixed steps | `fast.json` |
| **Balanced** (PaperQA default) | 10 | 5 | about 200 words | about 100 words | Planned by the model | PaperQA's defaults |
| **High quality** | 20 | 5 | about 200 words | about 100 words | Planned by the model | `high_quality.json` |
| **Debug** | 2 | 2 | 50 to 100 words | 25 to 50 words | Planned by the model | `debug.json` |

With fixed steps PaperQA asks the answer model for a few searches, then searches, gathers evidence and answers in that order and makes one call to finish, instead of having the model choose each step; that saves several model calls, and the passages are read once rather than each time the model gathers more evidence, so a question usually takes about half as long. High quality reads twice as many passages as Balanced, so it costs and takes roughly twice as long. Debug is the smallest real run, for checking that the app and your PDFs work. The `Next question:` estimate follows the preset you pick.

The bundled presets also change settings that live elsewhere: how many requests are sent at once belongs to **Speed limits**, the chunk size and document details in `high_quality.json` would need an index rebuild, and `debug.json`'s Anthropic model would replace the answer model. Those are left out, and so are the answer template, grounding and before-answer prompt, so a preset combines with any of them.

Picking a preset replaces any passages, sources or answer length you set before. You can still change those afterwards: your change is used on top of the preset, and the preset then shows as **Custom**.

#### Passages gathered and most sources

Two numbers control how much evidence goes into an answer. They are PaperQA's `answer.evidence_k` and `answer.answer_max_sources` settings:

- **Passages gathered** (PaperQA default 10, from 1 to 30): how many passages are pulled from your PDFs and checked against the question. The model reads each one, so more passages cost more and take longer, but can find evidence that fewer would miss.
- **Most sources in the answer** (PaperQA default 5, from 1 to 15, and never more than the passages gathered): the largest number of those passages the answer may cite. More sources cost a little more, because the model is sent more text to write the answer from.

#### Answer length

**Answer length** sets PaperQA's `answer.answer_length`, the phrase that tells the model how long to write:

- **Short**: "50 to 100 words", the phrase PaperQA's own fast preset uses.
- **Standard** (PaperQA default): "about 200 words, but can be longer".
- **Long**: "about 400 words, but can be longer".

A longer answer costs a little more and takes a little longer, because the model writes more, but the same passages are read either way.

#### Answer template

**Answer template** shapes the answer for your research task. Each template sets PaperQA's answer prompt, `prompts.qa`, which PaperQA uses to write the answer from the passages it found. A template is PaperQA's own prompt with its style sentence ("Write in the style of a scientific article...") swapped for an instruction of its own, so the passages, the question, the answer length and the citation rules stay as PaperQA wrote them:

- **Direct answer** (PaperQA default): PaperQA's own prompt, unchanged. A short, article-style reply to the question.
- **Study notes**: a one-sentence summary, then short bullet points under brief headings, covering the key terms, findings and methods.
- **Compare sources**: what each source finds or argues about the question in turn, then where they agree and where they differ.
- **Contradictions only**: only the points where the sources disagree, with what each side claims. If they do not disagree, the answer says so.

The same passages are found and read whichever template you pick, so the cost and time hardly change. The answer is never rewritten after PaperQA writes it.

#### Grounding

**Grounding** sets how closely the answer must keep to the passages PaperQA found. Each mode sets two PaperQA settings: the system prompt, `prompts.system`, which PaperQA sends with the request that writes the answer, and `answer.evidence_relevance_score_cutoff`, the lowest relevance score (out of 10) a passage needs to be sent to that request at all. A mode adds one instruction after PaperQA's own system prompt ("Answer in a direct and concise tone..."):

- **Strictly grounded**: the answer may use only what the passages say, with no facts or explanations from the model's own knowledge and every claim cited. If the passages do not directly answer the question, it replies "I cannot answer." Passages scored below 5 out of 10 are left out. Expect "I cannot answer" more often.
- **Balanced** (PaperQA default): PaperQA's own system prompt and its cutoff of 1. The answer is written from the passages and cites them.
- **Exploratory**: the answer may add widely accepted background knowledge to explain terms or connect the findings. Each sentence the passages do not support ends with `(Extra background information)` and has no citation. When the passages cover only part of the question, it answers that part and says what the sources do not cover. Same cutoff as Balanced.

Grounding leaves the answer prompt, `prompts.qa`, to the answer template, so any mode works with any template. The same passages are found and read in every mode, so the cost and time hardly change; Strictly grounded may send fewer of them to the answer.

#### Before-answer prompt

**Before-answer prompt** is optional text that sets PaperQA's `prompts.pre`. When it is set, PaperQA sends it to the answer model, with `{question}` replaced by your question, before the answer is written, and adds the reply after the passages as "Extra background information" for the answer to use. For example:

```text
List the technical terms and abbreviations in this question and say in a few words what each means: {question}
```

It differs from the answer template: a template changes the request that writes the answer, while the before-answer prompt is a request of its own that runs first, on the question alone. So it costs one more call to the answer model, a little more money and a few seconds more per question, and the `Next question:` estimate counts it. It is sent with the same system prompt as the answer, so the grounding mode applies to it too. `{question}` is the only value it can fill in; write `{{` and `}}` for braces of your own. It can be up to 2,000 characters. Leave the box empty to turn it off. Because the reply is added to the passages, PaperQA writes an answer even when no passage was relevant, rather than saying it found nothing; with Strictly grounded the model is still told to reply "I cannot answer." when the passages do not answer the question.

PaperQA also has an after-answer prompt, `prompts.post`, which the app does not offer. In the PaperQA version the app uses (2026.3.18) it is filled in with the question only, never the answer, and its reply replaces the answer and is then added a second time, so the answer that was written is lost.

#### Speed limits

**Speed limits** sets how fast PaperQA may send requests to the model: the `rate_limit` (tokens a minute) and `request_limit` (requests a minute) in each model config, and `answer.max_concurrent_requests` (requests sent at once). Pick the profile that fits your OpenAI account's limits, which you can see at [platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits):

| Profile | Tokens a minute | Requests a minute | At once (question) | At once (exhaustive) |
|---|---|---|---|---|
| Low limits | 30,000 | 500 | 4 | 4 |
| Standard | 200,000 | 500 | 5 | 10 |
| High limits (default) | 1,500,000 | 4,000 | 10 | 40 |

- **Low limits** suits a new or low-tier account. Ordinary questions are fine, but exhaustive search can take hours.
- **Standard** uses PaperQA's own figures for `gpt-4o-mini` on OpenAI's first paid tier.
- **High limits** suits an account allowed about 2,000,000 tokens and 5,000 requests a minute, with some room to spare.

A profile above your account's limits makes OpenAI turn requests away, and PaperQA waits and retries, which is slower. A profile below them makes long runs slower than they need to be. The profile does not change the cost of a question. With a local model there are no account limits, so only the number of requests sent at once matters.

PaperQA's rate limiter counts each request's length, and pictures are counted by their size as text, so with images and tables on a single passage can use a large share of a minute's tokens. On its own the rate limiter fails a request that has waited 60 seconds for its turn. The app lets a question's requests wait for as long as the question may run (`agent.timeout`, 500 seconds by default), and exhaustive search's requests wait for as long as it takes, so passages wait their turn instead of failing.

### One-off settings

To try other values on one question without changing your usual settings, open **Change for the next question only** and switch on **Use different settings for my next question**. The passage, source, length, template and grounding controls appear, starting at your usual values. Whatever you change there is sent with your next Query-mode question only: nothing is saved, and the switch turns itself off once the answer arrives. Exhaustive search ignores these values too.

### Research presets

**Research presets**, at the top of the sidebar, keeps the settings you use for a recurring piece of work under a name, so you can switch between workflows without setting everything again. The line under the heading names the preset that matches the settings in use, or says none does. Open **Save, use or change research presets** to:

- **Save** the settings in use: type a name under **Preset name** and click **Save preset**. A preset always holds the answer settings (quality preset, passages, sources, answer length, answer template, grounding and before-answer prompt). Tick **Include the library and index settings** to keep the embedding model, chunk size and overlap, whole-page pictures and images and tables too, and **Include the answer model** to keep whether answers come from OpenAI or the local model set up under **Local LLM setup**. Saving with the name of a saved preset replaces it.
- **Use** a saved preset: pick it under **Saved preset**, check the list of what it holds, and click **Use this preset**. Every setting it holds is put back. A setting you had not changed when you saved it goes back to PaperQA's default. The parts it does not include, and the speed limits, which follow your OpenAI account rather than the work, stay as they are.
- **Rename** a preset with **New name** and **Rename**, or click **Delete this preset**, which asks once more first (**Delete for good** or **Keep it**).

Presets are kept in `research_presets.json` in the project folder, which Git ignores. Each one holds your choices as they are saved in `saved_settings.json` and `local_llm.json`, so using a preset writes those files back, and the next question, the CLI and `pdf-research-rebuild` all follow it. A preset whose index settings differ from the ones in use means the PDFs have to be indexed again, so the panel says so before you use it, and the page then shows the changed-settings warning (see [Library and index settings](#library-and-index-settings)) until you rebuild or switch back. Such a preset cannot be used while a rebuild is running. Change any setting afterwards and the preset no longer shows as in use.

#### Import and export

To back presets up or use them on another computer, open **Import or export research presets**:

- **Export**: pick the presets under **Presets to export** (all of them to start with) and click **Export presets** to download `research-presets.json`. The file holds only the setting choices, the same PaperQA-backed values the presets keep: no API keys and no folder paths. A preset that uses a local model holds that server's address, which usually only works on this computer, so it is left out unless you tick **Include local model addresses**. Without it, such a preset keeps whatever answer model is in use where it is imported.
- **Import**: pick a file made with **Export presets** under **Presets file**. The whole file is checked before anything is saved: if it is not a presets file, or any preset in it has a name, setting or value this app cannot use, the panel lists every problem and imports nothing. Otherwise it lists the presets in the file and **Import presets** adds them to **Saved preset**. A preset with the same name as a saved one (ignoring capitals) is skipped unless you tick **Replace saved presets with the same name**. Importing only saves the presets; use one as usual to put its settings in place, which is when a preset with different index settings raises the changed-settings warning.

### Library and index settings

Open **Library and index settings** in the **Needs an index rebuild** group to change how your PDFs are indexed:

| Setting | What it changes in PaperQA | Affects |
| --- | --- | --- |
| Embedding model | `embedding`: `text-embedding-3-small` (default) or `text-embedding-3-large` | Indexing and questions |
| Chunk size (characters) | `parsing.reader_config["chunk_chars"]`, from 2000 to 20000 characters (default 5000) | Indexing and questions |
| Chunk overlap (characters) | `parsing.reader_config["overlap"]`, from 0 to 1750 characters (default 250) | Indexing only |
| Whole-page pictures | `parsing.reader_config["full_page"]` (default off) | Indexing and questions |
| Images and tables | `parsing.multimodal`: off (default), on without enrichment, or on with enrichment | Indexing and questions |

Each control's help says what the setting does and what it costs. Nothing changes until you click **Save index settings**. The choices are saved to `saved_settings.json`, so the command line, `pdf-research-rebuild` and exhaustive search use them too. **Go back to the default index settings** removes them. The PDF folder (`PAPER_DIR` in `.env`), the PDF reader and whether the index keeps full folder paths are shown but cannot be changed there.

When you save a change to one of these settings, the page warns that your settings no longer match the index, and lists each changed setting with its old and new value. To do this, the app writes the rebuild settings of the last index that holds PDFs to `last_index.json` in `INDEX_DIR`, whenever the page loads or `pdf-research-rebuild` finishes. The warning stays until you put the settings back, which brings back the old index, or build an index with the new settings, which **Rebuild the index** does from the page. Changing an answer setting never raises it.

### Images and tables

Turn **Images and tables** on when the answers you need are in charts, diagrams, tables or slides rather than in the text. Leave it off for PDFs that are mostly text: it makes indexing slower and every question that finds a picture cost more, because the pictures are sent to the model with their passages. Enrichment is an extra paid model call (`parsing.enrichment_llm`, `gpt-4o` by default) for every image and table while indexing, so a search can find them by what they show; without it they are found only through the text around them.

**Whole-page pictures** decides what is cut out of each page. With it off, PaperQA's PDF reader keeps each picture stored in the PDF (photos, scanned pages, pasted-in charts) and each table with ruled lines as a separate picture. Charts and diagrams drawn with lines and shapes, and tables without lines, are missed. With it on, every page is kept as one picture, which catches everything at the cost of a larger picture for every page. The sidebar says the same under the Images and tables control.

Pictures are drawn at 150 dots per inch (`parsing.reader_config["dpi"]`, set in `bootstrap.PICTURE_DPI`), twice as sharp as PaperQA's default, which leaves the text in a table hard to read. PaperQA does not name an index after this, so it is not a sidebar setting. If a figure cannot be cut out of a page, the app reads that PDF again with whole-page pictures instead of dropping its pictures.

Reading images and tables needs the `pypdfium2` package, which is installed with this project (`paper-qa-pypdf[media]`). An older install may lack it; the app will not save the setting until it is installed.

Pictures make questions slower as well as dearer: see [Speed limits](#speed-limits) for how they count against the rate limiter, and [The next-question estimate](#the-next-question-estimate) for typical times.

### Rebuild the index

From the app, click **Rebuild the index**. The button appears at the top of the page whenever the index is out of date: it is empty, which includes having just saved new index settings, it is missing PDFs from the paper list, or some PDFs failed to index. It is always under **Library and index settings** in the sidebar as well, for when you have added PDFs.

1. The page lists how many PDFs are in scope, the index that will be built and what is paid for: the embedding model and, with images and tables on with enrichment, a model call for every image and table. It gives no cost figure, because that depends on the text in the PDFs.
2. Nothing starts until you click **Start paid rebuild**. **Cancel** leaves everything as it was.
3. The rebuild runs the same code as `pdf-research-rebuild`, in its own process, with the library and index settings saved in the sidebar. PDFs already in the index are left alone; new ones, and ones that failed before, are indexed.
4. A progress bar shows how many PDFs are done. Questions and the index settings are switched off until it finishes, and the output goes to `rebuild.log` in `INDEX_DIR`.
5. When it finishes, the page shows the end of the log (**Show the end of the rebuild log**) and the new state of the index, and the warning about changed settings goes away.

The rebuild carries on if you close the page, and a reloaded page picks its progress up again. Stopping the Streamlit server does not stop it, but the page then no longer follows it. There is no stop button; PaperQA saves the index after each PDF, so a rebuild that is interrupted carries on from there next time.

From a terminal:

```bash
cd ~/gitrepos/pdf-research-assistant
pdf-research-rebuild
```

Use the terminal command when:

- running the project for the first time and you want to build the index explicitly
- you add new PDFs and want to rebuild the index before querying again
- you want a terminal-only indexing run instead of letting the first query build the index

On a clean rebuild, it is normal to see `Manifest PDFs: <n>` and `Indexed before run: 0` before indexing starts. At the end it prints `Indexed after run`, `Failed PDFs` and a `FAILED:` line for each PDF that could not be indexed.

See `windows-helper-commands.example.md` for optional PowerShell commands that help check index build progress and troubleshoot rebuild issues on Windows. If you want a version with your own local paths ready to copy and paste, create `windows-helper-commands.md` from it.

### The next-question estimate

Just above the question box, `Next question:` gives a rough sense of what your next question will cost and how long it will take: `Fast`, `Balanced`, `Slower` or `Higher cost`, followed by a rough cost and time such as "roughly less than 1 cent, up to about 2 minutes". It is worked out from the settings the question will run with, including any values changed for the next question only, and changes as soon as you change a setting. Open **What this estimate is based on** for the details. It takes into account:

- how many passages are gathered: each one is read by the model, a batch at a time
- the most sources in the answer and the answer length: these set how much the answer model reads and writes
- the before-answer prompt: when one is set, it adds a call to the answer model before the answer
- the answer model: its price per token comes from LiteLLM's price list
- the speed limits profile: how many passages are read at once, and whether the token limit holds the question back
- images and tables, from the index settings: when they are on, each passage is sent with the pictures from its pages, and these often cost more than the text. The number of pictures is a guess from a typical page, since counting them would mean reading the index. Whole-page pictures send one larger picture for every page, and the picture resolution makes each one larger or smaller. With `gpt-4o-mini` and images and tables on, a question usually shows `Higher cost` and takes three or four minutes. With a lower speed limits profile it can take far longer, because each picture counts against the token limit by its size (see [Speed limits](#speed-limits)).

It also counts about 16 seconds to start the question and load PaperQA, and the short calls PaperQA's agent makes to plan the search, usually about eight. When the model plans the search, the agent usually gathers evidence about three times, reading the passages again each time, so that is counted in the time too. With PaperQA's defaults a question usually takes a minute or more; with the **Fast** preset, about half a minute. It is a rough guide, not a quote: the real figures depend on the question and on how many searches PaperQA decides to make. With PaperQA's defaults and `gpt-4o-mini` it shows `Balanced`. With a local model there is no cost for the answer model, and the time, which is based on OpenAI's speed, may be well short. In Exhaustive mode it shows `Higher cost`, and the run's own estimate appears after you ask.

### Answers

Each answer in the chat includes:

- a `Copy answer` button that copies the full answer text to the clipboard (Windows only)
- a `Show source passages` expander with the retrieved evidence passages used for the answer
- an `Open chunk` button beside each source passage, which opens the chunk PaperQA retrieved for it: the chunk text, whether the answer cited it, its relevance score, and any images or tables stored with it. Images and tables only appear when the index was built with the `Images and tables` setting turned on.
- a relevance score under each source passage, out of 10. This is PaperQA's `Context.score`: the rating the summary model gave the passage for your question while summarising it, not a search similarity. Scores tend to bunch at 0 or 8 to 9, so use them as a rough guide to which chunk to open. Answers given before scores were kept, and passages PaperQA did not score, show none.
- a `Show settings used` expander listing the index the question searched and the settings that shaped the answer: the answer model, the quality preset (or `Custom`), how many passages were gathered, the most sources the answer may cite, the answer length, the passage summary length, the search steps, the answer template, the grounding mode, the before-answer prompt (or `Off`), and the speed limits profile. The values are read from the settings the query actually ran with, so they stay correct for older answers after you change a setting. A value changed for that question only is marked `This question only`. Exhaustive-search answers do not have this list.
- the question's cost. This counts every call the question made, including the OpenAI embedding of the question that finds the passages, so a question answered by a local model still costs a fraction of a cent. A cost that would round to `$0.0000` shows as `less than $0.0001`, here, in **Search previous answers** and in **Total cost this session**.

If PaperQA stops without writing an answer, the reply says so instead of being blank: "No answer was written: PaperQA stopped before writing one", with PaperQA's own reason, and suggests trying again, asking another way, or gathering more passages.

### Search previous answers

Every answer to a Query-mode question, from the app or the CLI, is kept on disk so you can find it again after the chat is cleared or the app restarts. Open **Search previous answers** above the chat, type a few words under **Words to look for** and click **Search**. Up to 10 matching answers are listed, best match first, under the search box. Each shows its question, when it was asked, the index it used and its cost, then the answer with its **Show source passages** (including **Open chunk**) and **Show settings used**, just as it looked under the question. **Clear search results** puts them away.

The search looks for whole words in the questions and answers (so `paper` does not find `papers`), ranks answers that contain more of the words, or rarer ones, first, and understands `AND`, `OR` and `NOT` in capitals. It makes no API calls, and it does not change the PDF index, the chat or the next question.

Answers are kept in PaperQA's own index of past answers, the `answers` folder in `INDEX_DIR`, in the same form PaperQA's `pqa ask` uses, so `pqa search -i answers`, pointed at the same index folder, finds them too, and answers saved by `pqa ask` into that folder show up in the app (without a date, passages or settings, which only this app records). Answers from every index are kept together, and each says which index it came from. **Clear chat** does not remove them; delete the `answers` folder to start again. The pictures shown under **Open chunk** are kept once each in `answers/pictures`, however many answers showed them, so answers from an index with whole-page pictures stay small; answers kept before this change carry their pictures inside them and still show them. If a picture has been deleted from that folder, **Open chunk** says it could not be shown. Exhaustive-search reports are kept in `exhaustive_runs/` instead and are not searched. If an answer cannot be kept, for example because the disk is full, it is still shown, with a line saying it could not be kept for searching. A save that fails for a moment, as can happen on Windows when another program briefly holds the index files, is tried again a few times first.

PaperQA names each kept answer's file after its text, so two answers with exactly the same text (such as `I cannot answer.`) share one file, and only the later one is found.

### CLI

```bash
cd ~/gitrepos/pdf-research-assistant
pdf-research
```

Type a question at the prompt to search your indexed PDFs and return a cited answer with page references. Type `quit` to exit.

The CLI uses the settings saved in the sidebar and the local LLM setup. Like the Streamlit app, it runs each question in a fresh helper process so repeated questions start with clean query state. Each answer is kept for **Search previous answers** in the app.

### Exhaustive Search

Normal queries are search-first: PaperQA picks candidate papers, ranks the top chunks, and caps the sources in the answer, so a relevant paper can be missed without warning. Exhaustive search checks every chunk of every paper in scope instead, which suits searching for ideas across a whole library.

Query mode stays the default. To use exhaustive search:

- In the Streamlit UI, choose `Exhaustive` under `Search mode` in the sidebar.
- In the CLI, run `pdf-research --exhaustive`.

Before each run you see the paper count, any papers that cannot be searched, the estimated cost, and the estimated time. Nothing is spent until you confirm: click **Run paid search** in the app, or type `yes` in the CLI. Afterwards you see the actual cost and runtime, and the app offers **Download report** and **Show full report**.

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
> Exhaustive search is more thorough than a normal query, but it can still misjudge relevance. A model scores each chunk, so it can reject something relevant or accept something that is not. On an index built without images and tables, scanned pages and figures are not searched; with them on, each chunk is sent with the pictures and tables from its pages, which the model may read less reliably than text. Quotes are marked verified or not verified, and a quote marked not verified may still be genuine, so check it against the PDF. Treat the report as a list to review, not a conclusion.

Exhaustive search reads the saved index and never builds it, so run `pdf-research-rebuild` first. Its speed comes from the **Speed limits** profile in the sidebar; pick a lower profile if your OpenAI account has lower limits. If the index was built with images and tables on, each passage is sent with its pictures, which count against the token limit by their size, so a run with whole-page pictures is much slower. The plan counts the pictures saved with each passage and includes them in its cost and time. Passages wait their turn for as long as it takes rather than failing and being listed as not checked.

### Local LLM

By default, answers are written by OpenAI's `gpt-4o-mini`. To use a model running on your own computer instead:

1. Install and start [Ollama](https://ollama.com), or another local server that speaks the OpenAI API, such as LM Studio or llama.cpp.
2. Download a model that supports tool calling, for example `ollama pull llama3.2`.
3. In the app's sidebar, open **Local LLM setup**, choose the server type, and enter the server address and model name.
4. Click **Check connection**. The app asks the server which models it has and tells you whether your settings are ready.
5. Click **Use this local model**. Settings are only saved if the check passes.

**Switch back to OpenAI** in the same panel removes the saved settings.

Each box and button in the panel has a **?** icon that explains it in plain words, and the panel lists how a local model differs from OpenAI.

The settings are saved in `local_llm.json` in the project folder, which Git ignores, and apply to the app, the CLI and exhaustive search.

Each request to a local model may take up to 10 minutes (OpenAI's are cut off after 60 seconds), and a failed request is retried up to 3 times. A slow computer can take minutes over one request when several are sent at once, and a shorter limit would cancel them and leave nothing to answer from.

> **Note**
> Only the model that reads passages and writes answers changes. Finding passages still uses OpenAI embeddings, so `OPENAI_API_KEY` is still needed, each question costs a fraction of a cent for its embedding, and switching does not rebuild your index. Local models are usually slower than OpenAI, and exhaustive search shows no cost estimate for them.

> **Tip**
> Small models, such as the 3B `llama3.2`, often lose their way when they plan their own searches: the answer says there are no papers, or no answer is written. If that happens, choose the **Fast** quality preset, which runs PaperQA's steps in a fixed order, and **Low limits**, which sends fewer requests at once. On a laptop without a graphics card, `llama3.2` gave a cited answer this way in about ten minutes, while the model-planned **Balanced** preset did not.

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

## Files the App Writes

These are created as you use the app. The ones in the project folder are ignored by Git.

| File | Where | What it holds |
| --- | --- | --- |
| `saved_settings.json` | project folder | The answer settings and the library and index settings chosen in the sidebar |
| `local_llm.json` | project folder | The local LLM setup |
| `research_presets.json` | project folder | Research presets |
| `exhaustive_runs/` | project folder | One folder per exhaustive search run |
| `index_names.json` | `INDEX_DIR` | Friendly index names |
| `last_index.json` | `INDEX_DIR` | The settings of the last built index, for the changed-settings warning |
| `rebuild.log` | `INDEX_DIR` | Output of the last rebuild started from the app |
| `answers/` | `INDEX_DIR` | Kept answers, for **Search previous answers**, with the pictures shown with them in `answers/pictures` |

## Additional Notes

- The search index is stored in the folder set by `INDEX_DIR`.
- If `manifest.csv` exists, the app uses it to decide which PDFs are in scope and which metadata to use.
- If `manifest.csv` does not exist, the app indexes all PDFs under `PAPER_DIR`.
- `windows-helper-commands.example.md` is a safe-to-share template. If you want a version with your own local paths ready to copy and paste, create `windows-helper-commands.md` from it.
- The app automatically ignores the broken loopback proxy placeholder `127.0.0.1:9` if it appears in `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY`.
- `query_once.py`, `exhaustive.py` and the `*_client.py` modules are internal helpers used by the app and CLI; they are not intended as separate user entry points.
- If you change `MANIFEST_PATH` to a narrower manifest while keeping the same `PAPER_DIR` and `INDEX_DIR`, the next rebuild or normal query removes the other papers from that index. Use a separate `INDEX_DIR` for a subset.
- If PaperQA reports that a PDF is empty but the file opens normally, it may be image-only and need OCR before it can be indexed.
- Answers cite specific pages from your PDFs when available.

## Feature Requests

Planned improvements and known bugs are tracked in [GitHub Issues](https://github.com/boardtc/pdf-research-assistant/issues). The features described above were built in the order set out in the [roadmap issue](https://github.com/boardtc/pdf-research-assistant/issues/40), which is now finished and closed.

If an issue matters to you, please add a thumbs-up reaction to the issue instead of commenting `+1`. That makes it easier to see which ideas are most useful to potential users.

- Browse open ideas: [GitHub Issues](https://github.com/boardtc/pdf-research-assistant/issues)
- Use a thumbs-up reaction to vote for an idea
- Add a comment if you have a specific use case or suggestion

## Adding New PDFs

1. Add the PDF under your configured `PAPER_DIR`.
2. If you are using a manifest, add a row to `manifest.csv`.
3. Rebuild the index with **Rebuild the index** in the app or `pdf-research-rebuild` before querying again.
4. Start the UI or CLI and ask a question about your PDFs.
