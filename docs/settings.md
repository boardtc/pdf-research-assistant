# Settings and features reference

Every setting and part of the page, explained once. The [README](../README.md) says which of these to change for each task. Files, environment variables and the code layout are in [Files and internals](internals.md).

Each setting starts with a short list: the PaperQA setting it changes, its values and default, and what it does to the cost and time of a question.

- [How settings are saved](#how-settings-are-saved)
- Answer settings: [Search mode](#search-mode), [Quality preset](#quality-preset), [Passages gathered](#passages-gathered), [Most sources in the answer](#most-sources-in-the-answer), [Answer length](#answer-length), [Answer template](#answer-template), [Grounding](#grounding), [Before-answer prompt](#before-answer-prompt), [Speed limits](#speed-limits), [One-off settings](#one-off-settings)
- Library and index settings: [Saving them and the changed-settings warning](#library-and-index-settings), [Embedding model](#embedding-model), [Chunk size](#chunk-size), [Chunk overlap](#chunk-overlap), [Images and tables](#images-and-tables), [Whole-page pictures](#whole-page-pictures), [Settings you cannot change](#settings-you-cannot-change)
- Which PDFs are searched: [Paper list](#paper-list), [Search subfolders](#search-subfolders), [Saving the scope](#saving-the-scope)
- [Local LLM setup](#local-llm-setup)
- [Research presets](#research-presets), [Import and export](#import-and-export)
- Features: [Index name](#index-name), [Rebuild the index](#rebuild-the-index), [The next-question estimate](#the-next-question-estimate), [Answers](#answers), [Search previous answers](#search-previous-answers), [Exhaustive search](#exhaustive-search), [Command line](#command-line)

## How settings are saved

The sidebar has two groups of settings. The two groups are defined in `src/pdf_research_assistant/settings_groups.py`.

- **Applies to your next question**: the search mode, the quality preset, the passages gathered and most sources in the answer, the answer length, the answer template, the grounding mode, the before-answer prompt, the speed limits, the one-off settings panel and the local LLM setup. Change these whenever you like; the index is not rebuilt.
- **Needs an index rebuild**: the library and index settings: the PDF folder, embedding model, PDF reader, chunk size and overlap, whole-page pictures, and images and tables. Each is marked `Needs rebuild`, because changing any of them means every PDF has to be indexed again. This group matches the settings PaperQA uses to name an index.

A changed answer setting is saved straight away in `saved_settings.json` in the project folder, which Git ignores, and applies to the next question in the app and the CLI. **Go back to the default settings** removes every saved choice. The library and index settings and the scope are saved there too, but only when you click their save button.

Exhaustive search ignores the quality preset and the passage, source, length, template, grounding and before-answer prompt settings, because it checks every passage in scope and writes no combined answer, but it follows the speed limits.

## Answer settings

### Search mode

- Values: `Query` (default) or `Exhaustive`.
- Not a PaperQA setting: it picks which of the app's two ways of searching runs.
- Cost and time: Exhaustive checks every passage in scope, so it costs and takes far more. It shows its own estimate and asks before it spends anything.

`Query` is a normal PaperQA question. `Exhaustive` checks every chunk of every paper in scope: see [Exhaustive search](#exhaustive-search).

### Quality preset

- PaperQA settings: `answer.evidence_k`, `answer.answer_max_sources`, `answer.answer_length`, `answer.evidence_summary_length` and `agent.agent_type`.
- Values: Fast, Balanced (default), High quality, Debug. Shows as Custom once you change one of its numbers.
- Cost and time: High quality costs and takes about twice as long as Balanced. Fast usually takes about half as long.

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

### Passages gathered

- PaperQA setting: `answer.evidence_k`.
- Values: 1 to 30. Default 10 (PaperQA's default).
- Cost and time: the model reads each passage, so more passages cost more and take longer.

**Passages gathered** is how many passages are pulled from your PDFs and checked against the question. The model reads each one, so more passages cost more and take longer, but can find evidence that fewer would miss.

### Most sources in the answer

- PaperQA setting: `answer.answer_max_sources`.
- Values: 1 to 15, and never more than the passages gathered. Default 5 (PaperQA's default).
- Cost and time: more sources cost a little more.

**Most sources in the answer** is the largest number of the gathered passages the answer may cite. More sources cost a little more, because the model is sent more text to write the answer from.

### Answer length

- PaperQA setting: `answer.answer_length`.
- Values: Short, Standard (default), Long.
- Cost and time: a longer answer costs a little more and takes a little longer.

**Answer length** sets the phrase that tells the model how long to write:

- **Short**: "50 to 100 words", the phrase PaperQA's own fast preset uses.
- **Standard** (PaperQA default): "about 200 words, but can be longer".
- **Long**: "about 400 words, but can be longer".

A longer answer costs a little more and takes a little longer, because the model writes more, but the same passages are read either way.

### Answer template

- PaperQA setting: `prompts.qa`.
- Values: Direct answer (default), Study notes, Compare sources, Contradictions only.
- Cost and time: hardly change.

**Answer template** shapes the answer for your research task. Each template sets PaperQA's answer prompt, `prompts.qa`, which PaperQA uses to write the answer from the passages it found. A template is PaperQA's own prompt with its style sentence ("Write in the style of a scientific article...") swapped for an instruction of its own, so the passages, the question, the answer length and the citation rules stay as PaperQA wrote them:

- **Direct answer** (PaperQA default): PaperQA's own prompt, unchanged. A short, article-style reply to the question.
- **Study notes**: a one-sentence summary, then short bullet points under brief headings, covering the key terms, findings and methods.
- **Compare sources**: what each source finds or argues about the question in turn, then where they agree and where they differ.
- **Contradictions only**: only the points where the sources disagree, with what each side claims. If they do not disagree, the answer says so.

The same passages are found and read whichever template you pick, so the cost and time hardly change. The answer is not rewritten after PaperQA writes it, except that a quote not found in the passages loses its quote marks (see [Answers](#answers)).

### Grounding

- PaperQA settings: `prompts.system` and `answer.evidence_relevance_score_cutoff`.
- Values: Strictly grounded, Balanced (default), Exploratory.
- Cost and time: hardly change.

**Grounding** sets how closely the answer must keep to the passages PaperQA found. Each mode sets two PaperQA settings: the system prompt, `prompts.system`, which PaperQA sends with the request that writes the answer, and `answer.evidence_relevance_score_cutoff`, the lowest relevance score (out of 10) a passage needs to be sent to that request at all. A mode adds one instruction after PaperQA's own system prompt ("Answer in a direct and concise tone..."):

- **Strictly grounded**: the answer may use only what the passages say, with no facts or explanations from the model's own knowledge and every claim cited. If the passages do not directly answer the question, it replies "I cannot answer." Passages scored below 5 out of 10 are left out. Expect "I cannot answer" more often.
- **Balanced** (PaperQA default): PaperQA's own system prompt and its cutoff of 1. The answer is written from the passages and cites them.
- **Exploratory**: the answer may add widely accepted background knowledge to explain terms or connect the findings. Each sentence the passages do not support ends with `(Extra background information)` and has no citation. When the passages cover only part of the question, it answers that part and says what the sources do not cover. Same cutoff as Balanced.

Grounding leaves the answer prompt, `prompts.qa`, to the answer template, so any mode works with any template. The same passages are found and read in every mode, so the cost and time hardly change; Strictly grounded may send fewer of them to the answer.

### Before-answer prompt

- PaperQA setting: `prompts.pre`.
- Values: any text up to 2,000 characters, with `{question}` as the only value it can fill in. Default empty, which turns it off.
- Cost and time: one more call to the answer model, so a little more money and a few seconds more per question.

**Before-answer prompt** is optional text. When it is set, PaperQA sends it to the answer model, with `{question}` replaced by your question, before the answer is written, and adds the reply after the passages as "Extra background information" for the answer to use. For example:

```text
List the technical terms and abbreviations in this question and say in a few words what each means: {question}
```

It differs from the answer template: a template changes the request that writes the answer, while the before-answer prompt is a request of its own that runs first, on the question alone. So it costs one more call to the answer model, a little more money and a few seconds more per question, and the `Next question:` estimate counts it. It is sent with the same system prompt as the answer, so the grounding mode applies to it too. `{question}` is the only value it can fill in; write `{{` and `}}` for braces of your own. It can be up to 2,000 characters. Leave the box empty to turn it off. Because the reply is added to the passages, PaperQA writes an answer even when no passage was relevant, rather than saying it found nothing; with Strictly grounded the model is still told to reply "I cannot answer." when the passages do not answer the question.

PaperQA also has an after-answer prompt, `prompts.post`, which the app does not offer. In the PaperQA version the app uses (2026.3.18) it is filled in with the question only, never the answer, and its reply replaces the answer and is then added a second time, so the answer that was written is lost.

### Speed limits

- PaperQA settings: `rate_limit` (tokens a minute) and `request_limit` (requests a minute) in each model config, and `answer.max_concurrent_requests` (requests sent at once).
- Values: Low limits, Standard, High limits (default).
- Cost and time: no change to the cost. A profile that fits your account makes long runs as fast as they can be.

**Speed limits** sets how fast PaperQA may send requests to the model. Pick the profile that fits your OpenAI account's limits, which you can see at [platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits):

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

- Not a separate PaperQA setting: it sends other values of the settings above with one question.
- Cost and time: those of the values you pick.

To try other values on one question without changing your usual settings, open **Change for the next question only** and switch on **Use different settings for my next question**. The passage, source, length, template and grounding controls appear, starting at your usual values. Whatever you change there is sent with your next Query-mode question only: nothing is saved, and the switch turns itself off once the answer arrives. Exhaustive search ignores these values too.

## Library and index settings

Open **Library and index settings** in the **Needs an index rebuild** group to change how your PDFs are indexed:

| Setting | What it changes in PaperQA | Affects |
| --- | --- | --- |
| [Embedding model](#embedding-model) | `embedding`: `text-embedding-3-small` (default) or `text-embedding-3-large` | Indexing and questions |
| [Chunk size (characters)](#chunk-size) | `parsing.reader_config["chunk_chars"]`, from 2000 to 20000 characters (default 5000) | Indexing and questions |
| [Chunk overlap (characters)](#chunk-overlap) | `parsing.reader_config["overlap"]`, from 0 to 1750 characters (default 250) | Indexing only |
| [Whole-page pictures](#whole-page-pictures) | `parsing.reader_config["full_page"]` (default off) | Indexing and questions |
| [Images and tables](#images-and-tables) | `parsing.multimodal`: off (default), on without enrichment, or on with enrichment | Indexing and questions |

Each control's help says what the setting does and what it costs. Nothing changes until you click **Save index settings**. The choices are saved to `saved_settings.json`, so the command line, `pdf-research-rebuild` and exhaustive search use them too. **Go back to the default index settings** removes them.

When you save a change to one of these settings, the page warns that your settings no longer match the index, and lists each changed setting with its old and new value. To do this, the app writes the rebuild settings of the last index that holds PDFs to `last_index.json` in `INDEX_DIR`, whenever the page loads or `pdf-research-rebuild` finishes. The warning stays until you put the settings back, which brings back the old index, or build an index with the new settings, which **Rebuild the index** does from the page. Changing an answer setting never raises it.

### Embedding model

- PaperQA setting: `embedding`.
- Values: `text-embedding-3-small` (default) or `text-embedding-3-large`.
- Cost and time: indexing with the large model costs about six times as much. Each question is turned into numbers by the same model.

The OpenAI model that turns text into numbers so passages can be found by meaning. The large model can find passages the small one misses.

### Chunk size

- PaperQA setting: `parsing.reader_config["chunk_chars"]`.
- Values: 2000 to 20000 characters, in steps of 500. Default 5000 (PaperQA's default).
- Cost and time: the model reads whole passages, so larger chunks make each question cost more.

How long each stored passage of a PDF is. Smaller chunks make each passage more focused; larger ones keep more of the surrounding text together.

### Chunk overlap

- PaperQA setting: `parsing.reader_config["overlap"]`.
- Values: 0 to 1750 characters, in steps of 50. Default 250 (PaperQA's default).
- Cost and time: indexing only. More overlap means more passages to store for the same PDF.

How much text neighbouring passages share, so a sentence is not cut off between them.

### Images and tables

- PaperQA settings: `parsing.multimodal`, and `parsing.enrichment_llm` for enrichment.
- Values: Off (default), On without enrichment, On with enrichment.
- Cost and time: indexing is slower, and every question that finds a picture costs more. Enrichment adds a paid model call for every image and table while indexing.

Turn **Images and tables** on when the answers you need are in charts, diagrams, tables or slides rather than in the text. Leave it off for PDFs that are mostly text: it makes indexing slower and every question that finds a picture cost more, because the pictures are sent to the model with their passages. Enrichment is an extra paid model call (`parsing.enrichment_llm`, `gpt-4o` by default) for every image and table while indexing, so a search can find them by what they show; without it they are found only through the text around them.

Pictures are drawn at 150 dots per inch (`parsing.reader_config["dpi"]`, set in `bootstrap.PICTURE_DPI`), twice as sharp as PaperQA's default, which leaves the text in a table hard to read. PaperQA does not name an index after this, so it is not a sidebar setting. If a figure cannot be cut out of a page, the app reads that PDF again with whole-page pictures instead of dropping its pictures.

Reading images and tables needs the `pypdfium2` package, which is installed with this project (`paper-qa-pypdf[media]`). An older install may lack it; the app will not save the setting until it is installed.

Pictures make questions slower as well as dearer: see [Speed limits](#speed-limits) for how they count against the rate limiter, and [The next-question estimate](#the-next-question-estimate) for typical times.

### Whole-page pictures

- PaperQA setting: `parsing.reader_config["full_page"]`.
- Values: on or off. Default off. It does nothing while images and tables are off.
- Cost and time: whole-page pictures are larger, so passages that carry them cost more to read, and exhaustive search is much slower.

**Whole-page pictures** decides what is cut out of each page. With it off, PaperQA's PDF reader keeps each picture stored in the PDF (photos, scanned pages, pasted-in charts) and each table with ruled lines as a separate picture. Charts and diagrams drawn with lines and shapes, and tables without lines, are missed. With it on, every page is kept as one picture, which catches everything at the cost of a larger picture for every page. The sidebar says the same under the Images and tables control.

### Settings you cannot change

The PDF folder (`PAPER_DIR` in `.env`), the PDF reader and whether the index keeps full folder paths are shown in **Library and index settings** but cannot be changed there. They are part of PaperQA's index name, so they are listed with the others.

## Which PDFs are searched

Open **Which PDFs are searched** in the sidebar to choose the scope of the index in use.

### Paper list

- PaperQA setting: `manifest_file`.
- Values: **Only the PDFs in the paper list** (`manifest.csv`; the default when the file exists) or **All PDFs in the folder**. The control is switched off while there is no paper list.
- Cost and time: PDFs that come into scope are indexed when you click **Rebuild the index**, which costs money for embeddings.
- Affects indexing, questions and exhaustive search.

When it indexes a PDF, PaperQA takes the PDF's citation from its row in the paper list, word for word, and names the document after it the way it names one after a citation it reads (`Bretag (2020) ...` becomes `Bretag2020`). A PDF whose row has no citation, or that has no row, has its citation read from its first pages by the answer model instead. A PDF already in the index keeps the citation it was indexed with, so changing a citation in the paper list only affects PDFs indexed afterwards.

### Search subfolders

- PaperQA setting: `recurse_subdirectories`.
- Values: on (default) or off: whether PDFs in folders inside the PDF folder are searched.
- Cost and time: as for the paper list.
- Affects indexing, questions and exhaustive search.

### Saving the scope

Neither setting is part of PaperQA's index name, so changing one does not start a new index, and the [changed-settings warning](#library-and-index-settings) does not apply. PaperQA keeps the index and brings it into line with the new scope when you click **Rebuild the index**: it indexes the PDFs that came into scope, which costs money for embeddings, and removes the ones that left. Questions search the index as it is, so until that rebuild they still search the PDFs that left the scope and miss the ones that came into it. So these settings have a warning of their own, shown before anything is saved. When you click **Save scope** and the change would add or remove PDFs in an index that already holds some, the page says that it changes the index in use, gives how many PDFs the rebuild would add and remove and names the first few, and saves only when you click **Save the new scope**; **Cancel** saves nothing. PDFs that come into scope but are already in the index are listed separately, because they cost nothing. A change that adds or removes nothing, or one made while the index is empty, is saved at once.

After the scope is saved, the page counts only the PDFs in scope. It warns, and offers **Rebuild the index** on the page, while the PDF folder has PDFs in scope that the index does not list yet, or while the index holds PDFs that are no longer in scope (including PDFs deleted from the folder).

The choices are saved to `saved_settings.json`, so the page's counts, the rebuild, the command line, exhaustive search and the next-question estimate all use the same scope. They cannot be changed while a rebuild is running. Subfolders cannot be turned off while the paper list names PDFs only inside subfolders, because no PDF would be left to search. A paper list that names PDFs inside subfolders ignores those rows while subfolders are off.

Pointing `MANIFEST_PATH` at a narrower manifest has the same effect: see [Which PDFs are in scope](internals.md#which-pdfs-are-in-scope).

## Local LLM setup

- PaperQA settings: `llm`, `summary_llm` and `agent.agent_llm`, with their model configs, which the app points at your local server.
- Values: OpenAI's `gpt-4o-mini` (default), or a model on Ollama or another server that speaks the OpenAI API.
- Cost and time: no cost for the answer model, but each question still pays a fraction of a cent for its OpenAI embedding. Local models are usually slower.

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

## Research presets

**Research presets**, at the top of the sidebar, keeps the settings you use for a recurring piece of work under a name, so you can switch between workflows without setting everything again. The line under the heading names the preset that matches the settings in use, or says none does. Open **Save, use or change research presets** to:

- **Save** the settings in use: type a name under **Preset name** and click **Save preset**. A preset always holds the answer settings (quality preset, passages, sources, answer length, answer template, grounding and before-answer prompt). Tick **Include the library and index settings** to keep the embedding model, chunk size and overlap, whole-page pictures and images and tables too, and **Include the answer model** to keep whether answers come from OpenAI or the local model set up under **Local LLM setup**. Saving with the name of a saved preset replaces it.
- **Use** a saved preset: pick it under **Saved preset**, check the list of what it holds, and click **Use this preset**. Every setting it holds is put back. A setting you had not changed when you saved it goes back to PaperQA's default. The parts it does not include, and the speed limits, which follow your OpenAI account rather than the work, stay as they are.
- **Rename** a preset with **New name** and **Rename**, or click **Delete this preset**, which asks once more first (**Delete for good** or **Keep it**).

Presets are kept in `research_presets.json` in the project folder, which Git ignores. Each one holds your choices as they are saved in `saved_settings.json` and `local_llm.json`, so using a preset writes those files back, and the next question, the CLI and `pdf-research-rebuild` all follow it. A preset whose index settings differ from the ones in use means the PDFs have to be indexed again, so the panel says so before you use it, and the page then shows the [changed-settings warning](#library-and-index-settings) until you rebuild or switch back. Such a preset cannot be used while a rebuild is running. Change any setting afterwards and the preset no longer shows as in use.

### Import and export

To back presets up or use them on another computer, open **Import or export research presets**:

- **Export**: pick the presets under **Presets to export** (all of them to start with) and click **Export presets** to download `research-presets.json`. The file holds only the setting choices, the same PaperQA-backed values the presets keep: no API keys and no folder paths. A preset that uses a local model holds that server's address, which usually only works on this computer, so it is left out unless you tick **Include local model addresses**. Without it, such a preset keeps whatever answer model is in use where it is imported.
- **Import**: pick a file made with **Export presets** under **Presets file**. The whole file is checked before anything is saved: if it is not a presets file, or any preset in it has a name, setting or value this app cannot use, the panel lists every problem and imports nothing. Otherwise it lists the presets in the file and **Import presets** adds them to **Saved preset**. A preset with the same name as a saved one (ignoring capitals) is skipped unless you tick **Replace saved presets with the same name**. Importing only saves the presets; use one as usual to put its settings in place, which is when a preset with different index settings raises the changed-settings warning.

## Features

### Index name

Under the title, the page names the active index and the paper list in use. PaperQA names each index folder after a hash of its library and index settings, such as `pqa_index_1a2b...`, so to tell indexes apart, open **Active index details**, type a name in **Index name** and click **Save name**. The page then shows that name instead of the folder name: in the active index line, in the warning about changed settings, in the rebuild confirm step and in each answer's `Show settings used`. The folder name stays beside it, and the full folder path stays in **Active index details**. Names are saved to `index_names.json` in `INDEX_DIR`, keyed by folder name, so a name survives a restart and stays with its index: change a library or index setting and the page shows the other index's name, or its folder if it has none; put the setting back and the old name returns. Save an empty name to go back to the folder name. Two indexes cannot share a name.

### Rebuild the index

From the app, click **Rebuild the index**. The button appears at the top of the page whenever the index is out of date: it is empty, which includes having just saved new index settings, it is missing PDFs from the paper list, the PDF folder has PDFs it does not list yet, or some PDFs failed to index. It is always under **Library and index settings** in the sidebar as well, for when you have changed a PDF that is already indexed.

Questions never index PDFs themselves: they search the index as it is, so a PDF added to the folder is not searched until the next rebuild. The page says how many PDFs in the folder are not in the index yet, and a question asked before there is any index fails straight away, at no cost.

1. The page lists how many PDFs are in scope, the index that will be built, and a rough cost and time for the PDFs that will actually be indexed (see [The rebuild estimate](#the-rebuild-estimate)). PDFs already in the index are not counted.
2. Nothing starts until you click **Start paid rebuild**. **Cancel** leaves everything as it was.
3. The rebuild runs the same code as `pdf-research-rebuild`, in its own process, with the library and index settings saved in the sidebar. PDFs already in the index are left alone; new ones, and ones that failed before, are indexed.
4. A progress bar shows how many PDFs are done. Questions and the index settings are switched off until it finishes, and the output goes to `rebuild.log` in `INDEX_DIR`. The index counts and warnings at the top of the page are hidden meanwhile, because the rebuild is changing them.
5. **Stop rebuild**, under the progress bar, ends the rebuild early (see [Stopping a rebuild](#stopping-a-rebuild)).
6. When it finishes, the page shows what the rebuild really cost, the end of the log (**Show the end of the rebuild log**) and the new state of the index, and the warning about changed settings goes away. The cost is shown after a stopped or failed rebuild too, since the PDFs it started were paid for. It is also written at the end of `rebuild.log`, such as `Real cost: $0.0123 in all: embeddings $0.0021, describing images and tables $0.0102.`

The rebuild carries on if you close the page, and a reloaded page picks its progress up again. Stopping the Streamlit server does not stop it, but the page then no longer follows it.

If PaperQA reports that a PDF is empty but the file opens normally, it may be image-only and need OCR before it can be indexed.

On Windows, writing the search index now and then fails with `Access is denied` on one of its new files. The rebuild then lets go of the failed writer, adds the PDFs not yet saved to a new one, and tries again, up to 5 times; each retry is a `Writing the index failed; trying again` line in the log. If the build still stops early, the log ends with how many PDFs are not indexed yet, `pdf-research-rebuild` exits with code 4, and the page says how many of the PDFs the index holds. Rebuilding again carries on from there.

#### The rebuild estimate

The estimate makes no paid calls. It counts only the PDFs the rebuild will index: those the index does not list yet, and those that failed before and are still in scope.

- **Embeddings**: the app opens the PDFs locally, counts their pages and reads the text of the middle page of each, counting its tokens with the embedding model's own tokenizer, to work out how much text each PDF holds. PaperQA splits that text into passages of `chunk_chars` characters, each repeating the last `overlap` characters of the one before (see [Chunk size](#chunk-size)), and sends them all to the embedding model (`settings.embedding`). The cost is that many tokens at the model's price in LiteLLM's price list.
- **Citations**: a PDF with no citation in the paper list has its citation read from its first three pages by the answer model, in one call (see [Paper list](#paper-list)). The estimate counts a call for each PDF that has no row in the paper list, or a row without a citation. Without a paper list every PDF's citation is read; with a local model the calls cost nothing.
- **Images and tables**: with images and tables on with enrichment, `parsing.enrichment_llm` is called once for each picture, with the picture and the text of the pages around it. The pictures are not counted in the PDFs: they are guessed from a typical page, 0.7 per page, as for [the next-question estimate](#the-next-question-estimate) (or one per page with whole-page pictures). The page says that this part is a guess, and it is usually by far the largest part.
- **Time**: about 80 seconds to start the rebuild process and to save and close the index at the end, plus, for each PDF, the time the app took to read its text, about 4.5 seconds for PaperQA to read it, embed it and save it, the citation call, and one round of picture descriptions with enrichment. PaperQA indexes `concurrency` PDFs at a time. The times were fitted to a full rebuild of a library of 120 PDFs (3,163 pages), which took 16 minutes and cost 5 cents; the estimate for it had said about 6 cents. The time is rougher than the cost: it depends on OpenAI's speed and on how busy the computer is.

Reading every page of a large library takes minutes (about 70 ms a page on a library of 120 PDFs on a synced drive), so the app reads PDFs for at most 5 seconds, from across the whole library in the same order each time, and assumes the rest are like them. It spends most of that time counting pages, from the page count each PDF records (about 25 ms a PDF once the drive has fetched it, so a library of 120 PDFs is usually counted in full), because a library's few long books decide most of its length, and a sample that misses or catches one can be several times out. The rest goes on reading one page from as many PDFs as possible. The page then says how many of the PDFs the estimate was worked out from. A PDF that a synced drive must fetch first can take longer; the page waits at most 7 seconds and goes on with the PDFs read by then. The estimate is worked out once each time you click **Rebuild the index**.

#### Stopping a rebuild

Click **Stop rebuild** to end it early. This is a request, not a kill: the page writes a `rebuild.stop` file beside the log, and the rebuild process, which checks for it each time a PDF is about to start, starts no more PDFs. PDFs already under way (as many as PaperQA's index concurrency setting allows at once) are finished, and PaperQA then makes its usual closing save of the index, so nothing already paid for is lost. That is why stopping can take a minute or two, and the page says `Stopping` until it has ended. Afterwards the page says whether the index can still be used and how many PDFs it holds, by reading PaperQA's own file list (`files.zip`), and the rebuild log ends with how many PDFs were not started. A later rebuild carries on from there. The rebuild process is never killed, because PaperQA rewrites `files.zip` in place after committing the search index, and a kill in the middle of that write could leave the index damaged. A stop request still works if you close the page straight after clicking. If you end the rebuild process yourself, the page's note after the failed rebuild says whether the index can still be read.

#### From a terminal

```bash
cd ~/gitrepos/pdf-research-assistant
pdf-research-rebuild
```

Use the terminal command when:

- running the project for the first time and you want to build the index explicitly
- you add new PDFs and want to rebuild the index before querying again
- you want a terminal-only indexing run instead of using the app's button

It starts by printing the same [estimate](#the-rebuild-estimate) as the app's confirm step, under `Rough estimate before starting:`, then starts at once. On a clean rebuild, it is normal to see `Manifest PDFs: <n>` and `Indexed before run: 0` before indexing starts. At the end it prints `Indexed after run`, `Failed PDFs`, a `FAILED:` line for each PDF that could not be indexed, and what the rebuild really cost (`Real cost: ...`), counted from LiteLLM's cost tracking the same way questions count theirs, with embeddings and image and table descriptions given separately.

See [`windows-helper-commands.example.md`](../windows-helper-commands.example.md) for optional PowerShell commands that help check index build progress and troubleshoot rebuild issues on Windows. If you want a version with your own local paths ready to copy and paste, create `windows-helper-commands.md` from it.

### The next-question estimate

Just above the question box, `Next question:` gives a rough sense of what your next question will cost and how long it will take: `Fast`, `Balanced`, `Slower` or `Higher cost`, followed by a rough cost and time such as "roughly less than 1 cent, under half a minute". It is worked out from the settings the question will run with, including any values changed for the next question only, and changes as soon as you change a setting. Open **What this estimate is based on** for the details. It takes into account:

- how many passages are gathered: each one is read by the model, a batch at a time
- the most sources in the answer and the answer length: these set how much the answer model reads and writes
- the before-answer prompt: when one is set, it adds a call to the answer model before the answer
- the answer model: its price per token comes from LiteLLM's price list
- the speed limits profile: how many passages are read at once, and whether the token limit holds the question back
- images and tables, from the index settings: when they are on, each passage is sent with the pictures from its pages, and these often cost more than the text. The number of pictures is a guess from a typical page, since counting them would mean reading the index. Whole-page pictures send one larger picture for every page, and the picture resolution makes each one larger or smaller. With `gpt-4o-mini` and images and tables on, a question usually shows `Higher cost`. With cut-out pictures it usually takes under a minute; with whole-page pictures two or three minutes, because each whole-page picture uses a large share of the token limit and passages wait their turn. With a lower speed limits profile it can take far longer, because each picture counts against the token limit by its size (see [Speed limits](#speed-limits)).

It also counts about 3 seconds to start the question (PaperQA is loaded beforehand, while you type), and the short calls PaperQA's agent makes to plan the search, about eight on average. When the model plans the search, the agent gathers evidence about three times on average, reading the passages again each time, so that is counted in the time too. With PaperQA's defaults a question usually takes under half a minute; with the **Fast** preset, about 10 seconds. A question that needs many searches can take a minute or more, and so can any question while the computer is busy. It is a rough guide, not a quote: the real figures depend on the question and on how many searches PaperQA decides to make. With PaperQA's defaults and `gpt-4o-mini` it shows `Balanced`. With a local model there is no cost for the answer model, and the time, which is based on OpenAI's speed, may be well short. In Exhaustive mode it shows `Higher cost`, and the run's own estimate appears after you ask.

### Answers

Each question runs in a fresh helper process so repeated questions in the same session start with clean query state. The page starts that process before you ask, so PaperQA is already loaded when the question arrives. Answers cite specific pages from your PDFs when available. Each answer in the chat includes:

- a `Copy answer` button that copies the full answer text to the clipboard (Windows only)
- a `Show source passages` expander with the retrieved evidence passages used for the answer
- an `Open chunk` button beside each source passage, which opens the chunk PaperQA retrieved for it: the chunk text, whether the answer cited it, its relevance score, and any images or tables stored with it. Images and tables only appear when the index was built with the `Images and tables` setting turned on.
- a relevance score under each source passage, out of 10. This is PaperQA's `Context.score`: the rating the summary model gave the passage for your question while summarising it, not a search similarity. Scores tend to bunch at 0 or 8 to 9, so use them as a rough guide to which chunk to open. Answers given before scores were kept, and passages PaperQA did not score, show none.
- a `Show settings used` expander listing the index the question searched and the settings that shaped the answer: the answer model, the quality preset (or `Custom`), how many passages were gathered, the most sources the answer may cite, the answer length, the passage summary length, the search steps, the answer template, the grounding mode, the before-answer prompt (or `Off`), and the speed limits profile. The values are read from the settings the query actually ran with, so they stay correct for older answers after you change a setting. A value changed for that question only is marked `This question only`. Exhaustive-search answers do not have this list.
- the question's cost. This counts every call the question made, including the OpenAI embedding of the question that finds the passages, so a question answered by a local model still costs a fraction of a cent. A cost that would round to `$0.0000` shows as `less than $0.0001`, here, in **Search previous answers** and in **Total cost this session**. A question that fails shows what it spent before failing under the error, such as the embedding of the question or model calls made before a timeout, and that is added to **Total cost this session** too.

Text the answer puts in quote marks is checked against the full text of the chunks the question read. The answer model sometimes puts its own wording in quote marks, which would read as a direct quote from the PDF. A quote of four or more words that is not found word for word in any of those chunks loses its quote marks and is followed by `(paraphrase, not a direct quote)`, so the words stay but no longer look quoted. The check ignores case, spacing, line breaks, words hyphenated across lines, curly or straight quote marks and other punctuation; an ellipsis (`...`) or a `[bracketed insertion]` in a quote may stand for any text, as long as the parts either side are found in that order in one chunk. Shorter quotes, such as a quoted term, are left alone. The marked text is what the chat, `Copy answer`, the command line and **Search previous answers** show. A quote found in a chunk the question read but not the one the answer cites is still left as a quote. Exhaustive search checks its quotes its own way (see [Exhaustive search](#exhaustive-search)).

The answer model cites passages by PaperQA's ids, such as `pqac-0ed33f3f`, which PaperQA turns into source names like `(Lang2021 pages 2-4)` and lists under References. The model sometimes writes an id a character short or long, such as `pqac-0ed33f3`, which PaperQA does not recognise. Such an id is treated as the one passage whose id it starts (or, when too long, starts with): it shows as that passage's name, the passage is marked cited under `Show source passages`, and its source is listed under References. An id that matches no passage or several, or is under four characters after `pqac-`, is removed, as PaperQA removes made-up ids, and may leave an empty `()`.

If PaperQA stops without writing an answer, the reply says so instead of being blank: "No answer was written: PaperQA stopped before writing one", with PaperQA's own reason, and suggests trying again, asking another way, or gathering more passages.

The **Usage** part of the sidebar shows the number of questions and the total cost this session, and **Clear chat**.

### Search previous answers

Every answer to a Query-mode question, from the app or the CLI, is kept on disk so you can find it again after the chat is cleared or the app restarts. Open **Search previous answers** above the chat, type a few words under **Words to look for** and click **Search**. Up to 10 matching answers are listed, best match first, under the search box. Each shows its question, when it was asked, the index it used and its cost, then the answer with its **Show source passages** (including **Open chunk**) and **Show settings used**, just as it looked under the question. **Clear search results** puts them away.

The search looks for whole words in the questions and answers (so `paper` does not find `papers`), ranks answers that contain more of the words, or rarer ones, first, and understands `AND`, `OR` and `NOT` in capitals. It makes no API calls, and it does not change the PDF index, the chat or the next question.

Answers are kept in PaperQA's own index of past answers, the `answers` folder in `INDEX_DIR`, in the same form PaperQA's `pqa ask` uses, so `pqa search -i answers`, pointed at the same index folder, finds them too, and answers saved by `pqa ask` into that folder show up in the app (without a date, passages or settings, which only this app records). Answers from every index are kept together, and each says which index it came from. **Clear chat** does not remove them; delete the `answers` folder to start again, or delete one answer as below. The pictures shown under **Open chunk** are kept once each in `answers/pictures`, however many answers showed them, so answers from an index with whole-page pictures stay small; answers kept before this change carry their pictures inside them and still show them. If a picture has been deleted from that folder, **Open chunk** says it could not be shown. Exhaustive-search reports are kept in `exhaustive_runs/` instead and are not searched. If an answer cannot be kept, for example because the disk is full, it is still shown, with a line saying it could not be kept for searching. A save that fails for a moment, as can happen on Windows when another program briefly holds the index files, is tried again a few times first.

PaperQA names each kept answer's file after its text, so two answers with exactly the same text (such as `I cannot answer.`) share one file, and only the later one is found.

To delete one answer, such as a wrong, test or private one, click **Delete this answer** under it, then **Delete for good** (or **Keep it** to leave it). The answer is taken out of PaperQA's index of answers, so neither the app nor `pqa search -i answers` finds it again, and the search runs again straight away, so the results and the count of answers searched show what is kept now. Pictures kept only for that answer are removed from `answers/pictures`; pictures another kept answer shows stay. Answers with exactly the same text share one file, so they are deleted together, and the confirm step says how many other answers that is. A delete that fails, for example because Windows or another program is holding the index files, says why and leaves the answer kept; try again in a moment. Deleting makes no API calls and does not change the PDF index.

### Exhaustive search

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

A possible miss is a chunk scored 0 that names one of the question's topic words at least 5 times and is not a reference list. Words found in more than a quarter of the run's chunks don't count, because they are the subject of the whole collection rather than of the question: in a teaching collection, "teacher enthusiasm and student learning" checks only "enthusiasm". Runs of fewer than 20 chunks count every topic word. General question words such as "study", "research" and "literature" never count.

> **Important**
> Exhaustive search is more thorough than a normal query, but it can still misjudge relevance. A model scores each chunk, so it can reject something relevant or accept something that is not. On an index built without images and tables, scanned pages and figures are not searched; with them on, each chunk is sent with the pictures and tables from its pages, which the model may read less reliably than text. Quotes are marked verified or not verified, and a quote marked not verified may still be genuine, so check it against the PDF. Treat the report as a list to review, not a conclusion.

Exhaustive search reads the saved index and never builds it, so run `pdf-research-rebuild` first. Its speed comes from the **Speed limits** profile in the sidebar; pick a lower profile if your OpenAI account has lower limits. If the index was built with images and tables on, each passage is sent with its pictures, which count against the token limit by their size, so a run with whole-page pictures is much slower. The plan counts the pictures saved with each passage and includes them in its cost and time. Passages wait their turn for as long as it takes rather than failing and being listed as not checked.

### Command line

```bash
cd ~/gitrepos/pdf-research-assistant
pdf-research
```

Type a question at the prompt to search your indexed PDFs and return a cited answer with page references. Type `quit` to exit.

After each answer the CLI prints what the question cost, in the same format as the app: `Cost: $0.0012`, or `Cost: less than $0.0001` when a local model answered and only the embeddings were paid for. When you type `quit`, it prints how many questions you asked and their total cost:

```text
Question (or quit): What do the papers say about sleep and memory?

Sleep supports memory consolidation (Walker 2017 pages 3-5) ...
Cost: $0.0124

Question (or quit): quit

Questions asked this session: 1. Total cost: $0.0124
```

A question that fails prints its error, then a cost line with what it spent before failing, and that cost is added to the total. If the query helper crashes without reporting anything, there is no cost line and the question adds nothing to the total. If the answer could not be kept for searching later, a note under the cost line says why. With `--exhaustive`, each run reports its own actual cost instead, as described under [Exhaustive search](#exhaustive-search), and there is no session total.

The CLI uses the settings saved in the sidebar and the local LLM setup. Like the Streamlit app, it runs each question in a fresh helper process so repeated questions start with clean query state, started while you type the question. Each answer is kept for **Search previous answers** in the app.
