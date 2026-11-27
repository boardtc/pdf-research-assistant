# Files and internals

How the project is laid out, the files it writes, and the environment variables it reads. For what each setting does, see the [settings reference](settings.md). For the tests, see [TESTING.md](../TESTING.md).

## Commands

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

## Project layout

This project uses a standard Python `src` layout.

- `src/pdf_research_assistant/` contains the application code
- `tests/` contains the test suite (see [TESTING.md](../TESTING.md))
- `docs/` contains this file and the [settings reference](settings.md)
- `pyproject.toml` defines dependencies and the `pdf-research` and `pdf-research-rebuild` commands

The `pdf_research_assistant` folder under `src` is the importable package, so code uses imports such as `from pdf_research_assistant.bootstrap import build_settings`.

The modules in `src/pdf_research_assistant/`:

| Module | What it does |
| --- | --- |
| `app.py` | The Streamlit page |
| `cli.py` | The `pdf-research` command line |
| `rebuild.py` | The `pdf-research-rebuild` command, also run by the app's rebuild button |
| `rebuild_client.py` | Starts a rebuild from the app and follows its progress |
| `rebuild_estimate.py` | The rough cost and time of a rebuild, from a quick local read of the PDFs still to index |
| `bootstrap.py` | Reads `.env`, finds the PDFs and the index, and builds the PaperQA settings everything else uses |
| `paper_list.py` | Hands PaperQA the paper list's rows when it indexes PDFs, keyed by each path as PaperQA looks it up (with backslashes on Windows) and keeping each row's citation as written |
| `pdf_parser.py` | Reads PDFs, with a fallback for some PDFs whose fonts the usual reader cannot decode, and a second reading of ProQuest Ebook Central chapters whose italic words and citation years the usual reader moves to the end of the line |
| `query_once.py` | Answers one question in its own process, for the app and the CLI |
| `quote_check.py` | Checks that text an answer puts in quote marks is word for word in the passages, and unquotes it when it is not |
| `query_client.py` | Starts the next question's process ahead of time, so PaperQA is already loaded when the question is asked |
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
| `costs.py` | Shows a cost in dollars the same way in the app and the CLI |
| `research_presets.py` | Research presets, and their import and export |
| `local_llm.py` | The local LLM settings in `local_llm.json` |
| `index_names.py` | Friendly names for index folders |
| `index_record.py` | Remembers the settings of the last built index, for the changed-settings warning |
| `scope_change.py` | Works out which PDFs a change to the paper list or **Search subfolders** would add to or remove from the index, for the warning before the scope is saved |
| `saved_answers.py` | Keeps each answer for **Search previous answers**, searches them, and deletes one with the pictures only it shows |

`query_once.py`, `exhaustive.py` and the `*_client.py` modules are internal helpers used by the app and CLI; they are not intended as separate user entry points.

## Environment variables

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

The app automatically ignores the broken loopback proxy placeholder `127.0.0.1:9` if it appears in `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY`.

## Which PDFs are in scope

- The search index is stored in the folder set by `INDEX_DIR`.
- If `manifest.csv` exists, the app uses it to decide which PDFs are in scope and which metadata to use, unless you choose **All PDFs in the folder** under [Which PDFs are searched](settings.md#which-pdfs-are-searched).
- If `manifest.csv` does not exist, the app indexes all PDFs under `PAPER_DIR`, subfolders included unless you turn **Search subfolders** off.
- If you change `MANIFEST_PATH` to a narrower manifest while keeping the same `PAPER_DIR` and `INDEX_DIR`, the next rebuild removes the other papers from that index (questions search the index as it is, so they keep finding them until then). Use a separate `INDEX_DIR` for a subset.

## Files the app writes

These are created as you use the app. The ones in the project folder are ignored by Git.

| File | Where | What it holds |
| --- | --- | --- |
| `saved_settings.json` | project folder | The answer settings, the library and index settings and the scope (paper list, subfolders) chosen in the sidebar |
| `local_llm.json` | project folder | The local LLM setup |
| `research_presets.json` | project folder | Research presets |
| `exhaustive_runs/` | project folder | One folder per exhaustive search run |
| `index_names.json` | `INDEX_DIR` | Friendly index names |
| `last_index.json` | `INDEX_DIR` | The settings of the last built index, for the changed-settings warning |
| `rebuild.log` | `INDEX_DIR` | Output of the last rebuild started from the app |
| `rebuild.stop` | `INDEX_DIR` | Written by **Stop rebuild** to ask a running rebuild to stop |
| `rebuild.cost.json` | `INDEX_DIR` | What the last rebuild started from the app really cost, by kind of call |
| `answers/` | `INDEX_DIR` | Kept answers, for **Search previous answers**, with the pictures shown with them in `answers/pictures` |

## Windows notes

- The Streamlit `Copy answer` clipboard button is currently Windows-only. Core indexing and querying work on Windows, macOS, and Linux.
- [`windows-helper-commands.example.md`](../windows-helper-commands.example.md) is a safe-to-share template of optional PowerShell commands for checking index build progress and troubleshooting rebuilds. If you want a version with your own local paths ready to copy and paste, create `windows-helper-commands.md` from it.
