# Testing

This project keeps test tooling separate from runtime dependencies so standard users do not need to install it.

## Install test dependencies

```powershell
python -m pip install -e .[dev]
```

This installs the package in editable mode together with the `dev` extras defined in `pyproject.toml`.

## Enable the pre-push test hook

This repository includes a versioned Git `pre-push` hook in `.githooks/pre-push` that runs the full test suite before each push.

Run this once in your local clone to tell Git to use the tracked hooks directory:

```powershell
git config core.hooksPath .githooks
```

After that, every `git push` from that clone will run:

```powershell
python -m pytest
```

If the tests fail, Git blocks the push. The full suite takes about two to three minutes on a Windows laptop (1 minute 48 seconds and 2 minutes 54 seconds in two runs of about 1,250 tests), so expect `git push` to sit at "Running test suite before push..." for that long. pytest prints the time on its last line.

Important:

- The hook file is tracked in the repository, but `core.hooksPath` is a local Git setting stored in `.git/config`.
- That means forks and fresh clones do not automatically enable the hook unless the contributor runs `git config core.hooksPath .githooks`.
- To enforce tests for everyone regardless of local hook setup, use CI as well.

## GitHub Actions CI

This repository also runs the test suite in GitHub Actions on every push and pull request.

The workflow lives at `.github/workflows/tests.yml` and currently runs:

```powershell
python -m pytest
```

on a `windows-latest` runner with Python 3.11.

This CI workflow complements the local `pre-push` hook:

- the local hook gives feedback before a push leaves your machine
- GitHub Actions provides a consistent remote check for pushes and pull requests
- this repository uses the local `pre-push` hook as the primary before-push safeguard
- branch protection can require the `test` status check if you choose a PR-based workflow later

CI installs the newest versions the project allows, so it can run a newer PaperQA model client (`fhlmi`) than your own install. A few tests check behaviour that differs between versions and are skipped where they do not apply, which is why a local run may report one skipped test.

## Run all tests

```powershell
python -m pytest
```

## Integration tests

`tests\test_integration_tiny_pdf.py` indexes a real one-page PDF (`tests\fixtures\tiny-enthusiasm.pdf`) through the rebuild entry point, then runs exhaustive search and a Fast-preset question on it, checks the answer is kept for **Search previous answers**, and loads the real Streamlit page with Streamlit's `AppTest` to check that sidebar changes stick between runs as they do in a browser. It makes no API calls and needs no key: PaperQA's local `sparse` embedding replaces OpenAI embeddings, and the model returns a fixed reply through LiteLLM's `mock_response`.

These tests are marked `integration` and run with the rest of the suite, including in the pre-push hook and CI. They take a minute or so, mostly loading PaperQA. For a quicker run, skip them:

```powershell
python -m pytest -m "not integration"
```

To change the fixture's text, edit and run `tests\fixtures\make_tiny_pdf.py`. It needs `reportlab`, which is not a project dependency.

## Run one test file

Each module in `src\pdf_research_assistant` has a test file of the same name, `tests\test_<module>.py`. For example:

```powershell
python -m pytest tests\test_query_once.py
```

## Run one test file with verbose output

```powershell
python -m pytest -v tests\test_app.py
```

To run only the tests whose names match some words:

```powershell
python -m pytest tests\test_app.py -k "research_preset"
```

## Run coverage for one file

Point `--cov` at the module the test file covers:

```powershell
python -m pytest tests\test_query_once.py --cov=pdf_research_assistant.query_once --cov-branch --cov-report=term-missing
```

```powershell
python -m pytest tests\test_exhaustive.py --cov=pdf_research_assistant.exhaustive --cov-branch --cov-report=term-missing
```

## Run coverage for the whole project

```powershell
python -m pytest --cov=pdf_research_assistant --cov-branch --cov-report=term-missing
```

## Current test strategy

- Unit tests use `pytest` as the test runner.
- Coverage reporting uses `pytest-cov`.
- Mocking uses Python's built-in `unittest.mock`.
- No test calls OpenAI or any other paid API.

### Test files

| File | What it covers |
| --- | --- |
| `tests/conftest.py` | Shared setup for every test (see below) |
| `tests/test_app.py` | The Streamlit page, with fake Streamlit widgets (see below) |
| `tests/test_cli.py` | The `pdf-research` command line |
| `tests/test_bootstrap.py` | `.env` handling, the manifest, the index and building PaperQA settings |
| `tests/test_pdf_parser.py` | Reading PDFs and the font-encoding fallback |
| `tests/test_query_once.py` | Answering one question, its cost, the rate limiter wait and the missing-answer note |
| `tests/test_query_client.py` | Starting the next question's process ahead of time and handing it the question |
| `tests/test_rebuild.py` | The `pdf-research-rebuild` command |
| `tests/test_rebuild_client.py` | Starting and following a rebuild from the app |
| `tests/test_exhaustive.py` | Exhaustive search runs |
| `tests/test_exhaustive_client.py` | Planning, starting and following exhaustive runs |
| `tests/test_settings_groups.py` | The sidebar settings, their groups and allowed values |
| `tests/test_saved_settings.py` | Saving and applying the sidebar choices |
| `tests/test_quality_presets.py` | The quality presets |
| `tests/test_answer_templates.py` | The answer templates |
| `tests/test_grounding_modes.py` | The grounding modes |
| `tests/test_prompt_hooks.py` | The before-answer prompt |
| `tests/test_rate_profiles.py` | The speed limits profiles and the rate limiter wait, against PaperQA's real limiter |
| `tests/test_estimate.py` | The next-question estimate, including its fit to timed real questions |
| `tests/test_costs.py` | How a cost is shown in dollars |
| `tests/test_research_presets.py` | Research presets, import and export |
| `tests/test_local_llm.py` | Local LLM settings and the local request timeout |
| `tests/test_index_names.py` | Friendly index names |
| `tests/test_index_record.py` | The record of the last built index's settings |
| `tests/test_saved_answers.py` | Keeping and searching previous answers, in a real PaperQA answers index |
| `tests/test_integration_tiny_pdf.py` | The integration tests above |

### Shared setup in `conftest.py`

`tests/conftest.py` applies to every test:

- It points `saved_settings.SETTINGS_PATH` and `local_llm.CONFIG_PATH` at files in a temporary folder for the whole run, so your own `saved_settings.json` and `local_llm.json` never change what the tests see, and the tests never write over them.
- It ends the whole run if any single test takes longer than 10 minutes, printing where every thread was, so a stuck test fails the pre-push hook or CI instead of holding it for hours. Set `PDF_RESEARCH_ASSISTANT_TEST_TIME_LIMIT` to a number of seconds to change the limit, or to `0` to turn it off (for example while stepping through a test in a debugger).

### The app test harness

`app.py` draws the page when it is imported, so `tests/test_app.py` imports it with a fake `streamlit` module and a fake `bootstrap` module in place (`imported_pdf_research_assistant`). Each fake widget hands back its starting value: a select box returns its first or `index` option, a number box its `value`, a text box its `value`, and so on. A test chooses other values by patching the widget it cares about.

Buttons work in two ways:

- **The click queue.** Buttons without a key prefix listed below take their answers, in the order they are drawn, from the `button_values` list a test passes in. A test that draws more of these buttons than the list holds fails with an `IndexError`, which usually means a new unkeyed button was added to the page.
- **Clicked by key.** Buttons whose key starts with `rebuild_`, `research_preset_` or `answer_search_` are clicked only when their key is in the test's `clicked_keys`. Buttons whose key starts with `local_llm_`, `open_chunk_`, `answer_settings_` or `index_settings_` are drawn on most pages and always stay unclicked here; their own tests draw just that part of the page with its own fake button.

So a new button should get a key with one of these prefixes, or a new prefix added to the list in `imported_pdf_research_assistant`, rather than taking a place in the click queue that other tests rely on.

The fake widgets do not keep a value between runs the way a browser does, so a bug that only shows on the second run of the page is missed here. The `AppTest` tests in `tests/test_integration_tiny_pdf.py` cover that for the sidebar.

The same harness also points `saved_settings.SETTINGS_PATH` and `research_presets.PRESETS_PATH` at files that do not exist, replaces `subprocess.run` so no question really runs, and replaces the rebuild and exhaustive-search clients so nothing is started.

### Other notes

- Exhaustive search tests make no real API calls. `tests/test_exhaustive.py` runs PaperQA's real `map_fxn_summary` with a fake summary model, so failure handling goes through the real code. They include a test that every in-scope paper appears in the results, and tests that each failure type is never reported as "no relevant evidence".
- `tests/test_saved_answers.py` writes to a real PaperQA answers index in a temporary folder. On Windows, saving to that index now and then fails with `Access is denied` while another process seems to hold one of its new files for a moment. The app tries a failed save again a few times, including a save that failed only after its answer was committed, so this should not fail a test. A test in this file that fails on a save with `Access is denied` points to a save path that is not retried, not to bad luck; run the file in a loop to catch it. Before #116 the save that finishes a committed answer had no retry; in 100 runs of `test_a_save_that_fails_after_committing_keeps_the_answer_once` it failed once.

## Checking exhaustive search quality

Passing tests show that failures are handled and every paper is reported. They cannot show that relevance judgments are right, because a model makes those. Exhaustive search is more thorough than a normal query, but it can still misjudge relevance.

To check quality, run a real question whose answers you already know, on a small set of papers, and record which known papers were found or missed, any false positives, any failed PDFs, and the runtime and cost. This makes paid API calls, so it is not part of the automated suite.

## Where to record results

- Keep test and coverage commands in this file.
- Put actual run results in pull requests, GitHub tickets, or CI logs rather than a checked-in results file.
- Do not maintain a permanent results document in the repo, because it will go stale quickly.
