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

If the tests fail, Git blocks the push.

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

on a `windows-latest` runner.

This CI workflow complements the local `pre-push` hook:

- the local hook gives fast feedback before a push leaves your machine
- GitHub Actions provides a consistent remote check for pushes and pull requests
- this repository uses the local `pre-push` hook as the primary before-push safeguard
- branch protection can require the `test` status check if you choose a PR-based workflow later

## Run all tests

```powershell
python -m pytest
```

## Run one test file

```powershell
python -m pytest tests\test_query_once.py
```

```powershell
python -m pytest tests\test_bootstrap.py
```

```powershell
python -m pytest tests\test_app.py
```

```powershell
python -m pytest tests\test_cli.py
```

```powershell
python -m pytest tests\test_rebuild.py
```

```powershell
python -m pytest tests\test_exhaustive.py
```

```powershell
python -m pytest tests\test_exhaustive_client.py
```

## Run one test with verbose output

```powershell
python -m pytest -v tests\test_query_once.py
```

```powershell
python -m pytest -v tests\test_bootstrap.py
```

```powershell
python -m pytest -v tests\test_app.py
```

```powershell
python -m pytest -v tests\test_cli.py
```

```powershell
python -m pytest -v tests\test_rebuild.py
```

## Run coverage for one file

```powershell
python -m pytest tests\test_query_once.py --cov=pdf_research_assistant.query_once --cov-report=term-missing
```

```powershell
python -m pytest tests\test_bootstrap.py --cov=pdf_research_assistant.bootstrap --cov-report=term-missing
```

```powershell
python -m pytest tests\test_app.py --cov=pdf_research_assistant.app --cov-report=term-missing
```

```powershell
python -m pytest tests\test_cli.py --cov=pdf_research_assistant.cli --cov-report=term-missing
```

```powershell
python -m pytest tests\test_rebuild.py --cov=pdf_research_assistant.rebuild --cov-report=term-missing
```

```powershell
python -m pytest tests\test_exhaustive.py --cov=pdf_research_assistant.exhaustive --cov-branch --cov-report=term-missing
```

```powershell
python -m pytest tests\test_exhaustive_client.py --cov=pdf_research_assistant.exhaustive_client --cov-branch --cov-report=term-missing
```

## Run coverage for the whole project

```powershell
python -m pytest --cov=. --cov-report=term-missing
```

## Current test strategy

- Unit tests use `pytest` as the test runner.
- Coverage reporting uses `pytest-cov`.
- Mocking uses Python's built-in `unittest.mock`.
- Current unit test files are `tests/test_query_once.py`, `tests/test_bootstrap.py`, `tests/test_app.py`, `tests/test_cli.py`, `tests/test_rebuild.py`, `tests/test_exhaustive.py`, and `tests/test_exhaustive_client.py`.
- `src/pdf_research_assistant/query_once.py` and `src/pdf_research_assistant/bootstrap.py` have straightforward function-oriented tests.
- `src/pdf_research_assistant/app.py` includes heavier mocking because the Streamlit page executes UI logic at import time.
- Exhaustive search tests make no real API calls. `tests/test_exhaustive.py` runs PaperQA's real `map_fxn_summary` with a fake summary model, so failure handling goes through the real code. They include a test that every in-scope paper appears in the results, and tests that each failure type is never reported as "no relevant evidence".

## Checking exhaustive search quality

Passing tests show that failures are handled and every paper is reported. They cannot show that relevance judgments are right, because a model makes those. Exhaustive search is more thorough than a normal query, but it can still misjudge relevance.

To check quality, run a real question whose answers you already know, on a small set of papers, and record which known papers were found or missed, any false positives, any failed PDFs, and the runtime and cost. This makes paid API calls, so it is not part of the automated suite.

## Where to record results

- Keep test and coverage commands in this file.
- Put actual run results in pull requests, GitHub tickets, or CI logs rather than a checked-in results file.
- Do not maintain a permanent results document in the repo, because it will go stale quickly.
