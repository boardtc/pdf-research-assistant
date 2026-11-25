import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collect(root):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout + result.stderr


def test_a_bare_pytest_run_collects_only_the_projects_own_tests(tmp_path):
    # The pre-push hook runs a bare `python -m pytest` from the project root. Copies of the
    # project in folders such as .claude/worktrees must not be collected with it.
    shutil.copy(PROJECT_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_own.py").write_text("def test_own():\n    pass\n")
    stray = tmp_path / ".claude" / "worktrees" / "copy" / "tests"
    stray.mkdir(parents=True)
    (stray / "test_stray.py").write_text("def test_stray():\n    pass\n")

    output = _collect(tmp_path)

    assert "test_own.py::test_own" in output
    assert "test_stray" not in output


def test_the_tests_use_litellms_own_copy_of_its_price_list():
    # Without this, importing LiteLLM downloads its price list first, which added 30 seconds or more to
    # every test run, and made the run depend on the network.
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"
