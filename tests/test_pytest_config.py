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
