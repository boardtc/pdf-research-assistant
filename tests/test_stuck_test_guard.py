import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

STUCK_MODULE_FIXTURE = """
import time
import pytest

@pytest.fixture(scope="module")
def stuck():
    time.sleep(60)
    yield

def test_uses_the_stuck_fixture(stuck):
    pass
"""

TWO_SLOW_TESTS = """
import time

def test_first():
    time.sleep(2.5)

def test_second():
    time.sleep(2.5)
"""


def run_guarded(folder: Path, test_source: str, limit: str):
    """Run pytest on one test file in its own process, with the stuck-test guard and the given limit."""
    (folder / "test_guarded.py").write_text(test_source, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(TESTS_DIR), "PDF_RESEARCH_ASSISTANT_TEST_TIME_LIMIT": limit}
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "stuck_test_guard", "-p", "no:cacheprovider", "test_guarded.py"],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
    )
    return result, time.monotonic() - started


def test_a_module_fixture_that_never_finishes_setting_up_ends_the_run(tmp_path):
    # The first test of a module waits for the module's fixtures, which the limit must cover too: a rebuild
    # in such a fixture once waited for a lock for over 13 hours.
    result, seconds = run_guarded(tmp_path, STUCK_MODULE_FIXTURE, "2")

    assert result.returncode != 0
    assert "Timeout" in result.stderr
    assert "stuck" in result.stderr
    assert seconds < 40


def test_the_limit_starts_again_for_each_test(tmp_path):
    # Together the two tests take longer than the limit; each alone takes well under it.
    result, _seconds = run_guarded(tmp_path, TWO_SLOW_TESTS, "4")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_a_limit_of_zero_turns_the_guard_off(tmp_path):
    result, _seconds = run_guarded(tmp_path, TWO_SLOW_TESTS, "0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
