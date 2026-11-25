import contextlib
import faulthandler
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest

# Longest any one test may run before the whole run is ended, in seconds; the slowest test takes well under
# a minute. Set PDF_RESEARCH_ASSISTANT_TEST_TIME_LIMIT to change it, or to 0 to turn the limit off.
TEST_TIME_LIMIT_SECONDS = int(os.environ.get("PDF_RESEARCH_ASSISTANT_TEST_TIME_LIMIT", "600"))


def pytest_configure(config):
    # A copy of the real stderr for the stacks printed when a test runs too long: during a test, pytest
    # sends stderr to a file of its own, which is lost when the run is ended.
    capture = config.pluginmanager.getplugin("capturemanager")
    with capture.global_and_fixture_disabled() if capture else contextlib.nullcontext():
        config._stuck_test_stderr = os.fdopen(os.dup(sys.__stderr__.fileno()), "w")


@pytest.fixture(autouse=True)
def end_a_stuck_test(pytestconfig):
    """End the run, printing where every thread is, if a test runs past the time limit.

    A stuck test, such as one waiting on a lock that never comes free, would otherwise hold the pre-push
    hook or a CI run for hours. faulthandler's own thread ends the process even when the test is stuck
    inside native code.
    """
    if TEST_TIME_LIMIT_SECONDS <= 0:
        yield
        return
    faulthandler.dump_traceback_later(TEST_TIME_LIMIT_SECONDS, exit=True, file=pytestconfig._stuck_test_stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="session", autouse=True)
def no_saved_local_llm_settings(tmp_path_factory):
    """Keep a developer's own local_llm.json out of the tests, which assume the hosted model by default."""
    from pdf_research_assistant import local_llm

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(local_llm, "CONFIG_PATH", tmp_path_factory.mktemp("local-llm") / "local_llm.json")
        yield


@pytest.fixture(scope="session", autouse=True)
def no_saved_sidebar_settings(tmp_path_factory):
    """Keep a developer's own saved_settings.json out of the tests, which assume the default settings."""
    from pdf_research_assistant import saved_settings

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(saved_settings, "SETTINGS_PATH", tmp_path_factory.mktemp("saved") / "saved_settings.json")
        yield

