"""End the run, printing where every thread is, if one test runs past the time limit.

A stuck test, such as one waiting on a lock that never comes free, would otherwise hold the pre-push hook or a
CI run for hours. The limit covers each test's whole turn: setting up its fixtures, including module and
session fixtures the first test of a module or run waits for, the test itself, and tearing down. faulthandler's
own thread ends the process even when the test is stuck inside native code.

conftest.py loads these hooks for the project's tests; `-p stuck_test_guard` loads them for any other run.
"""

import contextlib
import faulthandler
import os
import sys

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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if TEST_TIME_LIMIT_SECONDS <= 0:
        return (yield)
    faulthandler.dump_traceback_later(TEST_TIME_LIMIT_SECONDS, exit=True, file=item.config._stuck_test_stderr)
    try:
        return (yield)
    finally:
        faulthandler.cancel_dump_traceback_later()
