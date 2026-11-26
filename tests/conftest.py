import os
import sys
from pathlib import Path


# Use the price list that comes with LiteLLM instead of downloading it on import. The download added
# 30 seconds or more to every run and made the tests depend on the network. Set before anything imports
# LiteLLM; processes the tests start inherit it. Set it to False first to test against the latest list.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for folder in (SRC_ROOT, PROJECT_ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import pytest

# The time limit on each test, fixtures included; see stuck_test_guard.py.
from stuck_test_guard import pytest_configure, pytest_runtest_protocol  # noqa: E402,F401


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

