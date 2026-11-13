import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest


@pytest.fixture(scope="session", autouse=True)
def no_saved_local_llm_settings(tmp_path_factory):
    """Keep a developer's own local_llm.json out of the tests, which assume the hosted model by default."""
    from pdf_research_assistant import local_llm

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(local_llm, "CONFIG_PATH", tmp_path_factory.mktemp("local-llm") / "local_llm.json")
        yield

