import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from pdf_research_assistant import saved_settings, settings_groups


def _settings():
    return SimpleNamespace(answer=SimpleNamespace(evidence_k=10, answer_max_sources=5))


def test_load_returns_nothing_when_no_file_is_saved(tmp_path):
    assert saved_settings.load(tmp_path / "missing.json") == {}


def test_load_returns_nothing_when_the_file_is_not_json(tmp_path):
    path = tmp_path / "saved_settings.json"
    path.write_text("not json", encoding="utf-8")

    assert saved_settings.load(path) == {}


def test_load_returns_nothing_when_the_file_is_not_a_set_of_values(tmp_path):
    path = tmp_path / "saved_settings.json"
    path.write_text("[3, 2]", encoding="utf-8")

    assert saved_settings.load(path) == {}


def test_load_keeps_only_values_that_can_still_be_used(tmp_path):
    path = tmp_path / "saved_settings.json"
    path.write_text(
        json.dumps(
            {
                "evidence_k": 3,
                "answer_max_sources": 99,
                "answer_model": "some-model",
                "chunk_chars": 1000,
                "no_such_setting": 1,
            }
        ),
        encoding="utf-8",
    )

    assert saved_settings.load(path) == {"evidence_k": 3}


def test_save_value_writes_the_value_and_keeps_the_others(tmp_path):
    path = tmp_path / "saved_settings.json"
    saved_settings.save_value("evidence_k", 3, path)

    saved_settings.save_value("answer_max_sources", 2, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"evidence_k": 3, "answer_max_sources": 2}


def test_save_value_replaces_an_earlier_value(tmp_path):
    path = tmp_path / "saved_settings.json"
    saved_settings.save_value("evidence_k", 3, path)

    saved_settings.save_value("evidence_k", 20, path)

    assert saved_settings.load(path) == {"evidence_k": 20}


def test_save_value_rejects_a_value_outside_the_limits_and_saves_nothing(tmp_path):
    path = tmp_path / "saved_settings.json"

    with pytest.raises(ValueError):
        saved_settings.save_value("evidence_k", 0, path)

    assert not path.exists()


def test_save_value_rejects_a_setting_the_sidebar_cannot_change(tmp_path):
    path = tmp_path / "saved_settings.json"

    with pytest.raises(ValueError, match="chunk_chars"):
        saved_settings.save_value("chunk_chars", 1000, path)

    assert not path.exists()


def test_save_value_rejects_an_unknown_setting(tmp_path):
    with pytest.raises(KeyError):
        saved_settings.save_value("no_such_setting", 1, tmp_path / "saved_settings.json")


def test_forget_removes_only_the_named_values(tmp_path):
    path = tmp_path / "saved_settings.json"
    saved_settings.save_value("evidence_k", 3, path)
    saved_settings.save_value("answer_max_sources", 2, path)

    assert saved_settings.forget(["evidence_k"], path) is True
    assert saved_settings.load(path) == {"answer_max_sources": 2}


def test_forget_says_when_none_of_the_named_values_were_saved(tmp_path):
    path = tmp_path / "saved_settings.json"
    saved_settings.save_value("evidence_k", 3, path)

    assert saved_settings.forget(["answer_max_sources"], path) is False
    assert saved_settings.load(path) == {"evidence_k": 3}


def test_forget_says_when_nothing_was_saved(tmp_path):
    path = tmp_path / "saved_settings.json"

    assert saved_settings.forget(["evidence_k"], path) is False
    assert not path.exists()


def test_apply_writes_each_saved_value_onto_the_settings(tmp_path):
    path = tmp_path / "saved_settings.json"
    saved_settings.save_value("evidence_k", 3, path)
    settings = _settings()

    saved_settings.apply(settings, path)

    assert settings.answer.evidence_k == 3
    assert settings.answer.answer_max_sources == 5


def test_apply_leaves_the_settings_alone_when_nothing_is_saved(tmp_path):
    settings = _settings()

    saved_settings.apply(settings, tmp_path / "missing.json")

    assert settings.answer == _settings().answer


def test_paths_default_to_the_project_file(tmp_path):
    path = tmp_path / "saved_settings.json"
    with mock.patch.object(saved_settings, "SETTINGS_PATH", path):
        saved_settings.save_value("evidence_k", 4, None)
        settings = _settings()
        saved_settings.apply(settings)
        loaded = saved_settings.load()
        forgotten = saved_settings.forget(["evidence_k"])

    assert settings.answer.evidence_k == 4
    assert loaded == {"evidence_k": 4}
    assert forgotten is True


def test_the_saved_file_is_kept_out_of_git():
    gitignore = (Path(saved_settings.__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")

    assert saved_settings.SETTINGS_PATH.name in gitignore.splitlines()


@pytest.fixture
def real_bootstrap(tmp_path, monkeypatch):
    """Import the real bootstrap module against a temporary PDF folder, with PaperQA left as it is."""
    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setenv("PAPER_DIR", str(papers))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("MANIFEST_PATH", str(tmp_path / "manifest.csv"))
    sys.modules.pop("pdf_research_assistant.bootstrap", None)
    module = importlib.import_module("pdf_research_assistant.bootstrap")
    yield module
    sys.modules.pop("pdf_research_assistant.bootstrap", None)


def test_a_saved_value_reaches_the_settings_each_answer_records(real_bootstrap, tmp_path):
    """The query process builds its settings afresh, so the per-answer display shows the saved value."""
    path = tmp_path / "saved_settings.json"
    saved_settings.save_value("evidence_k", 3, path)
    saved_settings.save_value("answer_max_sources", 2, path)
    saved_settings.save_value("answer_length", "50 to 100 words", path)

    with mock.patch.object(saved_settings, "SETTINGS_PATH", path), \
            mock.patch.object(real_bootstrap.local_llm, "load_config", return_value=None):
        settings = real_bootstrap.build_settings()

    shown = {item["key"]: item["value"] for item in settings_groups.record_answer_settings(settings)["values"]}
    assert shown["evidence_k"] == "3"
    assert shown["answer_max_sources"] == "2"
    assert shown["answer_length"] == "50 to 100 words"
