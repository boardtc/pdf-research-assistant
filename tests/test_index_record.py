import json
from pathlib import Path
from unittest import mock

import pytest

from pdf_research_assistant import index_record, settings_groups


def _real_settings(paper_dir: Path):
    from paperqa import Settings
    from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

    return Settings(
        parsing=ParsingSettings(multimodal=False, use_doc_details=False),
        agent=AgentSettings(index=IndexSettings(paper_directory=paper_dir, index_directory=paper_dir / "index")),
    )


def _index_dir(settings) -> Path:
    """Return the active index folder; its parent exists, as it does once any index is built."""
    root = Path(settings.agent.index.index_directory)
    root.mkdir(parents=True, exist_ok=True)
    return root / settings.get_index_name()


def test_describe_index_settings_gives_each_index_setting_as_text(tmp_path):
    settings = _real_settings(tmp_path)

    values = index_record.describe_index_settings(settings)

    assert list(values) == [setting.key for setting in settings_groups.REINDEX_SETTINGS]
    assert values["chunk_chars"] == str(settings.parsing.reader_config["chunk_chars"])
    assert values["multimodal"] == "Off"


def test_remember_saves_the_index_name_and_its_settings_beside_the_index_folders(tmp_path):
    settings = _real_settings(tmp_path)
    index_dir = _index_dir(settings)

    assert index_record.remember(settings, index_dir) is True

    saved = json.loads((index_dir.parent / index_record.RECORD_NAME).read_text(encoding="utf-8"))
    assert saved == {"index": index_dir.name, "values": index_record.describe_index_settings(settings)}


def test_remember_leaves_the_file_alone_when_it_already_holds_this_index(tmp_path):
    settings = _real_settings(tmp_path)
    index_dir = _index_dir(settings)
    index_record.remember(settings, index_dir)

    with mock.patch.object(Path, "write_text") as write_text:
        assert index_record.remember(settings, index_dir) is False

    write_text.assert_not_called()


def test_remember_replaces_the_record_of_another_index(tmp_path):
    settings = _real_settings(tmp_path)
    old_dir = _index_dir(settings)
    index_record.remember(settings, old_dir)
    settings.parsing.reader_config["chunk_chars"] = 3000
    new_dir = _index_dir(settings)

    assert index_record.remember(settings, new_dir) is True

    assert index_record.load(new_dir)["index"] == new_dir.name


def test_remember_gives_up_quietly_when_the_record_cannot_be_written(tmp_path):
    settings = _real_settings(tmp_path)

    with mock.patch.object(Path, "write_text", side_effect=OSError("read-only")):
        assert index_record.remember(settings, _index_dir(settings)) is False


def test_load_gives_none_when_nothing_was_recorded(tmp_path):
    assert index_record.load(tmp_path / "index" / "some-index") is None


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[1, 2]",
        '{"values": {}}',
        '{"index": 3, "values": {}}',
        '{"index": "old"}',
        '{"index": "old", "values": [1]}',
    ],
)
def test_load_gives_none_when_the_record_is_unusable(tmp_path, content):
    (tmp_path / index_record.RECORD_NAME).write_text(content, encoding="utf-8")

    assert index_record.load(tmp_path / "some-index") is None


def test_find_changes_gives_none_when_no_index_was_recorded(tmp_path):
    settings = _real_settings(tmp_path)

    assert index_record.find_changes(settings, _index_dir(settings)) is None


def test_find_changes_gives_none_when_the_recorded_index_is_the_active_one(tmp_path):
    settings = _real_settings(tmp_path)
    index_dir = _index_dir(settings)
    index_record.remember(settings, index_dir)

    assert index_record.find_changes(settings, index_dir) is None


def test_find_changes_lists_each_index_setting_changed_since_the_recorded_index(tmp_path):
    settings = _real_settings(tmp_path)
    old_dir = _index_dir(settings)
    index_record.remember(settings, old_dir)
    old_chunk = settings.parsing.reader_config["chunk_chars"]
    settings.parsing.reader_config["chunk_chars"] = 3000
    settings.embedding = "some-other-embedding"

    changes = index_record.find_changes(settings, _index_dir(settings))

    assert changes == {
        "index": old_dir.name,
        "settings": [
            {"key": "embedding", "label": "Embedding model", "was": "text-embedding-3-small", "now": "some-other-embedding"},
            {"key": "chunk_chars", "label": "Chunk size (characters)", "was": str(old_chunk), "now": "3000"},
        ],
    }


def test_find_changes_gives_none_once_a_changed_setting_is_put_back(tmp_path):
    settings = _real_settings(tmp_path)
    index_dir = _index_dir(settings)
    index_record.remember(settings, index_dir)
    old_chunk = settings.parsing.reader_config["chunk_chars"]
    settings.parsing.reader_config["chunk_chars"] = 3000
    assert index_record.find_changes(settings, _index_dir(settings)) is not None

    settings.parsing.reader_config["chunk_chars"] = old_chunk

    assert index_record.find_changes(settings, _index_dir(settings)) is None


def test_find_changes_ignores_query_only_settings(tmp_path):
    settings = _real_settings(tmp_path)
    index_dir = _index_dir(settings)
    index_record.remember(settings, index_dir)

    settings.llm = "some-other-model"
    settings.answer.evidence_k = 3
    settings.answer.answer_length = "50 to 100 words"
    settings_groups.get_setting("rate_profile").write(settings, "Low limits")

    assert index_record.find_changes(settings, _index_dir(settings)) is None


def test_find_changes_skips_settings_the_record_does_not_hold(tmp_path):
    settings = _real_settings(tmp_path)
    values = index_record.describe_index_settings(settings) | {"chunk_chars": "1"}
    del values["embedding"]
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / index_record.RECORD_NAME).write_text(
        json.dumps({"index": "old-index", "values": values}), encoding="utf-8"
    )

    changes = index_record.find_changes(settings, _index_dir(settings))

    assert [item["key"] for item in changes["settings"]] == ["chunk_chars"]


def test_find_changes_still_warns_when_no_recorded_value_differs(tmp_path):
    settings = _real_settings(tmp_path)
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / index_record.RECORD_NAME).write_text(
        json.dumps({"index": "old-index", "values": index_record.describe_index_settings(settings)}),
        encoding="utf-8",
    )

    assert index_record.find_changes(settings, _index_dir(settings)) == {"index": "old-index", "settings": []}
