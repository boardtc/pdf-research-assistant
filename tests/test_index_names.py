import json
from pathlib import Path
from unittest import mock

import pytest

from pdf_research_assistant import index_names


def _index_dir(tmp_path: Path, folder: str = "pqa_index_abc") -> Path:
    """Return an index folder inside an index root that exists, as it does once any index is built."""
    return tmp_path / folder


def _write_names(tmp_path: Path, text: str) -> None:
    (tmp_path / index_names.NAMES_FILE).write_text(text, encoding="utf-8")


def test_load_returns_nothing_when_no_names_were_saved(tmp_path):
    assert index_names.load(tmp_path) == {}


def test_load_returns_nothing_for_a_file_that_is_not_json(tmp_path):
    _write_names(tmp_path, "not json")

    assert index_names.load(tmp_path) == {}


def test_load_returns_nothing_when_the_file_is_not_a_mapping(tmp_path):
    _write_names(tmp_path, '["My library"]')

    assert index_names.load(tmp_path) == {}


def test_load_skips_entries_that_are_not_a_folder_and_a_name(tmp_path):
    _write_names(tmp_path, json.dumps({"pqa_index_abc": "My library", "pqa_index_def": 3, "pqa_index_ghi": "  "}))

    assert index_names.load(tmp_path) == {"pqa_index_abc": "My library"}


def test_get_name_finds_the_name_saved_for_this_folder(tmp_path):
    _write_names(tmp_path, json.dumps({"pqa_index_abc": "My library"}))

    assert index_names.get_name(_index_dir(tmp_path)) == "My library"
    assert index_names.get_name(_index_dir(tmp_path, "pqa_index_def")) is None


def test_clean_trims_and_joins_spaces(tmp_path):
    assert index_names.clean("  Large   embeddings\n test ") == "Large embeddings test"


def test_clean_refuses_a_name_that_is_too_long():
    with pytest.raises(ValueError, match=f"{index_names.MAX_LENGTH} characters"):
        index_names.clean("x" * (index_names.MAX_LENGTH + 1))


def test_clean_accepts_a_name_at_the_longest_allowed():
    assert index_names.clean("x" * index_names.MAX_LENGTH) == "x" * index_names.MAX_LENGTH


def test_save_name_writes_the_name_beside_the_index_folders_keyed_by_folder(tmp_path):
    index_names.save_name(_index_dir(tmp_path), " My library ")

    saved = json.loads((tmp_path / index_names.NAMES_FILE).read_text(encoding="utf-8"))
    assert saved == {"pqa_index_abc": "My library"}


def test_save_name_keeps_the_names_of_other_indexes(tmp_path):
    index_names.save_name(_index_dir(tmp_path), "My library")

    index_names.save_name(_index_dir(tmp_path, "pqa_index_def"), "Large embeddings")

    assert index_names.load(tmp_path) == {"pqa_index_abc": "My library", "pqa_index_def": "Large embeddings"}


def test_save_name_replaces_the_name_of_the_same_index(tmp_path):
    index_names.save_name(_index_dir(tmp_path), "My library")

    index_names.save_name(_index_dir(tmp_path), "Whole library")

    assert index_names.load(tmp_path) == {"pqa_index_abc": "Whole library"}


def test_save_name_with_an_empty_name_removes_the_name(tmp_path):
    index_names.save_name(_index_dir(tmp_path), "My library")
    index_names.save_name(_index_dir(tmp_path, "pqa_index_def"), "Large embeddings")

    index_names.save_name(_index_dir(tmp_path), "   ")

    assert index_names.load(tmp_path) == {"pqa_index_def": "Large embeddings"}


def test_save_name_with_an_empty_name_for_an_unnamed_index_changes_nothing(tmp_path):
    index_names.save_name(_index_dir(tmp_path), "")

    assert index_names.load(tmp_path) == {}


def test_save_name_refuses_a_name_another_index_already_has(tmp_path):
    index_names.save_name(_index_dir(tmp_path), "My library")

    with pytest.raises(ValueError, match="already the name of the index pqa_index_abc"):
        index_names.save_name(_index_dir(tmp_path, "pqa_index_def"), "my LIBRARY")

    assert index_names.load(tmp_path) == {"pqa_index_abc": "My library"}


def test_save_name_lets_an_index_keep_its_own_name_with_new_capitals(tmp_path):
    index_names.save_name(_index_dir(tmp_path), "My library")

    index_names.save_name(_index_dir(tmp_path), "My Library")

    assert index_names.load(tmp_path) == {"pqa_index_abc": "My Library"}


def test_save_name_refuses_a_name_that_is_too_long_and_saves_nothing(tmp_path):
    with pytest.raises(ValueError):
        index_names.save_name(_index_dir(tmp_path), "x" * (index_names.MAX_LENGTH + 1))

    assert not (tmp_path / index_names.NAMES_FILE).exists()


def test_save_name_reports_a_file_that_cannot_be_written(tmp_path):
    with mock.patch.object(Path, "write_text", side_effect=OSError("read-only")):
        with pytest.raises(OSError):
            index_names.save_name(_index_dir(tmp_path), "My library")
