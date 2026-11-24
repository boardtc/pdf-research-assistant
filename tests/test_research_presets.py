import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from paperqa import Settings

from pdf_research_assistant import local_llm, research_presets, saved_settings, settings_groups

OLLAMA = {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3"}
SAVED = {
    "quality_preset": "Fast",
    "evidence_k": 12,
    "answer_template": "Study notes",
    "grounding": "Strictly grounded",
    "pre_prompt": "Explain the terms in {question}.",
    "rate_profile": "Low limits",
    "chunk_chars": 3000,
    "multimodal": "on_with_enrichment",
}
ANSWER_PART = {
    "quality_preset": "Fast",
    "evidence_k": 12,
    "answer_template": "Study notes",
    "grounding": "Strictly grounded",
    "pre_prompt": "Explain the terms in {question}.",
}
INDEX_PART = {"chunk_chars": 3000, "multimodal": "on_with_enrichment"}


def _write(path: Path, values) -> Path:
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


# What a preset holds.


def test_answer_settings_are_every_saved_instant_setting_except_speed_limits():
    expected = [
        setting.key
        for setting in settings_groups.INSTANT_SETTINGS
        if setting.write is not None and setting.key != "rate_profile"
    ]

    assert list(research_presets.ANSWER_KEYS) == expected
    assert "quality_preset" in expected and "pre_prompt" in expected


def test_index_settings_are_every_saved_index_setting():
    assert list(research_presets.INDEX_KEYS) == [
        setting.key for setting in settings_groups.REINDEX_SETTINGS if setting.write is not None
    ]


def test_a_preset_with_everything_holds_the_answer_and_index_choices_and_the_local_model():
    preset = research_presets.make_preset(SAVED, OLLAMA | {"extra": 1}, include_index=True, include_model=True)

    assert preset == {"answer": ANSWER_PART, "index": INDEX_PART, "model": OLLAMA}


def test_a_preset_records_openai_as_no_local_model():
    preset = research_presets.make_preset({}, None, include_index=False, include_model=True)

    assert preset == {"answer": {}, "model": None}


def test_a_preset_can_leave_out_the_index_and_the_model():
    preset = research_presets.make_preset(SAVED, OLLAMA, include_index=False, include_model=False)

    assert preset == {"answer": ANSWER_PART}


def test_a_preset_skips_saved_values_that_cannot_be_used():
    preset = research_presets.make_preset({"evidence_k": 999, "unknown": 1}, None, True, False)

    assert preset == {"answer": {}, "index": {}}


def test_a_preset_refuses_local_model_details_that_cannot_be_used():
    with pytest.raises(ValueError):
        research_presets.make_preset({}, {"provider": "nope"}, False, True)


# Reading the presets file.


def test_load_returns_nothing_without_a_file(tmp_path):
    assert research_presets.load(tmp_path / "missing.json") == {}


def test_load_returns_nothing_when_the_file_is_not_json(tmp_path):
    path = tmp_path / "presets.json"
    path.write_text("not json", encoding="utf-8")

    assert research_presets.load(path) == {}


def test_load_returns_nothing_when_the_file_is_not_a_set_of_presets(tmp_path):
    assert research_presets.load(_write(tmp_path / "presets.json", [1, 2])) == {}


def test_load_sorts_presets_by_name_ignoring_capitals(tmp_path):
    path = _write(tmp_path / "presets.json", {"beta": {"answer": {}}, "Alpha": {"answer": {}}, "gamma": {"answer": {}}})

    assert list(research_presets.load(path)) == ["Alpha", "beta", "gamma"]


def test_load_skips_entries_that_are_not_presets(tmp_path):
    path = _write(
        tmp_path / "presets.json",
        {"no answer part": {"index": {}}, "not a dict": 3, " ": {"answer": {}}, "kept": {"answer": {}}},
    )

    assert research_presets.load(path) == {"kept": {"answer": {}}}


def test_load_drops_values_that_can_no_longer_be_used(tmp_path):
    path = _write(
        tmp_path / "presets.json",
        {
            "p": {
                "answer": {"evidence_k": 999, "answer_length": "50 to 100 words", "rate_profile": "Low limits"},
                "index": {"chunk_chars": 1, "overlap": 100, "evidence_k": 3},
                "model": OLLAMA,
            }
        },
    )

    assert research_presets.load(path) == {
        "p": {"answer": {"answer_length": "50 to 100 words"}, "index": {"overlap": 100}, "model": OLLAMA}
    }


def test_load_keeps_openai_as_the_model(tmp_path):
    path = _write(tmp_path / "presets.json", {"p": {"answer": {}, "model": None}})

    assert research_presets.load(path) == {"p": {"answer": {}, "model": None}}


def test_load_leaves_out_an_index_part_that_is_not_a_set_of_values(tmp_path):
    path = _write(tmp_path / "presets.json", {"p": {"answer": {}, "index": "bad"}})

    assert research_presets.load(path) == {"p": {"answer": {}}}


@pytest.mark.parametrize("model", [{"provider": "nope", "address": "x", "model": ""}, "llama3"])
def test_load_leaves_out_a_model_that_cannot_be_used(tmp_path, model):
    path = _write(tmp_path / "presets.json", {"p": {"answer": {}, "model": model}})

    assert research_presets.load(path) == {"p": {"answer": {}}}


# Names.


def test_clean_name_puts_the_name_on_one_line_with_single_spaces():
    assert research_presets.clean_name("  Lit\n review   quick ") == "Lit review quick"


def test_clean_name_refuses_an_empty_name():
    with pytest.raises(ValueError, match="Type a name"):
        research_presets.clean_name("   ")


def test_clean_name_refuses_a_long_name():
    with pytest.raises(ValueError, match="at most 60"):
        research_presets.clean_name("x" * 61)


def test_find_name_ignores_capitals():
    presets = {"Lit review": {}}

    assert research_presets.find_name(presets, "lit REVIEW") == "Lit review"
    assert research_presets.find_name(presets, "other") is None


# Saving, renaming and deleting.


def test_save_preset_adds_a_preset(tmp_path):
    path = tmp_path / "presets.json"

    assert research_presets.save_preset(" Quick  look ", {"answer": {"evidence_k": 3}}, path) == ("Quick look", False)
    assert research_presets.load(path) == {"Quick look": {"answer": {"evidence_k": 3}}}


def test_save_preset_replaces_one_with_the_same_name_ignoring_capitals(tmp_path):
    path = _write(tmp_path / "presets.json", {"Quick look": {"answer": {"evidence_k": 3}}, "Other": {"answer": {}}})

    assert research_presets.save_preset("quick LOOK", {"answer": {"evidence_k": 4}}, path) == ("quick LOOK", True)
    assert research_presets.load(path) == {"Other": {"answer": {}}, "quick LOOK": {"answer": {"evidence_k": 4}}}


def test_save_preset_refuses_an_unusable_name_and_saves_nothing(tmp_path):
    path = tmp_path / "presets.json"

    with pytest.raises(ValueError):
        research_presets.save_preset("", {"answer": {}}, path)
    assert not path.exists()


def test_save_preset_keeps_names_as_typed(tmp_path):
    path = tmp_path / "presets.json"

    research_presets.save_preset("Revue “rapide”", {"answer": {}}, path)

    assert "Revue “rapide”" in path.read_text(encoding="utf-8")


def test_rename_preset_gives_it_a_new_name_and_keeps_its_choices(tmp_path):
    path = _write(tmp_path / "presets.json", {"Old": {"answer": {"evidence_k": 3}}})

    assert research_presets.rename_preset("Old", "  New name ", path) == "New name"
    assert research_presets.load(path) == {"New name": {"answer": {"evidence_k": 3}}}


def test_rename_preset_can_change_only_the_capitals(tmp_path):
    path = _write(tmp_path / "presets.json", {"lit review": {"answer": {}}})

    assert research_presets.rename_preset("lit review", "Lit Review", path) == "Lit Review"
    assert list(research_presets.load(path)) == ["Lit Review"]


def test_rename_preset_refuses_a_name_another_preset_has(tmp_path):
    path = _write(tmp_path / "presets.json", {"One": {"answer": {}}, "Two": {"answer": {}}})

    with pytest.raises(ValueError, match="already a preset called “Two”"):
        research_presets.rename_preset("One", "two", path)
    assert list(research_presets.load(path)) == ["One", "Two"]


def test_rename_preset_refuses_a_preset_that_is_gone(tmp_path):
    path = _write(tmp_path / "presets.json", {"One": {"answer": {}}})

    with pytest.raises(ValueError, match="no preset called “Gone”"):
        research_presets.rename_preset("Gone", "New", path)


def test_rename_preset_refuses_an_unusable_name(tmp_path):
    path = _write(tmp_path / "presets.json", {"One": {"answer": {}}})

    with pytest.raises(ValueError, match="Type a name"):
        research_presets.rename_preset("One", " ", path)


def test_delete_preset_removes_it(tmp_path):
    path = _write(tmp_path / "presets.json", {"One": {"answer": {}}, "Two": {"answer": {}}})

    assert research_presets.delete_preset("One", path) is True
    assert list(research_presets.load(path)) == ["Two"]


def test_delete_preset_reports_a_preset_that_is_gone(tmp_path):
    path = _write(tmp_path / "presets.json", {"One": {"answer": {}}})

    assert research_presets.delete_preset("Gone", path) is False
    assert list(research_presets.load(path)) == ["One"]


# Which preset is in use.


def test_a_preset_with_everything_matches_only_when_every_part_is_in_use():
    preset = research_presets.make_preset(SAVED, OLLAMA, True, True)

    assert research_presets.matches(preset, SAVED, OLLAMA)
    assert not research_presets.matches(preset, SAVED | {"evidence_k": 13}, OLLAMA)
    assert not research_presets.matches(preset, SAVED | {"chunk_chars": 4000}, OLLAMA)
    assert not research_presets.matches(preset, SAVED, None)


def test_a_preset_ignores_the_parts_it_leaves_out_and_the_speed_limits():
    preset = research_presets.make_preset(SAVED, OLLAMA, False, False)

    assert research_presets.matches(preset, SAVED | {"chunk_chars": 4000, "rate_profile": "High limits"}, None)


def test_a_preset_with_openai_does_not_match_a_local_model():
    preset = {"answer": {}, "model": None}

    assert research_presets.matches(preset, {}, None)
    assert not research_presets.matches(preset, {}, OLLAMA)


def test_find_active_returns_every_matching_preset_in_order():
    presets = {
        "A": {"answer": {"evidence_k": 12}},
        "B": {"answer": {"evidence_k": 3}},
        "C": {"answer": {"evidence_k": 12}, "index": {}},
    }

    assert research_presets.find_active(presets, {"evidence_k": 12}, None) == ["A", "C"]
    assert research_presets.find_active(presets, {"evidence_k": 5}, None) == []


def test_changes_index_says_whether_the_index_settings_would_change():
    assert research_presets.changes_index({"answer": {}, "index": {"chunk_chars": 3000}}, {})
    assert not research_presets.changes_index({"answer": {}, "index": {"chunk_chars": 3000}}, {"chunk_chars": 3000})
    assert not research_presets.changes_index({"answer": {"evidence_k": 3}}, {})


# Picking a preset.


def test_picking_a_preset_with_everything_restores_every_choice_and_keeps_the_speed_limits(tmp_path):
    settings_path = _write(tmp_path / "saved.json", {"evidence_k": 20, "answer_length": "50 to 100 words",
                                                     "overlap": 500, "rate_profile": "High limits"})
    model_path = tmp_path / "local_llm.json"
    preset = research_presets.make_preset(SAVED, OLLAMA, True, True)

    research_presets.pick_preset(preset, settings_path, model_path)

    assert saved_settings.load(settings_path) == ANSWER_PART | INDEX_PART | {"rate_profile": "High limits"}
    assert local_llm.load_config(model_path) == OLLAMA
    assert research_presets.matches(preset, saved_settings.load(settings_path), local_llm.load_config(model_path))


def test_picking_a_preset_without_the_index_or_model_leaves_them_as_they_are(tmp_path):
    settings_path = _write(tmp_path / "saved.json", {"evidence_k": 20, "overlap": 500})
    model_path = tmp_path / "local_llm.json"
    local_llm.save_config(OLLAMA, model_path)

    research_presets.pick_preset({"answer": {"answer_template": "Study notes"}}, settings_path, model_path)

    assert saved_settings.load(settings_path) == {"answer_template": "Study notes", "overlap": 500}
    assert local_llm.load_config(model_path) == OLLAMA


def test_picking_a_preset_with_openai_removes_the_local_model(tmp_path):
    settings_path = tmp_path / "saved.json"
    model_path = tmp_path / "local_llm.json"
    local_llm.save_config(OLLAMA, model_path)

    research_presets.pick_preset({"answer": {}, "model": None}, settings_path, model_path)

    assert local_llm.load_config(model_path) is None


def test_picking_a_preset_with_default_index_settings_forgets_the_saved_ones(tmp_path):
    settings_path = _write(tmp_path / "saved.json", {"chunk_chars": 3000, "evidence_k": 4})

    research_presets.pick_preset({"answer": {"evidence_k": 4}, "index": {}}, settings_path, tmp_path / "llm.json")

    assert saved_settings.load(settings_path) == {"evidence_k": 4}


def test_a_picked_preset_reaches_the_paperqa_settings():
    """The quality preset goes on before the number saved after it, as when it was saved."""
    values = {}
    preset = research_presets.make_preset(
        {"quality_preset": "Fast", "evidence_k": 12, "chunk_chars": 3000}, None, True, False
    )
    for part in ("answer", "index"):
        values |= preset[part]
    settings = Settings()

    settings_groups.apply_usable(settings, values)

    assert settings.answer.evidence_k == 12
    assert settings.answer.answer_length == "50 to 100 words"
    assert settings.parsing.reader_config["chunk_chars"] == 3000


# Describing a preset.


def test_describe_preset_lists_each_choice_by_the_name_the_sidebar_shows():
    preset = research_presets.make_preset(
        SAVED | {"answer_length": "50 to 100 words", "pre_prompt": ""}, OLLAMA, True, True
    )

    lines = research_presets.describe_preset(preset)

    assert "**Quality preset:** `Fast`" in lines
    assert "**Passages gathered:** `12`" in lines
    assert "**Answer length:** `Short`" in lines
    assert "**Before-answer prompt:** `Off`" in lines
    assert "**Images and tables:** `On with enrichment`" in lines
    assert "**Chunk size (characters):** `3000`" in lines
    assert "**Answer model:** `llama3 (Ollama at http://localhost:11434)`" in lines
    assert not any("Speed limits" in line for line in lines)


def test_describe_preset_says_when_it_uses_the_defaults_and_openai():
    lines = research_presets.describe_preset({"answer": {}, "index": {}, "model": None})

    assert lines == [
        "Answer settings: PaperQA's defaults.",
        "Library and index settings: the defaults.",
        "**Answer model:** `OpenAI`",
    ]


def test_describe_preset_says_what_it_leaves_as_it_is():
    lines = research_presets.describe_preset({"answer": {"evidence_k": 3}})

    assert lines == [
        "**Passages gathered:** `3`",
        "Library and index settings: not included, so they stay as they are.",
        "Answer model: not included, so it stays as it is.",
    ]
