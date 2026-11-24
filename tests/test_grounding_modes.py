from paperqa import Settings
from paperqa.prompts import CANNOT_ANSWER_PHRASE, default_system_prompt
import pytest

from pdf_research_assistant import answer_templates, grounding_modes


def test_modes_have_unique_names_and_the_default_is_one_of_them():
    names = [mode.name for mode in grounding_modes.MODES]

    assert names == ["Strictly grounded", "Balanced", "Exploratory"]
    assert grounding_modes.DEFAULT_MODE in names


def test_every_mode_says_what_it_is_for():
    assert all(mode.description.strip() for mode in grounding_modes.MODES)


def test_the_default_mode_is_paperqa_s_own_system_prompt_and_cutoff():
    settings = Settings()
    default = grounding_modes.get_mode(grounding_modes.DEFAULT_MODE)

    assert grounding_modes.mode_system_prompt(default) == default_system_prompt
    assert default.relevance_cutoff == settings.answer.evidence_relevance_score_cutoff


def test_every_other_mode_adds_its_instruction_after_paperqa_s_system_prompt():
    for mode in grounding_modes.MODES:
        if mode.instruction is None:
            continue
        assert grounding_modes.mode_system_prompt(mode) == f"{default_system_prompt} {mode.instruction}"


def test_the_cannot_answer_phrase_is_paperqa_s_own():
    assert grounding_modes.CANNOT_ANSWER_PHRASE == CANNOT_ANSWER_PHRASE


def test_the_balanced_cutoff_is_paperqa_s_default():
    assert grounding_modes.PAPERQA_CUTOFF == Settings().answer.evidence_relevance_score_cutoff


def test_strictly_grounded_refuses_without_direct_support_and_drops_weak_passages():
    mode = grounding_modes.get_mode("Strictly grounded")

    assert CANNOT_ANSWER_PHRASE in mode.instruction
    assert "own knowledge" in mode.instruction
    assert mode.relevance_cutoff > grounding_modes.get_mode("Balanced").relevance_cutoff


def test_exploratory_marks_background_knowledge_and_keeps_paperqa_s_cutoff():
    mode = grounding_modes.get_mode("Exploratory")

    assert grounding_modes.BACKGROUND_MARK in mode.instruction
    assert mode.relevance_cutoff == grounding_modes.get_mode("Balanced").relevance_cutoff


def test_the_background_mark_is_the_one_paperqa_can_filter_out():
    # PaperQA's answer.answer_filter_extra_background removes exactly this mark, whatever its capitals.
    assert grounding_modes.BACKGROUND_MARK.lower() == "(extra background information)"


def test_get_mode_returns_the_named_mode():
    assert grounding_modes.get_mode("Exploratory").name == "Exploratory"


def test_get_mode_raises_for_an_unknown_name():
    with pytest.raises(KeyError):
        grounding_modes.get_mode("Loose")


def test_apply_mode_sets_paperqa_s_system_prompt_and_relevance_cutoff():
    settings = Settings()

    grounding_modes.apply_mode(settings, "Strictly grounded")

    mode = grounding_modes.get_mode("Strictly grounded")
    assert settings.prompts.system == grounding_modes.mode_system_prompt(mode)
    assert settings.answer.evidence_relevance_score_cutoff == mode.relevance_cutoff


def test_apply_mode_leaves_the_answer_template_and_other_prompts_alone():
    settings = Settings()
    answer_templates.apply_template(settings, "Compare sources")
    before = settings.prompts.model_dump()

    grounding_modes.apply_mode(settings, "Exploratory")

    after = settings.prompts.model_dump()
    assert {key for key in before if before[key] != after[key]} == {"system"}
    assert answer_templates.read_template_name(settings) == "Compare sources"


def test_read_mode_name_finds_each_mode_from_the_settings_in_use():
    settings = Settings()
    assert grounding_modes.read_mode_name(settings) == "Balanced"

    for mode in grounding_modes.MODES:
        grounding_modes.apply_mode(settings, mode.name)
        assert grounding_modes.read_mode_name(settings) == mode.name


def test_read_mode_name_says_when_the_system_prompt_is_no_mode():
    settings = Settings()
    settings.prompts.system = "Answer like a pirate."

    assert grounding_modes.read_mode_name(settings) == grounding_modes.NO_MODE


def test_read_mode_name_says_when_the_cutoff_is_no_mode():
    settings = Settings()
    grounding_modes.apply_mode(settings, "Strictly grounded")
    settings.answer.evidence_relevance_score_cutoff = 3

    assert grounding_modes.read_mode_name(settings) == grounding_modes.NO_MODE


def test_reading_the_mode_does_not_change_the_answer_template():
    settings = Settings()
    for template in answer_templates.TEMPLATES:
        answer_templates.apply_template(settings, template.name)
        for mode in grounding_modes.MODES:
            grounding_modes.apply_mode(settings, mode.name)
            assert grounding_modes.read_mode_name(settings) == mode.name
            assert answer_templates.read_template_name(settings) == template.name
