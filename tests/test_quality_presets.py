from paperqa import Settings
import pytest

from pdf_research_assistant import answer_templates, grounding_modes, quality_presets, settings_groups


def test_there_are_four_presets_with_unique_names_and_the_default_is_one_of_them():
    names = [preset.name for preset in quality_presets.PRESETS]

    assert names == ["Fast", "Balanced", "High quality", "Debug"]
    assert quality_presets.DEFAULT_PRESET in names
    assert quality_presets.NO_PRESET not in names


def test_every_preset_says_what_it_is_for_and_which_paperqa_preset_it_follows():
    for preset in quality_presets.PRESETS:
        assert preset.description.strip().endswith(".")
        assert preset.based_on


def _paperqa_values(settings):
    return (
        settings.answer.evidence_k,
        settings.answer.answer_max_sources,
        settings.answer.answer_length,
        settings.answer.evidence_summary_length,
        settings.agent.agent_type,
    )


def _preset_values(preset):
    return (
        preset.evidence_k,
        preset.answer_max_sources,
        preset.answer_length,
        preset.summary_length,
        preset.agent_type,
    )


@pytest.mark.parametrize("preset", quality_presets.PRESETS, ids=lambda preset: preset.name)
def test_every_preset_matches_the_paperqa_preset_it_follows(preset):
    if preset.based_on.endswith(".json"):
        paperqa_settings = Settings.from_name(preset.based_on.removesuffix(".json"))
    else:
        paperqa_settings = Settings()

    assert _preset_values(preset) == _paperqa_values(paperqa_settings)


def test_the_default_preset_is_what_paperqa_uses_with_no_changes():
    assert quality_presets.read_preset_name(Settings()) == quality_presets.DEFAULT_PRESET


def test_answer_lengths_are_ones_the_answer_length_control_offers():
    offered = settings_groups.get_setting("answer_length").check.values

    assert all(preset.answer_length in offered for preset in quality_presets.PRESETS)


def test_numbers_are_inside_the_limits_of_their_own_controls():
    gathered = settings_groups.get_setting("evidence_k").check
    sources = settings_groups.get_setting("answer_max_sources").check

    for preset in quality_presets.PRESETS:
        assert gathered(preset.evidence_k) == preset.evidence_k
        assert sources(preset.answer_max_sources) == preset.answer_max_sources
        assert preset.answer_max_sources <= preset.evidence_k


@pytest.mark.parametrize("preset", quality_presets.PRESETS, ids=lambda preset: preset.name)
def test_applying_a_preset_puts_its_values_on_the_settings_and_reads_back_by_name(preset):
    settings = Settings()

    quality_presets.apply_preset(settings, preset.name)

    assert _paperqa_values(settings) == _preset_values(preset)
    assert quality_presets.read_preset_name(settings) == preset.name


def test_changing_any_one_preset_value_makes_the_settings_read_as_custom():
    changes = (
        lambda s: setattr(s.answer, "evidence_k", 7),
        lambda s: setattr(s.answer, "answer_max_sources", 4),
        lambda s: setattr(s.answer, "answer_length", "about 400 words, but can be longer"),
        lambda s: setattr(s.answer, "evidence_summary_length", "about 300 words"),
        lambda s: setattr(s.agent, "agent_type", quality_presets.FIXED_STEPS),
    )
    for change in changes:
        settings = Settings()
        change(settings)

        assert quality_presets.read_preset_name(settings) == quality_presets.NO_PRESET


def test_a_preset_leaves_the_template_grounding_before_answer_prompt_and_speed_limits_alone():
    settings = Settings()
    answer_templates.apply_template(settings, "Study notes")
    grounding_modes.apply_mode(settings, "Strictly grounded")
    settings.prompts.pre = "Explain: {question}"
    concurrency = settings.answer.max_concurrent_requests
    llm = settings.llm

    quality_presets.apply_preset(settings, "Fast")

    assert answer_templates.read_template_name(settings) == "Study notes"
    assert grounding_modes.read_mode_name(settings) == "Strictly grounded"
    assert settings.prompts.pre == "Explain: {question}"
    assert settings.answer.max_concurrent_requests == concurrency
    assert settings.llm == llm


def test_a_preset_keeps_the_index_name(tmp_path):
    settings = Settings()
    name = settings.get_index_name()

    for preset in quality_presets.PRESETS:
        quality_presets.apply_preset(settings, preset.name)
        assert settings.get_index_name() == name


def test_get_preset_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        quality_presets.get_preset("Thorough")


def test_the_keys_with_their_own_controls_are_saved_settings():
    for key in quality_presets.KEYS_WITH_CONTROLS:
        assert settings_groups.get_setting(key).write is not None


def test_describe_agent_type_names_paperqa_s_two_types_and_shows_any_other_as_it_is():
    assert quality_presets.describe_agent_type("ToolSelector") == "Planned by the model"
    assert quality_presets.describe_agent_type("fake") == "Fixed steps"
    assert quality_presets.describe_agent_type("SomeOtherAgent") == "SomeOtherAgent"
