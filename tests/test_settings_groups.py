import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from pdf_research_assistant import rate_profiles, settings_groups


def _other_pdf_reader(*_args, **_kwargs):
    return None


def _real_settings(paper_dir: Path):
    from paperqa import Settings
    from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

    return Settings(
        parsing=ParsingSettings(multimodal=False, use_doc_details=False),
        agent=AgentSettings(index=IndexSettings(paper_directory=paper_dir, index_directory=paper_dir / "index")),
    )


def _set_multimodal(settings):
    from paperqa.settings import MultimodalOptions

    settings.parsing.multimodal = MultimodalOptions.ON_WITHOUT_ENRICHMENT


# One change per setting, made the way PaperQA stores it.
CHANGES = {
    "answer_model": lambda s: setattr(s, "llm", "some-other-model"),
    "quality_preset": lambda s: settings_groups.get_setting("quality_preset").write(s, "Fast"),
    "evidence_summary_length": lambda s: setattr(s.answer, "evidence_summary_length", "25 to 50 words"),
    "agent_type": lambda s: setattr(s.agent, "agent_type", "fake"),
    "evidence_k": lambda s: setattr(s.answer, "evidence_k", 3),
    "answer_max_sources": lambda s: setattr(s.answer, "answer_max_sources", 2),
    "answer_length": lambda s: setattr(s.answer, "answer_length", "about 50 words"),
    "answer_template": lambda s: settings_groups.get_setting("answer_template").write(s, "Study notes"),
    "grounding": lambda s: settings_groups.get_setting("grounding").write(s, "Strictly grounded"),
    "pre_prompt": lambda s: settings_groups.get_setting("pre_prompt").write(s, "List the terms in: {question}"),
    "rate_profile": lambda s: settings_groups.get_setting("rate_profile").write(s, "Low limits"),
    "paper_directory": lambda s: setattr(s.agent.index, "paper_directory", Path(s.agent.index.paper_directory) / "other"),
    "use_absolute_paper_directory": lambda s: setattr(s.agent.index, "use_absolute_paper_directory", True),
    "embedding": lambda s: setattr(s, "embedding", "text-embedding-3-large"),
    "parse_pdf": lambda s: setattr(s.parsing, "parse_pdf", _other_pdf_reader),
    "chunk_chars": lambda s: s.parsing.reader_config.update(chunk_chars=1234),
    "overlap": lambda s: s.parsing.reader_config.update(overlap=12),
    "full_page": lambda s: s.parsing.reader_config.update(full_page=True),
    "multimodal": _set_multimodal,
    "manifest_scope": lambda s: setattr(s.agent.index, "manifest_file", None),
    "recurse_subdirectories": lambda s: setattr(s.agent.index, "recurse_subdirectories", False),
}


def test_every_setting_is_in_exactly_one_group():
    instant = [setting.key for setting in settings_groups.INSTANT_SETTINGS]
    reindex = [setting.key for setting in settings_groups.REINDEX_SETTINGS]

    assert instant == [
        "answer_model",
        "quality_preset",
        "evidence_k",
        "answer_max_sources",
        "answer_length",
        "evidence_summary_length",
        "agent_type",
        "answer_template",
        "grounding",
        "pre_prompt",
        "rate_profile",
    ]
    assert reindex == [
        "paper_directory",
        "use_absolute_paper_directory",
        "embedding",
        "parse_pdf",
        "chunk_chars",
        "overlap",
        "full_page",
        "multimodal",
    ]
    scope = [setting.key for setting in settings_groups.SCOPE_SETTINGS]
    assert scope == ["manifest_scope", "recurse_subdirectories"]
    assert [setting.key for setting in settings_groups.SETTINGS] == instant + reindex + scope
    assert all(not setting.needs_reindex for setting in settings_groups.SCOPE_SETTINGS)
    assert all(not setting.needs_reindex for setting in settings_groups.INSTANT_SETTINGS)
    assert all(setting.needs_reindex for setting in settings_groups.REINDEX_SETTINGS)


def test_every_setting_has_a_label_and_an_explanation():
    for setting in settings_groups.SETTINGS:
        assert setting.label
        assert setting.help.endswith(".")


@pytest.mark.parametrize("setting", settings_groups.REINDEX_SETTINGS, ids=lambda setting: setting.key)
def test_changing_a_reindex_setting_changes_the_paperqa_index_name(setting, tmp_path):
    settings = _real_settings(tmp_path)
    name_before = settings.get_index_name()
    value_before = setting.read(settings)

    CHANGES[setting.key](settings)

    assert setting.read(settings) != value_before
    assert settings.get_index_name() != name_before


@pytest.mark.parametrize("setting", settings_groups.INSTANT_SETTINGS, ids=lambda setting: setting.key)
def test_changing_an_instant_setting_keeps_the_paperqa_index_name(setting, tmp_path):
    settings = _real_settings(tmp_path)
    name_before = settings.get_index_name()
    value_before = setting.read(settings)

    CHANGES[setting.key](settings)

    assert setting.read(settings) != value_before
    assert settings.get_index_name() == name_before


def test_reindex_group_lists_everything_paperqa_puts_in_the_index_name(tmp_path):
    """Fails when a PaperQA upgrade adds or removes a part of the index name, so the group gets updated."""
    with mock.patch("paperqa.settings.hexdigest", return_value="x") as hexdigest:
        _real_settings(tmp_path).get_index_name()

    parts = hexdigest.call_args.args[0].split("|")
    assert len(parts) == len(settings_groups.REINDEX_SETTINGS)


def test_full_page_reads_as_off_when_paperqa_has_no_value_for_it():
    settings = SimpleNamespace(parsing=SimpleNamespace(reader_config={"chunk_chars": 5000, "overlap": 250}))

    assert settings_groups.get_setting("full_page").read(settings) is False


def test_get_setting_finds_a_setting_by_key():
    assert settings_groups.get_setting("embedding").label == "Embedding model"


def test_get_setting_rejects_an_unknown_key():
    with pytest.raises(KeyError, match="no_such_setting"):
        settings_groups.get_setting("no_such_setting")


def test_needs_reindex_tells_the_two_groups_apart():
    assert settings_groups.needs_reindex("chunk_chars") is True
    assert settings_groups.needs_reindex("answer_model") is False


def test_read_values_returns_the_current_value_of_each_setting_in_a_group(tmp_path):
    settings = _real_settings(tmp_path)

    values = settings_groups.read_values(settings, settings_groups.REINDEX_SETTINGS)

    assert list(values) == [setting.key for setting in settings_groups.REINDEX_SETTINGS]
    assert values["paper_directory"] == tmp_path
    assert values["chunk_chars"] == 5000
    assert values["overlap"] == 250
    assert values["full_page"] is False
    assert settings_groups.read_values(settings, settings_groups.INSTANT_SETTINGS) == {
        "answer_model": settings.llm,
        "quality_preset": "Balanced",
        "evidence_k": 10,
        "answer_max_sources": 5,
        "answer_length": "about 200 words, but can be longer",
        "evidence_summary_length": "about 100 words",
        "agent_type": "Planned by the model",
        "answer_template": "Direct answer",
        "grounding": "Balanced",
        # PaperQA's own value: None means no before-answer prompt.
        "pre_prompt": None,
        # A bare PaperQA Settings object has none of the profiles' limits.
        "rate_profile": rate_profiles.NO_PROFILE,
    }


class _Colour(Enum):
    DARK_RED = 1


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (True, "On"),
        (False, "Off"),
        (_Colour.DARK_RED, "Dark red"),
        (_other_pdf_reader, "_other_pdf_reader"),
        (Path("papers") / "mine", str(Path("papers") / "mine")),
        (5000, "5000"),
        (None, "Off"),
        ("text-embedding-3-small", "text-embedding-3-small"),
    ],
)
def test_describe_value_shows_each_kind_of_value_in_plain_words(value, shown):
    assert settings_groups.describe_value(value) == shown


def test_describe_value_falls_back_to_text_for_a_callable_without_a_name():
    class Reader:
        def __call__(self):
            return None

        def __str__(self):
            return "custom reader"

    assert settings_groups.describe_value(Reader()) == "custom reader"


def test_describe_setting_shows_the_label_and_current_value():
    settings = SimpleNamespace(embedding="text-embedding-3-small")

    line = settings_groups.describe_setting(settings_groups.get_setting("embedding"), settings)

    assert line == "**Embedding model:** `text-embedding-3-small`"


def test_describe_line_shows_a_label_and_value_as_markdown():
    assert settings_groups.describe_line("Answer length", "about 50 words") == "**Answer length:** `about 50 words`"


def test_record_answer_settings_keeps_the_index_and_every_instant_setting_as_text(tmp_path):
    settings = _real_settings(tmp_path)

    record = settings_groups.record_answer_settings(settings)

    assert record["index"] == settings.get_index_name()
    assert record["values"] == [
        {"key": setting.key, "label": setting.label, "value": settings_groups.describe_value(setting.read(settings)), "source": None}
        for setting in settings_groups.INSTANT_SETTINGS
    ]
    assert {item["key"]: item["value"] for item in record["values"]}["evidence_k"] == "10"
    assert json.loads(json.dumps(record)) == record


def test_describe_recorded_setting_shows_the_label_and_value():
    item = {"key": "evidence_k", "label": "Passages gathered", "value": "10", "source": None}

    assert settings_groups.describe_recorded_setting(item) == "**Passages gathered:** `10`"


def test_describe_recorded_setting_says_where_a_value_came_from_when_known():
    item = {"key": "evidence_k", "label": "Passages gathered", "value": "3", "source": "This question only"}

    assert settings_groups.describe_recorded_setting(item) == "**Passages gathered:** `3` :gray-badge[This question only]"


# Settings the user can change from the sidebar.


def test_the_settings_that_can_be_saved_from_the_sidebar():
    saved = [setting.key for setting in settings_groups.SETTINGS if setting.write is not None]

    assert saved == [
        "quality_preset",
        "evidence_k",
        "answer_max_sources",
        "answer_length",
        "answer_template",
        "grounding",
        "pre_prompt",
        "rate_profile",
        "embedding",
        "chunk_chars",
        "overlap",
        "full_page",
        "multimodal",
        "manifest_scope",
        "recurse_subdirectories",
    ]
    assert all(setting.check is not None for setting in settings_groups.SETTINGS if setting.write is not None)


@pytest.mark.parametrize(
    ("key", "value"), [("evidence_k", 3), ("answer_max_sources", 2), ("answer_length", "50 to 100 words")]
)
def test_writing_a_source_setting_changes_the_paperqa_value_it_reads(key, value, tmp_path):
    settings = _real_settings(tmp_path)
    setting = settings_groups.get_setting(key)

    setting.write(settings, value)

    assert setting.read(settings) == value


def test_source_settings_allow_what_paperqa_s_own_presets_use():
    gathered = settings_groups.get_setting("evidence_k").check
    sources = settings_groups.get_setting("answer_max_sources").check

    # PaperQA's bundled presets go as high as 30 passages and 15 sources.
    assert (gathered.low, gathered.high) == (1, 30)
    assert (sources.low, sources.high) == (1, 15)


def test_whole_number_returns_a_value_inside_its_limits():
    check = settings_groups.WholeNumber(1, 30)

    assert check(1) == 1
    assert check(30) == 30


@pytest.mark.parametrize("value", [0, 31, -1])
def test_whole_number_rejects_a_value_outside_its_limits(value):
    with pytest.raises(ValueError, match="between 1 and 30"):
        settings_groups.WholeNumber(1, 30)(value)


@pytest.mark.parametrize("value", [True, 3.0, "3", None])
def test_whole_number_rejects_anything_that_is_not_a_whole_number(value):
    with pytest.raises(ValueError, match="whole number"):
        settings_groups.WholeNumber(1, 30)(value)


def test_source_setting_help_mentions_cost_and_speed_in_plain_words():
    gathered = settings_groups.get_setting("evidence_k").help
    sources = settings_groups.get_setting("answer_max_sources").help

    assert "cost" in gathered and "longer" in gathered
    assert "cost" in sources


# Answer length choices.


def test_answer_length_offers_named_choices_including_paperqa_s_default(tmp_path):
    check = settings_groups.get_setting("answer_length").check
    default = settings_groups.get_setting("answer_length").read(_real_settings(tmp_path))

    assert [name for name, _phrase in check.choices] == ["Short", "Standard", "Long"]
    assert check.name_for(default) == "Standard"
    # PaperQA's own fast preset uses the short phrase.
    assert check.name_for("50 to 100 words") == "Short"


def test_one_of_returns_an_offered_value():
    check = settings_groups.OneOf((("Short", "a few words"), ("Long", "many words")))

    assert check("a few words") == "a few words"
    assert check.values == ("a few words", "many words")


@pytest.mark.parametrize("value", ["some words", "Short", None, 3])
def test_one_of_rejects_anything_that_was_not_offered(value):
    with pytest.raises(ValueError, match="Expected one of"):
        settings_groups.OneOf((("Short", "a few words"),))(value)


def test_one_of_names_an_offered_value_and_returns_none_for_any_other():
    check = settings_groups.OneOf((("Short", "a few words"),))

    assert check.name_for("a few words") == "Short"
    assert check.name_for("some words") is None


def test_answer_length_help_mentions_cost_speed_and_exhaustive_search():
    help_text = settings_groups.get_setting("answer_length").help

    assert "cost" in help_text and "longer" in help_text
    assert "Exhaustive search ignores" in help_text


# Answer templates.


def test_answer_template_offers_each_template_by_name_starting_at_paperqa_s_own_prompt(tmp_path):
    setting = settings_groups.get_setting("answer_template")

    assert setting.check.values == ("Direct answer", "Study notes", "Compare sources", "Contradictions only")
    assert setting.read(_real_settings(tmp_path)) == "Direct answer"


def test_writing_an_answer_template_changes_paperqa_s_answer_prompt(tmp_path):
    from pdf_research_assistant import answer_templates

    settings = _real_settings(tmp_path)

    settings_groups.get_setting("answer_template").write(settings, "Contradictions only")

    assert settings.prompts.qa == answer_templates.template_prompt(answer_templates.get_template("Contradictions only"))
    assert settings_groups.get_setting("answer_template").read(settings) == "Contradictions only"


def test_record_answer_settings_shows_the_template_by_name(tmp_path):
    settings = _real_settings(tmp_path)
    settings_groups.get_setting("answer_template").write(settings, "Study notes")

    record = settings_groups.record_answer_settings(settings, one_off=["answer_template"])

    item = next(item for item in record["values"] if item["key"] == "answer_template")
    assert item == {
        "key": "answer_template",
        "label": "Answer template",
        "value": "Study notes",
        "source": settings_groups.ONE_OFF_SOURCE,
    }


def test_answer_template_help_says_it_shapes_the_answer_and_exhaustive_search_ignores_it():
    help_text = settings_groups.get_setting("answer_template").help

    assert "same passages" in help_text
    assert "Exhaustive search ignores" in help_text


# Grounding modes.


def test_grounding_offers_each_mode_by_name_starting_at_paperqa_s_own_settings(tmp_path):
    setting = settings_groups.get_setting("grounding")

    assert setting.check.values == ("Strictly grounded", "Balanced", "Exploratory")
    assert setting.read(_real_settings(tmp_path)) == "Balanced"


def test_writing_a_grounding_mode_changes_paperqa_s_system_prompt_and_cutoff_but_not_the_template(tmp_path):
    from pdf_research_assistant import grounding_modes

    settings = _real_settings(tmp_path)
    settings_groups.get_setting("answer_template").write(settings, "Study notes")

    settings_groups.get_setting("grounding").write(settings, "Strictly grounded")

    mode = grounding_modes.get_mode("Strictly grounded")
    assert settings.prompts.system == grounding_modes.mode_system_prompt(mode)
    assert settings.answer.evidence_relevance_score_cutoff == mode.relevance_cutoff
    assert settings_groups.get_setting("grounding").read(settings) == "Strictly grounded"
    assert settings_groups.get_setting("answer_template").read(settings) == "Study notes"


def test_record_answer_settings_shows_the_grounding_mode_by_name(tmp_path):
    settings = _real_settings(tmp_path)
    settings_groups.get_setting("grounding").write(settings, "Exploratory")

    record = settings_groups.record_answer_settings(settings, one_off=["grounding"])

    item = next(item for item in record["values"] if item["key"] == "grounding")
    assert item == {
        "key": "grounding",
        "label": "Grounding",
        "value": "Exploratory",
        "source": settings_groups.ONE_OFF_SOURCE,
    }


def test_grounding_help_says_what_it_changes_and_exhaustive_search_ignores_it():
    help_text = settings_groups.get_setting("grounding").help

    assert "own knowledge" in help_text
    assert "relevance score" in help_text
    assert "Exhaustive search ignores" in help_text


# Before-answer prompt.


def test_pre_prompt_starts_off_and_writes_paperqa_s_pre_prompt(tmp_path):
    settings = _real_settings(tmp_path)
    setting = settings_groups.get_setting("pre_prompt")

    assert setting.read(settings) is None
    assert settings_groups.current_choice(setting, settings) == ""
    setting.write(settings, "List the terms in: {question}")

    assert settings.prompts.pre == "List the terms in: {question}"
    assert setting.read(settings) == "List the terms in: {question}"


def test_record_answer_settings_shows_the_pre_prompt_or_off(tmp_path):
    settings = _real_settings(tmp_path)
    setting = settings_groups.get_setting("pre_prompt")

    def recorded():
        record = settings_groups.record_answer_settings(settings)
        return next(item["value"] for item in record["values"] if item["key"] == "pre_prompt")

    assert recorded() == "Off"
    setting.write(settings, "List the terms in: {question}")
    assert recorded() == "List the terms in: {question}"


def test_pre_prompt_help_says_it_costs_a_call_differs_from_the_template_and_follows_the_grounding():
    help_text = settings_groups.get_setting("pre_prompt").help

    assert "{question}" in help_text
    assert "extra call" in help_text
    assert "answer template" in help_text
    assert "grounding" in help_text
    assert "Exhaustive search ignores" in help_text


def test_describe_line_keeps_a_long_prompt_on_one_line_and_its_backticks_readable():
    line = settings_groups.describe_line("Before-answer prompt", "First line\n\nUse `code` here")

    assert line == "**Before-answer prompt:** `` First line Use `code` here ``"


# Checking and applying values chosen in the app, saved or for one question only.


def test_usable_accepts_an_allowed_value_for_a_setting_the_sidebar_can_change():
    assert settings_groups.usable("evidence_k", 3) is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("no_such_setting", 3),
        ("answer_model", "some-other-model"),
        ("paper_directory", "papers"),
        ("evidence_k", 0),
        ("answer_length", "one paragraph"),
        ("answer_template", "Poem"),
        ("grounding", "Loose"),
        ("pre_prompt", "Summarise {answer}"),
        ("quality_preset", "Custom"),
        ("agent_type", "fake"),
    ],
)
def test_usable_rejects_unknown_settings_settings_the_sidebar_cannot_change_and_values_not_allowed(key, value):
    assert settings_groups.usable(key, value) is False


def test_apply_usable_puts_each_allowed_value_on_the_settings_and_says_which_it_applied(tmp_path):
    settings = _real_settings(tmp_path)

    applied = settings_groups.apply_usable(
        settings, {"evidence_k": 3, "answer_length": "50 to 100 words", "answer_max_sources": 99, "answer_model": "x"}
    )

    assert applied == ["evidence_k", "answer_length"]
    assert settings.answer.evidence_k == 3
    assert settings.answer.answer_length == "50 to 100 words"
    assert settings.answer.answer_max_sources == 5
    assert settings.llm != "x"


def test_apply_usable_changes_nothing_when_there_are_no_values(tmp_path):
    settings = _real_settings(tmp_path)

    assert settings_groups.apply_usable(settings, {}) == []
    assert settings.answer.evidence_k == 10


def test_record_answer_settings_marks_values_changed_for_this_question_only(tmp_path):
    settings = _real_settings(tmp_path)
    settings.answer.evidence_k = 3

    record = settings_groups.record_answer_settings(settings, one_off=["evidence_k"])

    sources = {item["key"]: item["source"] for item in record["values"]}
    assert sources == {
        "answer_model": None,
        "quality_preset": None,
        "evidence_k": settings_groups.ONE_OFF_SOURCE,
        "answer_max_sources": None,
        "answer_length": None,
        "evidence_summary_length": None,
        "agent_type": None,
        "answer_template": None,
        "grounding": None,
        "pre_prompt": None,
        "rate_profile": None,
    }
    assert {item["key"]: item["value"] for item in record["values"]}["evidence_k"] == "3"
    assert settings_groups.ONE_OFF_SOURCE == "This question only"


# Library and index settings the user can change from the sidebar.

INDEX_CHOICES = [
    ("embedding", "text-embedding-3-large"),
    ("chunk_chars", 3000),
    ("overlap", 100),
    ("full_page", True),
    ("multimodal", "on_without_enrichment"),
    ("multimodal", "on_with_enrichment"),
]


@pytest.mark.parametrize(("key", "value"), INDEX_CHOICES)
def test_writing_an_index_setting_changes_the_paperqa_value_and_the_index_name(key, value, tmp_path):
    settings = _real_settings(tmp_path)
    setting = settings_groups.get_setting(key)
    name_before = settings.get_index_name()

    setting.write(settings, setting.check(value))

    assert settings_groups.current_choice(setting, settings) == value
    assert settings.get_index_name() != name_before


def test_every_index_setting_starts_at_a_value_its_control_offers(tmp_path):
    settings = _real_settings(tmp_path)

    for setting in settings_groups.REINDEX_SETTINGS:
        if setting.write is not None:
            assert settings_groups.usable(setting.key, settings_groups.current_choice(setting, settings))


def test_saving_an_index_setting_at_the_value_in_use_keeps_the_index_name(tmp_path):
    """Otherwise saving a form the user left alone would send PaperQA looking for a new index."""
    settings = _real_settings(tmp_path)
    name_before = settings.get_index_name()

    for setting in settings_groups.REINDEX_SETTINGS:
        if setting.write is not None:
            setting.write(settings, settings_groups.current_choice(setting, settings))

    assert settings.get_index_name() == name_before
    assert settings.parsing.multimodal is False


def test_current_choice_is_the_value_read_unless_the_setting_stores_it_another_way():
    settings = SimpleNamespace(embedding="text-embedding-3-small", parsing=SimpleNamespace(multimodal=True))

    assert settings_groups.current_choice(settings_groups.get_setting("embedding"), settings) == "text-embedding-3-small"
    assert settings_groups.current_choice(settings_groups.get_setting("multimodal"), settings) == "on_with_enrichment"


def test_images_and_tables_choices_are_named_as_the_value_is_shown_everywhere_else(tmp_path):
    """The reindex warning and last_index.json show describe_value's text, so the choices use the same words."""
    setting = settings_groups.get_setting("multimodal")
    settings = _real_settings(tmp_path)

    for name, value in setting.check.choices:
        setting.write(settings, value)
        assert settings_groups.describe_value(setting.read(settings)) == name
    assert [name for name, _value in setting.check.choices] == ["Off", "On without enrichment", "On with enrichment"]


def test_chunk_limits_cover_paperqa_s_presets_and_keep_the_overlap_shorter_than_a_chunk():
    chunk = settings_groups.get_setting("chunk_chars").check
    overlap = settings_groups.get_setting("overlap").check

    # PaperQA's bundled presets use chunks of 5000 to 9000 characters and overlaps of 250 to 1750.
    assert chunk.low <= 5000 and chunk.high >= 9000
    assert overlap.low == 0 and overlap.high >= 1750
    # PaperQA's chunking never finishes when the overlap is as long as the chunk.
    assert overlap.high < chunk.low
    assert (chunk.step, overlap.step) == (500, 50)


def test_whole_number_steps_by_one_unless_told_otherwise():
    assert settings_groups.WholeNumber(1, 30).step == 1


def test_embedding_offers_openai_s_two_current_models():
    assert settings_groups.get_setting("embedding").check.values == ("text-embedding-3-small", "text-embedding-3-large")


@pytest.mark.parametrize("value", [True, False])
def test_on_or_off_returns_either_value(value):
    assert settings_groups.OnOrOff()(value) is value


@pytest.mark.parametrize("value", [1, 0, "on", None])
def test_on_or_off_rejects_anything_else(value):
    with pytest.raises(ValueError, match="on or off"):
        settings_groups.OnOrOff()(value)


@pytest.mark.parametrize("setting", settings_groups.REINDEX_SETTINGS, ids=lambda setting: setting.key)
def test_every_index_setting_says_whether_it_affects_indexing_only(setting):
    assert ("Affects indexing only" in setting.help) != ("Also affects questions" in setting.help)


def test_index_settings_that_cost_money_say_so():
    assert "cost" in settings_groups.get_setting("embedding").help
    assert "cost" in settings_groups.get_setting("multimodal").help
    assert "cost" in settings_groups.get_setting("chunk_chars").help


# Quality presets.


def test_quality_preset_offers_each_preset_by_name_starting_at_paperqa_s_own_settings(tmp_path):
    from pdf_research_assistant import quality_presets

    setting = settings_groups.get_setting("quality_preset")

    assert setting.check.values == tuple(preset.name for preset in quality_presets.PRESETS)
    assert setting.read(_real_settings(tmp_path)) == quality_presets.DEFAULT_PRESET


def test_writing_a_quality_preset_changes_the_paperqa_values_it_covers(tmp_path):
    settings = _real_settings(tmp_path)

    settings_groups.get_setting("quality_preset").write(settings, "High quality")

    assert settings.answer.evidence_k == 20
    assert settings_groups.get_setting("quality_preset").read(settings) == "High quality"


def test_apply_usable_puts_a_preset_on_first_so_a_number_changed_afterwards_wins(tmp_path):
    settings = _real_settings(tmp_path)

    applied = settings_groups.apply_usable(settings, {"evidence_k": 7, "quality_preset": "Fast"})

    assert applied == ["quality_preset", "evidence_k"]
    assert settings.answer.evidence_k == 7
    assert settings.answer.answer_max_sources == 3
    assert settings.agent.agent_type == "fake"
    assert settings_groups.get_setting("quality_preset").read(settings) == "Custom"


def test_record_answer_settings_shows_the_preset_summary_length_and_search_steps(tmp_path):
    settings = _real_settings(tmp_path)
    settings_groups.get_setting("quality_preset").write(settings, "Fast")

    values = {item["key"]: item["value"] for item in settings_groups.record_answer_settings(settings)["values"]}

    assert values["quality_preset"] == "Fast"
    assert values["evidence_summary_length"] == "25 to 50 words"
    assert values["agent_type"] == "Fixed steps"


def test_quality_preset_help_says_what_it_sets_what_it_leaves_alone_and_exhaustive_search_ignores_it():
    help_text = settings_groups.get_setting("quality_preset").help

    for words in ("passages", "sources", "answer length", "Custom", "template", "grounding", "speed limits"):
        assert words in help_text
    assert "Exhaustive search ignores it" in help_text


def test_search_steps_help_explains_both_ways_of_searching():
    help_text = settings_groups.get_setting("agent_type").help

    assert "Planned by the model" in help_text
    assert "Fixed steps" in help_text


def _settings_with_paper_list(tmp_path):
    settings = _real_settings(tmp_path)
    settings.agent.index.manifest_file = tmp_path / "manifest.csv"
    return settings


@pytest.mark.parametrize("setting", settings_groups.SCOPE_SETTINGS, ids=lambda setting: setting.key)
def test_changing_a_scope_setting_keeps_the_paperqa_index_name(setting, tmp_path):
    """The reason these settings need a warning of their own: PaperQA keeps the index but changes what is in it."""
    settings = _settings_with_paper_list(tmp_path)
    name_before = settings.get_index_name()
    value_before = setting.read(settings)

    CHANGES[setting.key](settings)

    assert setting.read(settings) != value_before
    assert settings.get_index_name() == name_before


@pytest.mark.parametrize("setting", settings_groups.SCOPE_SETTINGS, ids=lambda setting: setting.key)
def test_the_scope_help_says_the_index_changes_only_when_it_is_rebuilt(setting):
    """Questions search the index as it is (#130), so nothing is added or removed until a rebuild."""
    assert "**Rebuild the index**" in setting.help
    assert "next question" not in setting.help


def test_paperqa_rebuilds_remove_pdfs_out_of_scope_by_default():
    """A rebuild (get_directory_index with build=True) removes index entries out of scope only with this on."""
    from paperqa.settings import IndexSettings

    assert IndexSettings().sync_with_paper_directory is True


def test_scope_settings_are_real_paperqa_index_settings():
    from paperqa.settings import IndexSettings

    assert {"manifest_file", "recurse_subdirectories"} <= set(IndexSettings.model_fields)


def test_the_paper_list_starts_in_use_when_paperqa_has_one_and_off_when_it_has_none(tmp_path):
    setting = settings_groups.get_setting("manifest_scope")

    assert settings_groups.current_choice(setting, _settings_with_paper_list(tmp_path)) == "manifest"
    assert settings_groups.current_choice(setting, _real_settings(tmp_path)) == "all"


def test_choosing_all_pdfs_takes_the_paper_list_out_of_paperqa_s_settings(tmp_path):
    settings = _settings_with_paper_list(tmp_path)

    settings_groups.get_setting("manifest_scope").write(settings, "all")

    assert settings.agent.index.manifest_file is None


def test_choosing_the_paper_list_leaves_the_path_the_app_set(tmp_path):
    settings = _settings_with_paper_list(tmp_path)

    settings_groups.get_setting("manifest_scope").write(settings, "manifest")

    assert settings.agent.index.manifest_file == tmp_path / "manifest.csv"


def test_the_paper_list_choice_offers_the_list_or_all_pdfs_by_name():
    check = settings_groups.get_setting("manifest_scope").check

    assert check.values == ("manifest", "all")
    assert check.name_for("manifest") == "Only the PDFs in the paper list"
    assert check.name_for("all") == "All PDFs in the folder"


def test_writing_the_subfolder_choice_changes_paperqa_s_recursion(tmp_path):
    settings = _real_settings(tmp_path)

    settings_groups.get_setting("recurse_subdirectories").write(settings, False)

    assert settings.agent.index.recurse_subdirectories is False
    assert settings_groups.current_choice(settings_groups.get_setting("recurse_subdirectories"), settings) is False


@pytest.mark.parametrize("key", ["manifest_scope", "recurse_subdirectories"])
def test_scope_settings_can_be_saved_and_say_the_index_changes_in_place(key):
    setting = settings_groups.get_setting(key)

    assert setting.write is not None
    assert not setting.needs_reindex
    assert "same index" in setting.help
    assert setting.help.endswith(".")


@pytest.mark.parametrize(
    ("key", "value", "allowed"),
    [
        ("manifest_scope", "manifest", True),
        ("manifest_scope", "all", True),
        ("manifest_scope", "some", False),
        ("recurse_subdirectories", False, True),
        ("recurse_subdirectories", "no", False),
    ],
)
def test_scope_values_are_checked_like_every_saved_setting(key, value, allowed):
    assert settings_groups.usable(key, value) is allowed
