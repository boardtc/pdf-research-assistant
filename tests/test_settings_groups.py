from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from pdf_research_assistant import settings_groups


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
    "paper_directory": lambda s: setattr(s.agent.index, "paper_directory", Path(s.agent.index.paper_directory) / "other"),
    "use_absolute_paper_directory": lambda s: setattr(s.agent.index, "use_absolute_paper_directory", True),
    "embedding": lambda s: setattr(s, "embedding", "text-embedding-3-large"),
    "parse_pdf": lambda s: setattr(s.parsing, "parse_pdf", _other_pdf_reader),
    "chunk_chars": lambda s: s.parsing.reader_config.update(chunk_chars=1234),
    "overlap": lambda s: s.parsing.reader_config.update(overlap=12),
    "full_page": lambda s: s.parsing.reader_config.update(full_page=True),
    "multimodal": _set_multimodal,
}


def test_every_setting_is_in_exactly_one_group():
    instant = [setting.key for setting in settings_groups.INSTANT_SETTINGS]
    reindex = [setting.key for setting in settings_groups.REINDEX_SETTINGS]

    assert instant == ["answer_model"]
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
    assert [setting.key for setting in settings_groups.SETTINGS] == instant + reindex
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
    assert settings_groups.read_values(settings, settings_groups.INSTANT_SETTINGS) == {"answer_model": settings.llm}


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
