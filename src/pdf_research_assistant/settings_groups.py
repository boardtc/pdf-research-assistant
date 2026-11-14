"""Which settings apply to the next question, and which need the index rebuilt.

PaperQA names an index after the settings that decide how PDFs are read and stored
(Settings.get_index_name). Changing any of those makes PaperQA look for a different
index, so every PDF has to be indexed again. Those settings are REINDEX_SETTINGS.
Every other setting only changes how a question is answered: INSTANT_SETTINGS.

To add a setting, add one entry to the group it belongs to. The tests check both
groups against PaperQA itself. A setting the user can change from the sidebar also
has `write` and `check`; saved_settings.py then saves it and applies it to every query.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class WholeNumber:
    """Accepts a whole number from low to high, inclusive."""

    low: int
    high: int

    def __call__(self, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Expected a whole number, got {value!r}.")
        if not self.low <= value <= self.high:
            raise ValueError(f"Expected a number between {self.low} and {self.high}, got {value}.")
        return value


@dataclass(frozen=True)
class OneOf:
    """Accepts only one of the offered values. Each value has a short name to show in the sidebar."""

    # (name, value) pairs, in the order they are offered.
    choices: tuple[tuple[str, object], ...]

    @property
    def values(self) -> tuple:
        return tuple(value for _name, value in self.choices)

    def name_for(self, value: object) -> str | None:
        """Return the name of an offered value, or None if it was not offered."""
        return next((name for name, offered in self.choices if offered == value), None)

    def __call__(self, value: object) -> object:
        if self.name_for(value) is None:
            raise ValueError(f"Expected one of {list(self.values)}, got {value!r}.")
        return value


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    help: str
    needs_reindex: bool
    # Returns the setting's current value from a PaperQA Settings object.
    read: Callable[[object], object]
    # Puts a chosen value onto a PaperQA Settings object; None when the sidebar cannot change the setting.
    write: Callable[[object, object], None] | None = None
    # Returns a chosen value if it can be used, or raises ValueError.
    check: Callable[[object], object] | None = None


INSTANT_SETTINGS = (
    Setting(
        key="answer_model",
        label="Answer model",
        help="The model that reads the passages found in your PDFs and writes the answer.",
        needs_reindex=False,
        read=lambda settings: settings.llm,
    ),
    Setting(
        key="evidence_k",
        label="Passages gathered",
        help=(
            "How many passages are pulled from your PDFs and checked against the question. The model reads "
            "each one, so more passages cost more and take longer, but can find evidence that fewer would miss."
        ),
        needs_reindex=False,
        read=lambda settings: settings.answer.evidence_k,
        write=lambda settings, value: setattr(settings.answer, "evidence_k", value),
        # PaperQA sets no limit; its own presets use from 2 to 30.
        check=WholeNumber(1, 30),
    ),
    Setting(
        key="answer_max_sources",
        label="Most sources in the answer",
        help=(
            "The largest number of those passages the answer may cite. More sources make the answer draw on "
            "more of your PDFs, at a slightly higher cost, because the model is sent more text to write from."
        ),
        needs_reindex=False,
        read=lambda settings: settings.answer.answer_max_sources,
        write=lambda settings, value: setattr(settings.answer, "answer_max_sources", value),
        # PaperQA sets no limit; its own presets use from 2 to 15.
        check=WholeNumber(1, 15),
    ),
    Setting(
        key="answer_length",
        label="Answer length",
        help=(
            "How long the answer is asked to be. PaperQA puts this phrase into the request that writes the "
            "answer, so a longer answer costs a little more and takes a little longer to write, but the same "
            "passages are read either way. Exhaustive search ignores it, because it writes no combined answer."
        ),
        needs_reindex=False,
        read=lambda settings: settings.answer.answer_length,
        write=lambda settings, value: setattr(settings.answer, "answer_length", value),
        # Standard is PaperQA's default; Short is the phrase its fast preset uses.
        check=OneOf(
            (
                ("Short", "50 to 100 words"),
                ("Standard", "about 200 words, but can be longer"),
                ("Long", "about 400 words, but can be longer"),
            )
        ),
    ),
)

# One entry for each part of PaperQA's index name, in the same order.
REINDEX_SETTINGS = (
    Setting(
        key="paper_directory",
        label="PDF folder",
        help="The folder your PDFs are read from.",
        needs_reindex=True,
        read=lambda settings: settings.agent.index.paper_directory,
    ),
    Setting(
        key="use_absolute_paper_directory",
        label="Full folder path kept in the index",
        help="Whether the index records each PDF by its full path instead of its path inside the PDF folder.",
        needs_reindex=True,
        read=lambda settings: settings.agent.index.use_absolute_paper_directory,
    ),
    Setting(
        key="embedding",
        label="Embedding model",
        help="The model that turns text into numbers so passages can be found by meaning.",
        needs_reindex=True,
        read=lambda settings: settings.embedding,
    ),
    Setting(
        key="parse_pdf",
        label="PDF reader",
        help="The code that pulls the text out of each PDF.",
        needs_reindex=True,
        read=lambda settings: settings.parsing.parse_pdf,
    ),
    Setting(
        key="chunk_chars",
        label="Chunk size (characters)",
        help="How long each stored passage of a PDF is.",
        needs_reindex=True,
        read=lambda settings: settings.parsing.reader_config["chunk_chars"],
    ),
    Setting(
        key="overlap",
        label="Chunk overlap (characters)",
        help="How much text neighbouring passages share, so a sentence is not cut off between them.",
        needs_reindex=True,
        read=lambda settings: settings.parsing.reader_config["overlap"],
    ),
    Setting(
        key="full_page",
        label="Whole-page reading",
        help="Whether each PDF page is read as one piece instead of being cut into passages.",
        needs_reindex=True,
        read=lambda settings: settings.parsing.reader_config.get("full_page", False),
    ),
    Setting(
        key="multimodal",
        label="Images and tables",
        help="Whether images and tables are taken from the PDFs as well as text.",
        needs_reindex=True,
        read=lambda settings: settings.parsing.multimodal,
    ),
)

SETTINGS = INSTANT_SETTINGS + REINDEX_SETTINGS


def get_setting(key: str) -> Setting:
    for setting in SETTINGS:
        if setting.key == key:
            return setting
    raise KeyError(key)


def needs_reindex(key: str) -> bool:
    """Return whether changing this setting means the index has to be rebuilt."""
    return get_setting(key).needs_reindex


def read_values(settings: object, group: Iterable[Setting]) -> dict[str, object]:
    """Return the current value of each setting in the group, by key."""
    return {setting.key: setting.read(settings) for setting in group}


def describe_value(value: object) -> str:
    """Return a setting's value as short text a person can read."""
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, Enum):
        return value.name.replace("_", " ").capitalize()
    if callable(value):
        return getattr(value, "__name__", str(value))
    return str(value)


def describe_line(label: str, value: str) -> str:
    """Return one Markdown line with a label and a value."""
    return f"**{label}:** `{value}`"


def describe_setting(setting: Setting, settings: object) -> str:
    """Return one Markdown line with the setting's label and current value."""
    return describe_line(setting.label, describe_value(setting.read(settings)))


def record_answer_settings(settings: object) -> dict:
    """Return what to keep with an answer: the index it searched and each instant setting, as plain text.

    Only the instant settings are listed, because the index name already stands for every rebuild setting.
    "source" says where a value came from, such as a one-off change for this question; None means the
    usual settings.
    """
    return {
        "index": settings.get_index_name(),
        "values": [
            {
                "key": setting.key,
                "label": setting.label,
                "value": describe_value(setting.read(settings)),
                "source": None,
            }
            for setting in INSTANT_SETTINGS
        ],
    }


def describe_recorded_setting(item: dict) -> str:
    """Return one Markdown line for a setting kept with an answer, saying where it came from when known."""
    line = describe_line(item["label"], item["value"])
    if item.get("source"):
        line += f" :gray-badge[{item['source']}]"
    return line
