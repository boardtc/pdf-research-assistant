"""Which settings apply to the next question, and which need the index rebuilt.

PaperQA names an index after the settings that decide how PDFs are read and stored
(Settings.get_index_name). Changing any of those makes PaperQA look for a different
index, so every PDF has to be indexed again. Those settings are REINDEX_SETTINGS.
Every other setting only changes how a question is answered: INSTANT_SETTINGS.

To add a setting, add one entry to the group it belongs to. The tests check both
groups against PaperQA itself. A setting the user can change from the sidebar also
has `write` and `check`; saved_settings.py then saves it and applies it whenever
settings are built, so a saved index setting changes the index PaperQA looks for.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from pdf_research_assistant import rate_profiles


@dataclass(frozen=True)
class WholeNumber:
    """Accepts a whole number from low to high, inclusive."""

    low: int
    high: int
    # How far one click of the number box moves.
    step: int = 1

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
class OnOrOff:
    """Accepts only True or False."""

    def __call__(self, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"Expected on or off, got {value!r}.")
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
    # Returns the value in use in the form `check` accepts, when `read` returns it another way.
    choice: Callable[[object], object] | None = None


# How PaperQA's images-and-tables options are saved. The names match describe_value's text for each.
MEDIA_OFF = "off"
MEDIA_CHOICES = (
    ("Off", MEDIA_OFF),
    ("On without enrichment", "on_without_enrichment"),
    ("On with enrichment", "on_with_enrichment"),
)


def read_media_choice(settings: object) -> str:
    """Return the images-and-tables option in use, as it is saved.

    PaperQA takes True and False as well as its own options; True means on with enrichment.
    """
    value = settings.parsing.multimodal
    if isinstance(value, bool):
        return "on_with_enrichment" if value else MEDIA_OFF
    return value.name.lower()


def write_media_choice(settings: object, value: str) -> None:
    """Put a saved images-and-tables option onto a PaperQA Settings object.

    Off is written as False, which is how the app has always set it: PaperQA names the index after
    the text of this value, so its own "off" option would start a new index for the same setting.
    """
    from paperqa.settings import MultimodalOptions

    settings.parsing.multimodal = False if value == MEDIA_OFF else MultimodalOptions[value.upper()]


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
    Setting(
        key="rate_profile",
        label="Speed limits",
        help=(
            "How fast PaperQA may send requests to OpenAI. Pick the profile that fits your account's limits, "
            "which you can see at platform.openai.com/settings/organization/limits. A profile above your limits "
            "makes OpenAI turn requests away, which slows things down; one below them makes long runs, such as "
            "exhaustive search, much slower than they need to be. It does not change the cost of a question. "
            "With a local model there is no account limit, so only the number of requests sent at once matters."
        ),
        needs_reindex=False,
        read=rate_profiles.read_profile_name,
        write=rate_profiles.apply_profile,
        check=OneOf(tuple((profile.name, profile.name) for profile in rate_profiles.PROFILES)),
    ),
)

# One entry for each part of PaperQA's index name, in the same order. Each help text ends by saying
# whether the setting affects indexing only or also affects questions.
REINDEX_SETTINGS = (
    Setting(
        key="paper_directory",
        label="PDF folder",
        help="The folder your PDFs are read from. It is set with `PAPER_DIR` in `.env`. Affects indexing only.",
        needs_reindex=True,
        read=lambda settings: settings.agent.index.paper_directory,
    ),
    Setting(
        key="use_absolute_paper_directory",
        label="Full folder path kept in the index",
        help=(
            "Whether the index records each PDF by its full path instead of its path inside the PDF folder. "
            "The app relies on this being off, so it cannot be changed here. Affects indexing only."
        ),
        needs_reindex=True,
        read=lambda settings: settings.agent.index.use_absolute_paper_directory,
    ),
    Setting(
        key="embedding",
        label="Embedding model",
        help=(
            "The OpenAI model that turns text into numbers so passages can be found by meaning. The large "
            "model can find passages the small one misses, but indexing with it costs about six times as "
            "much. Also affects questions: each question is turned into numbers by the same model."
        ),
        needs_reindex=True,
        read=lambda settings: settings.embedding,
        write=lambda settings, value: setattr(settings, "embedding", value),
        check=OneOf(tuple((model, model) for model in ("text-embedding-3-small", "text-embedding-3-large"))),
    ),
    Setting(
        key="parse_pdf",
        label="PDF reader",
        help=(
            "The code that pulls the text out of each PDF. It is part of the app, so it cannot be changed "
            "here. Affects indexing only."
        ),
        needs_reindex=True,
        read=lambda settings: settings.parsing.parse_pdf,
    ),
    Setting(
        key="chunk_chars",
        label="Chunk size (characters)",
        help=(
            "How long each stored passage of a PDF is. Smaller chunks make each passage more focused; larger "
            "ones keep more of the surrounding text together. Also affects questions: the model reads whole "
            "passages, so larger chunks make each question cost more."
        ),
        needs_reindex=True,
        read=lambda settings: settings.parsing.reader_config["chunk_chars"],
        write=lambda settings, value: settings.parsing.reader_config.update(chunk_chars=value),
        # PaperQA's default is 5000 and its presets go up to 9000. The lowest chunk stays above the
        # highest overlap, because PaperQA's chunking never finishes when the two are equal.
        check=WholeNumber(2000, 20000, step=500),
    ),
    Setting(
        key="overlap",
        label="Chunk overlap (characters)",
        help=(
            "How much text neighbouring passages share, so a sentence is not cut off between them. "
            "More overlap means more passages to store for the same PDF. Affects indexing only."
        ),
        needs_reindex=True,
        read=lambda settings: settings.parsing.reader_config["overlap"],
        write=lambda settings, value: settings.parsing.reader_config.update(overlap=value),
        # PaperQA's default is 250 and its presets go up to 1750.
        check=WholeNumber(0, 1750, step=50),
    ),
    Setting(
        key="full_page",
        label="Whole-page pictures",
        help=(
            "When images and tables are on, whether each page is kept as one picture of the whole page "
            "instead of a separate picture of each figure and table. It does nothing while images and "
            "tables are off. Also affects questions: whole-page pictures are larger, so passages that "
            "carry them cost more to read."
        ),
        needs_reindex=True,
        read=lambda settings: settings.parsing.reader_config.get("full_page", False),
        write=lambda settings, value: settings.parsing.reader_config.update(full_page=value),
        check=OnOrOff(),
    ),
    Setting(
        key="multimodal",
        label="Images and tables",
        help=(
            "Whether images and tables are taken from the PDFs as well as text. Enrichment means a model "
            "writes a description of each image and table while indexing, so a search can find them; that "
            "is an extra paid model call for every one. Without enrichment they are stored but only found "
            "through the text around them. Also affects questions: the pictures are sent to the model "
            "with their passages, which costs more."
        ),
        needs_reindex=True,
        read=lambda settings: settings.parsing.multimodal,
        write=write_media_choice,
        check=OneOf(MEDIA_CHOICES),
        choice=read_media_choice,
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


def current_choice(setting: Setting, settings: object) -> object:
    """Return the setting's value in use, in the form its `check` accepts and its control offers."""
    return (setting.choice or setting.read)(settings)


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


def usable(key: str, value: object) -> bool:
    """Return whether a value chosen in the app can be put on this setting."""
    try:
        setting = get_setting(key)
    except KeyError:
        return False
    if setting.write is None:
        return False
    try:
        setting.check(value)
    except ValueError:
        return False
    return True


def apply_usable(settings: object, values: dict) -> list[str]:
    """Put each usable value onto a PaperQA Settings object, skip the rest, and return the keys applied."""
    applied = []
    for key, value in values.items():
        if usable(key, value):
            get_setting(key).write(settings, value)
            applied.append(key)
    return applied


ONE_OFF_SOURCE = "This question only"


def record_answer_settings(settings: object, one_off: Iterable[str] = ()) -> dict:
    """Return what to keep with an answer: the index it searched and each instant setting, as plain text.

    Only the instant settings are listed, because the index name already stands for every rebuild setting.
    "source" says where a value came from: ONE_OFF_SOURCE for the keys in one_off, which were changed for
    this question only; None means the usual settings.
    """
    one_off = set(one_off)
    return {
        "index": settings.get_index_name(),
        "values": [
            {
                "key": setting.key,
                "label": setting.label,
                "value": describe_value(setting.read(settings)),
                "source": ONE_OFF_SOURCE if setting.key in one_off else None,
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
