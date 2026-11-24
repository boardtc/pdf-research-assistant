"""Grounding modes: named choices of how closely an answer must keep to the passages PaperQA found.

Each mode sets two PaperQA settings:

- `prompts.system`, the system message PaperQA sends with the request that writes the answer. A mode
  adds one instruction after PaperQA's own system prompt, saying how far the answer may go beyond the
  passages.
- `answer.evidence_relevance_score_cutoff`, the lowest relevance score (out of 10, given to each passage
  while it is summarised) a passage needs to be sent to the answer at all.

The answer prompt itself (`prompts.qa`) is left alone, because it belongs to the answer template, so any
mode can be combined with any template. The default mode is PaperQA's own settings, unchanged.
"""

from dataclasses import dataclass

# The mark PaperQA's answer.answer_filter_extra_background setting looks for. The app leaves that setting
# off, so the mark stays in the answer for the reader to see.
BACKGROUND_MARK = "(Extra background information)"


@dataclass(frozen=True)
class GroundingMode:
    name: str
    description: str
    # Added after PaperQA's system prompt; None keeps PaperQA's system prompt as it is.
    instruction: str | None
    # Passages scored below this are not sent to the answer.
    relevance_cutoff: int


# PaperQA's own phrase for declining (paperqa.prompts.CANNOT_ANSWER_PHRASE); its answer prompt asks for it
# when the context is not enough, and its agent treats an answer containing it as unsuccessful.
CANNOT_ANSWER_PHRASE = "I cannot answer"

STRICT_INSTRUCTION = (
    "Use only what the context says. Do not add facts, figures or explanations from your own knowledge, "
    "and do not draw conclusions the context does not state. Support every claim with a citation. "
    f'If the context does not directly answer the question, reply "{CANNOT_ANSWER_PHRASE}." '
    "rather than giving a guessed or partial answer."
)

EXPLORATORY_INSTRUCTION = (
    "Where it helps the reader, you may add widely accepted background knowledge that is not in the context, "
    "to explain terms or connect the findings. End each sentence that the context does not support with "
    f"{BACKGROUND_MARK} and give it no citation. If the context answers only part of the question, answer "
    "that part and say what the sources do not cover."
)

# PaperQA's default cutoff; scores of 0 are dropped before the cutoff applies, so lower would change nothing.
PAPERQA_CUTOFF = 1
# Scores tend to bunch at 0 or at 8 to 9, so 5 keeps the clearly relevant passages and drops the weak ones.
STRICT_CUTOFF = 5


MODES = (
    GroundingMode(
        name="Strictly grounded",
        description=(
            "Only what the passages say, every claim cited, and weakly relevant passages left out. "
            'Says "I cannot answer" rather than guess, so expect that more often.'
        ),
        instruction=STRICT_INSTRUCTION,
        relevance_cutoff=STRICT_CUTOFF,
    ),
    GroundingMode(
        name="Balanced",
        description="PaperQA's usual grounding: the answer is written from the passages and cites them.",
        instruction=None,
        relevance_cutoff=PAPERQA_CUTOFF,
    ),
    GroundingMode(
        name="Exploratory",
        description=(
            f"May add general background to explain or connect the findings, marked {BACKGROUND_MARK} "
            "and not cited. Answers what it can when the passages cover only part of the question."
        ),
        instruction=EXPLORATORY_INSTRUCTION,
        relevance_cutoff=PAPERQA_CUTOFF,
    ),
)

DEFAULT_MODE = "Balanced"

# Shown when the settings in use match no mode, for example after a settings file was edited by hand.
NO_MODE = "Not a named mode"


def get_mode(name: str) -> GroundingMode:
    for mode in MODES:
        if mode.name == name:
            return mode
    raise KeyError(name)


def mode_system_prompt(mode: GroundingMode) -> str:
    """Return PaperQA's system prompt with the mode's instruction after it."""
    # Read from PaperQA each time, so the prompt follows the installed version.
    from paperqa.prompts import default_system_prompt

    if mode.instruction is None:
        return default_system_prompt
    return f"{default_system_prompt} {mode.instruction}"


def apply_mode(settings: object, name: str) -> None:
    """Put a mode's system prompt and relevance cutoff on a PaperQA Settings object."""
    mode = get_mode(name)
    settings.prompts.system = mode_system_prompt(mode)
    settings.answer.evidence_relevance_score_cutoff = mode.relevance_cutoff


def read_mode_name(settings: object) -> str:
    """Return the name of the mode whose system prompt and cutoff are in use, or NO_MODE."""
    for mode in MODES:
        if (
            settings.prompts.system == mode_system_prompt(mode)
            and settings.answer.evidence_relevance_score_cutoff == mode.relevance_cutoff
        ):
            return mode.name
    return NO_MODE
