"""Quality presets: named starting points for how thoroughly PaperQA searches and answers.

Each preset follows one of PaperQA's own bundled presets (paperqa/configs/*.json) or its defaults, and
sets five PaperQA settings:

- `answer.evidence_k`, how many passages are gathered and read;
- `answer.answer_max_sources`, how many of them the answer may cite;
- `answer.answer_length`, how long the answer is asked to be;
- `answer.evidence_summary_length`, how long each passage's summary is;
- `agent.agent_type`, whether a model plans each search ("ToolSelector") or PaperQA runs its steps in a
  fixed order ("fake": a few suggested searches, gather evidence, answer, then one call to finish), which
  saves a couple of the planning calls.

The bundled presets also change settings that belong elsewhere in the app, and those are left out: the
number of requests sent at once belongs to the speed limits, chunk size and document details change the
index, and the debug preset's Anthropic model replaces the answer model. Template, grounding and the
before-answer prompt are left alone too, so a preset combines with any of them.

The first three settings also have controls of their own. They are applied after the preset, so a
number changed after picking a preset wins, and the preset then reads as NO_PRESET.
"""

from dataclasses import dataclass

# PaperQA's agent types.
PLANNED_SEARCH = "ToolSelector"
FIXED_STEPS = "fake"

SHORT_ANSWER = "50 to 100 words"
STANDARD_ANSWER = "about 200 words, but can be longer"
SHORT_SUMMARY = "25 to 50 words"
STANDARD_SUMMARY = "about 100 words"


@dataclass(frozen=True)
class QualityPreset:
    name: str
    description: str
    # The PaperQA preset it follows: a file in paperqa/configs, or PaperQA's defaults.
    based_on: str
    evidence_k: int
    answer_max_sources: int
    answer_length: str
    summary_length: str
    agent_type: str


PRESETS = (
    QualityPreset(
        name="Fast",
        description=(
            "Reads 5 passages, cites up to 3 and writes a short answer. PaperQA searches in fixed steps "
            "instead of having the model plan each search, which saves several model calls and reads the "
            "passages only once."
        ),
        based_on="fast.json",
        evidence_k=5,
        answer_max_sources=3,
        answer_length=SHORT_ANSWER,
        summary_length=SHORT_SUMMARY,
        agent_type=FIXED_STEPS,
    ),
    QualityPreset(
        name="Balanced",
        description=(
            "PaperQA's usual settings: 10 passages, up to 5 sources, an answer of about 200 words, and the "
            "model plans its own searches."
        ),
        based_on="PaperQA's defaults",
        evidence_k=10,
        answer_max_sources=5,
        answer_length=STANDARD_ANSWER,
        summary_length=STANDARD_SUMMARY,
        agent_type=PLANNED_SEARCH,
    ),
    QualityPreset(
        name="High quality",
        description=(
            "Reads 20 passages, so it can find evidence that 10 would miss, at roughly twice the cost and "
            "time. The answer still cites up to 5 of them."
        ),
        based_on="high_quality.json",
        evidence_k=20,
        answer_max_sources=5,
        answer_length=STANDARD_ANSWER,
        summary_length=STANDARD_SUMMARY,
        agent_type=PLANNED_SEARCH,
    ),
    QualityPreset(
        name="Debug",
        description=(
            "The smallest real run: 2 passages, 2 sources and a short answer, with the model planning its "
            "searches as usual. Use it to check the app and your PDFs work, not for research."
        ),
        based_on="debug.json",
        evidence_k=2,
        answer_max_sources=2,
        answer_length=SHORT_ANSWER,
        summary_length=SHORT_SUMMARY,
        agent_type=PLANNED_SEARCH,
    ),
)

DEFAULT_PRESET = "Balanced"

# Shown when the settings in use match no preset, for example after changing the number of passages.
NO_PRESET = "Custom"

# The setting keys a preset sets that also have controls of their own. Picking a preset forgets their
# saved values, so the preset's own values are used.
KEYS_WITH_CONTROLS = ("evidence_k", "answer_max_sources", "answer_length")

AGENT_TYPE_NAMES = {PLANNED_SEARCH: "Planned by the model", FIXED_STEPS: "Fixed steps"}


def get_preset(name: str) -> QualityPreset:
    for preset in PRESETS:
        if preset.name == name:
            return preset
    raise KeyError(name)


def apply_preset(settings: object, name: str) -> None:
    """Put a preset's values on a PaperQA Settings object."""
    preset = get_preset(name)
    settings.answer.evidence_k = preset.evidence_k
    settings.answer.answer_max_sources = preset.answer_max_sources
    settings.answer.answer_length = preset.answer_length
    settings.answer.evidence_summary_length = preset.summary_length
    settings.agent.agent_type = preset.agent_type


def _values(settings: object) -> tuple:
    return (
        settings.answer.evidence_k,
        settings.answer.answer_max_sources,
        settings.answer.answer_length,
        settings.answer.evidence_summary_length,
        settings.agent.agent_type,
    )


def read_preset_name(settings: object) -> str:
    """Return the name of the preset whose values are all in use, or NO_PRESET."""
    in_use = _values(settings)
    for preset in PRESETS:
        if in_use == (
            preset.evidence_k,
            preset.answer_max_sources,
            preset.answer_length,
            preset.summary_length,
            preset.agent_type,
        ):
            return preset.name
    return NO_PRESET


def describe_agent_type(agent_type: str) -> str:
    """Return how PaperQA's agent type decides the searches, in plain words, or the type as it is."""
    return AGENT_TYPE_NAMES.get(agent_type, agent_type)
