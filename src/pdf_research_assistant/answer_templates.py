"""Answer templates: named ways of shaping the answer for a research task.

PaperQA writes the answer with its `qa` prompt (`settings.prompts.qa`), which is given the
passages it found and the question. A template is that prompt with PaperQA's style sentence
swapped for an instruction of its own. Everything else stays as PaperQA wrote it: the passages,
the question, the answer length and the rules for citing sources. So the same passages are found
and read whichever template is chosen; only the way the answer is written changes.

The default template is PaperQA's own prompt, unchanged.
"""

from dataclasses import dataclass

# The sentence in PaperQA's answer prompt that sets how the answer is written. Templates replace it.
PAPERQA_STYLE = "Write in the style of a scientific article, with concise sentences and coherent paragraphs. "


@dataclass(frozen=True)
class AnswerTemplate:
    name: str
    description: str
    # Replaces PAPERQA_STYLE in PaperQA's answer prompt; None keeps PaperQA's prompt as it is.
    instruction: str | None


TEMPLATES = (
    AnswerTemplate(
        name="Direct answer",
        description="PaperQA's usual answer: a short, article-style reply to the question, citing its sources.",
        instruction=None,
    ),
    AnswerTemplate(
        name="Study notes",
        description="Bullet-point notes under short headings, covering the key terms, findings and methods.",
        instruction=(
            "Write the answer as study notes: start with a one-sentence summary, then give short bullet points "
            "grouped under brief headings, covering the key terms, findings and methods. "
        ),
    ),
    AnswerTemplate(
        name="Compare sources",
        description="What each source says about the question in turn, then where they agree and where they differ.",
        instruction=(
            "Compare the sources rather than giving one combined answer: say in turn what each source finds or "
            "argues about the question, then end with a short paragraph on where they agree and where they differ. "
        ),
    ),
    AnswerTemplate(
        name="Contradictions only",
        description="Only the points where the sources disagree, with what each side claims. Says so if they agree.",
        instruction=(
            "Report only where the sources disagree with or contradict each other about the question: for each "
            "disagreement, say what each side claims. Leave out points the sources agree on. If the sources do "
            "not disagree, say so plainly. "
        ),
    ),
)

DEFAULT_TEMPLATE = "Direct answer"

# Shown when the answer prompt in use matches no template, for example after a settings file was edited by hand.
NO_TEMPLATE = "Not a named template"


def get_template(name: str) -> AnswerTemplate:
    for template in TEMPLATES:
        if template.name == name:
            return template
    raise KeyError(name)


def template_prompt(template: AnswerTemplate) -> str:
    """Return PaperQA's answer prompt with the template's instruction in place of its style sentence."""
    # Read from PaperQA each time, so the prompts follow the installed version.
    from paperqa import prompts

    if template.instruction is None:
        return prompts.qa_prompt
    if PAPERQA_STYLE not in prompts.qa_prompt:
        raise RuntimeError("PaperQA's answer prompt no longer has the style sentence that templates replace.")
    return prompts.qa_prompt.replace(PAPERQA_STYLE, template.instruction)


def apply_template(settings: object, name: str) -> None:
    """Put a template's answer prompt on a PaperQA Settings object."""
    settings.prompts.qa = template_prompt(get_template(name))


def read_template_name(settings: object) -> str:
    """Return the name of the template whose answer prompt is in use, or NO_TEMPLATE."""
    for template in TEMPLATES:
        if settings.prompts.qa == template_prompt(template):
            return template.name
    return NO_TEMPLATE
