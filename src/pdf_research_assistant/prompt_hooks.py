"""PaperQA's prompt hooks: extra requests it can send to the answer model around the answer itself.

- `prompts.pre`, the before-answer prompt. When it is set, PaperQA sends it to the answer model with the
  question filled in, before the answer is written. The reply is added after the passages as "Extra
  background information", so the request that writes the answer sees it too. It is a request of its own,
  so it costs one more answer-model call, and it goes with the same system prompt as the answer, which the
  grounding mode sets.
- `prompts.post`, the after-answer prompt, is not offered. In the installed PaperQA (Docs.aquery) it is
  filled in with the question only, never the answer, and its reply then replaces the answer and is added
  a second time, so the answer that was written is lost.

Both differ from the answer template, which changes the request that writes the answer (`prompts.qa`) and
adds no request of its own.
"""

import string
from dataclasses import dataclass

PRE_PROMPT_EXAMPLE = (
    "List the technical terms and abbreviations in this question and say in a few words what each means: "
    "{question}"
)

POST_PROMPT_NOTE = (
    "PaperQA also has an after-answer prompt (prompts.post), which is not offered here: in the PaperQA "
    "version this app uses it is given the question but not the answer, and its reply replaces the answer."
)


@dataclass(frozen=True)
class PromptText:
    """Accepts a prompt as text, up to `limit` characters, using only the named `{fields}`.

    Space around the text is dropped. Empty text is accepted and means the prompt is off. Literal braces
    must be doubled, as in Python's str.format, which is how PaperQA fills the prompt in.
    """

    limit: int
    fields: tuple[str, ...]

    def __call__(self, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Expected text, got {value!r}.")
        text = value.strip()
        if len(text) > self.limit:
            raise ValueError(f"The prompt can be at most {self.limit} characters long; it has {len(text)}.")
        allowed = ", ".join(f"{{{field}}}" for field in self.fields)
        try:
            parts = list(string.Formatter().parse(text))
        except ValueError as exc:
            raise ValueError("The prompt has unmatched braces. Write {{ and }} for braces of your own.") from exc
        for _literal, field, spec, conversion in parts:
            if field is None:
                continue
            if field not in self.fields or spec or conversion:
                raise ValueError(
                    f"The prompt can only fill in {allowed}; write {{{{ and }}}} for braces of your own."
                )
        return text


# PaperQA fills the before-answer prompt in with the question alone.
PRE_PROMPT_CHECK = PromptText(limit=2000, fields=("question",))


def read_pre_prompt(settings: object) -> str:
    """Return the before-answer prompt in use as the sidebar box shows it: empty when there is none."""
    return settings.prompts.pre or ""


def write_pre_prompt(settings: object, value: str) -> None:
    """Put a before-answer prompt on a PaperQA Settings object; empty text turns it off."""
    settings.prompts.pre = value or None
