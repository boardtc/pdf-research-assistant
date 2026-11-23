from paperqa import Settings
from paperqa.prompts import qa_prompt
from paperqa.settings import PromptSettings
import pytest

from pdf_research_assistant import answer_templates


def test_templates_have_unique_names_and_the_default_is_one_of_them():
    names = [template.name for template in answer_templates.TEMPLATES]

    assert names == ["Direct answer", "Study notes", "Compare sources", "Contradictions only"]
    assert answer_templates.DEFAULT_TEMPLATE in names


def test_every_template_says_what_it_is_for():
    assert all(template.description.strip() for template in answer_templates.TEMPLATES)


def test_the_default_template_is_paperqa_s_own_answer_prompt():
    assert answer_templates.template_prompt(answer_templates.get_template("Direct answer")) == qa_prompt


def test_every_other_template_swaps_only_the_style_sentence_of_paperqa_s_answer_prompt():
    for template in answer_templates.TEMPLATES[1:]:
        prompt = answer_templates.template_prompt(template)

        assert answer_templates.PAPERQA_STYLE not in prompt
        assert template.instruction in prompt
        assert prompt.replace(template.instruction, answer_templates.PAPERQA_STYLE) == qa_prompt


def test_every_template_prompt_is_one_paperqa_accepts_and_keeps_its_citation_rules():
    for template in answer_templates.TEMPLATES:
        prompt = answer_templates.template_prompt(template)

        # PaperQA checks the prompt only uses the variables its own answer prompt has.
        assert PromptSettings(qa=prompt).qa == prompt
        for variable in ("{context}", "{question}", "{answer_length}", "{example_citation}", "{prior_answer_prompt}"):
            assert variable in prompt
        assert "citation keys" in prompt


def test_template_prompt_refuses_when_paperqa_s_answer_prompt_has_no_style_sentence(monkeypatch):
    import paperqa.prompts

    monkeypatch.setattr(paperqa.prompts, "qa_prompt", "Answer {question} from {context}.")

    with pytest.raises(RuntimeError, match="style sentence"):
        answer_templates.template_prompt(answer_templates.get_template("Study notes"))


def test_get_template_returns_the_named_template():
    assert answer_templates.get_template("Study notes").name == "Study notes"


def test_get_template_raises_for_an_unknown_name():
    with pytest.raises(KeyError):
        answer_templates.get_template("Poem")


def test_apply_template_puts_the_template_s_prompt_on_paperqa_s_answer_prompt():
    settings = Settings()

    answer_templates.apply_template(settings, "Compare sources")

    assert settings.prompts.qa == answer_templates.template_prompt(answer_templates.get_template("Compare sources"))


def test_apply_template_leaves_paperqa_s_other_prompts_alone():
    settings = Settings()
    before = settings.prompts.model_dump()

    answer_templates.apply_template(settings, "Study notes")

    after = settings.prompts.model_dump()
    assert {key for key in before if before[key] != after[key]} == {"qa"}


def test_read_template_name_finds_each_template_from_the_prompt_in_use():
    settings = Settings()
    assert answer_templates.read_template_name(settings) == "Direct answer"

    for template in answer_templates.TEMPLATES:
        answer_templates.apply_template(settings, template.name)
        assert answer_templates.read_template_name(settings) == template.name


def test_read_template_name_says_when_the_prompt_in_use_is_no_template():
    settings = Settings()
    settings.prompts.qa = "Answer {question} from {context}."

    assert answer_templates.read_template_name(settings) == answer_templates.NO_TEMPLATE
