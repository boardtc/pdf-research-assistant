from unittest import mock

from paperqa import Settings
import pytest

from pdf_research_assistant import prompt_hooks


def test_the_example_before_answer_prompt_is_accepted():
    assert prompt_hooks.PRE_PROMPT_CHECK(prompt_hooks.PRE_PROMPT_EXAMPLE) == prompt_hooks.PRE_PROMPT_EXAMPLE


def test_a_before_answer_prompt_keeps_its_text_without_surrounding_space():
    assert prompt_hooks.PRE_PROMPT_CHECK("  List the terms in: {question}\n") == "List the terms in: {question}"


def test_a_before_answer_prompt_may_leave_the_question_out():
    assert prompt_hooks.PRE_PROMPT_CHECK("Name three review articles on this topic.") == (
        "Name three review articles on this topic."
    )


@pytest.mark.parametrize("value", ["", "   \n "])
def test_an_empty_before_answer_prompt_means_off(value):
    assert prompt_hooks.PRE_PROMPT_CHECK(value) == ""


def test_literal_braces_are_allowed_when_doubled():
    assert prompt_hooks.PRE_PROMPT_CHECK("Use {{curly}} braces for {question}") == "Use {{curly}} braces for {question}"


@pytest.mark.parametrize("value", [None, 3, ["{question}"]])
def test_a_before_answer_prompt_must_be_text(value):
    with pytest.raises(ValueError, match="Expected text"):
        prompt_hooks.PRE_PROMPT_CHECK(value)


def test_a_before_answer_prompt_has_a_length_limit():
    limit = prompt_hooks.PRE_PROMPT_CHECK.limit

    assert prompt_hooks.PRE_PROMPT_CHECK("x" * limit) == "x" * limit
    with pytest.raises(ValueError, match=f"{limit} characters"):
        prompt_hooks.PRE_PROMPT_CHECK("x" * (limit + 1))


@pytest.mark.parametrize("value", ["Summarise {answer}", "Terms in {question} and {context}", "{0} first"])
def test_a_before_answer_prompt_may_only_use_the_question(value):
    with pytest.raises(ValueError, match=r"\{question\}"):
        prompt_hooks.PRE_PROMPT_CHECK(value)


@pytest.mark.parametrize("value", ["A { left open", "A } on its own", "{question!z}"])
def test_a_before_answer_prompt_with_broken_braces_is_refused(value):
    with pytest.raises(ValueError, match="braces"):
        prompt_hooks.PRE_PROMPT_CHECK(value)


def test_writing_a_before_answer_prompt_sets_paperqa_s_pre_prompt():
    settings = Settings()

    prompt_hooks.write_pre_prompt(settings, "List the terms in: {question}")

    assert settings.prompts.pre == "List the terms in: {question}"
    assert prompt_hooks.read_pre_prompt(settings) == "List the terms in: {question}"


def test_writing_an_empty_before_answer_prompt_turns_paperqa_s_pre_prompt_off():
    settings = Settings()
    settings.prompts.pre = "Something"

    prompt_hooks.write_pre_prompt(settings, "")

    assert settings.prompts.pre is None
    assert prompt_hooks.read_pre_prompt(settings) == ""


def test_paperqa_has_no_pre_prompt_by_default():
    assert prompt_hooks.read_pre_prompt(Settings()) == ""


def test_every_accepted_prompt_formats_the_way_paperqa_formats_it():
    # Docs.aquery formats the pre-prompt with the question alone.
    for text in (prompt_hooks.PRE_PROMPT_EXAMPLE, "No question here", "Use {{braces}} with {question}"):
        prompt_hooks.PRE_PROMPT_CHECK(text).format(question="Why?")


def _ask_with_fake_model(settings):
    """Run PaperQA's own Docs.aquery on no papers with a stand-in model; return the calls and the answer."""
    import asyncio

    from lmi import LLMResult
    from paperqa import Docs

    calls = []

    async def call_single(messages, name=None, **_kwargs):
        calls.append((name, [message.content for message in messages]))
        return LLMResult(model="fake", text=f"<{name} reply>", prompt_count=1, completion_count=1)

    model = mock.Mock(call_single=call_single)
    settings.answer.get_evidence_if_no_contexts = False
    session = asyncio.run(
        Docs().aquery("Why?", settings=settings, llm_model=model, summary_llm_model=model, embedding_model=model)
    )
    return calls, session.raw_answer


def test_paperqa_sends_the_before_answer_prompt_with_the_system_prompt_and_passes_its_reply_on():
    settings = Settings()
    settings.prompts.system = "Grounding system prompt"
    prompt_hooks.write_pre_prompt(settings, "List the terms in: {question}")

    calls, _answer = _ask_with_fake_model(settings)

    assert [name for name, _ in calls] == ["pre", "answer"]
    assert calls[0][1] == ["Grounding system prompt", "List the terms in: Why?"]
    assert "Extra background information: <pre reply>" in calls[1][1][1]


def test_paperqa_s_after_answer_prompt_still_loses_the_answer():
    # The reason prompts.post is not offered. If this fails, PaperQA has changed and post may now be usable.
    settings = Settings()
    settings.prompts.post = "Check the answer to: {question}"

    calls, answer = _ask_with_fake_model(settings)

    assert calls[-1] == ("post", [settings.prompts.system, "Check the answer to: Why?"])
    assert answer == "<post reply>\n\n<post reply>"


def test_the_after_answer_note_names_paperqa_s_setting_and_why_it_is_not_offered():
    assert "prompts.post" in prompt_hooks.POST_PROMPT_NOTE
    assert "replaces the answer" in prompt_hooks.POST_PROMPT_NOTE
