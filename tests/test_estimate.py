from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import litellm
import pytest
from paperqa import Settings
from paperqa.settings import AgentSettings

from pdf_research_assistant import estimate, rate_profiles

MODEL = "gpt-4o-mini"


def _paperqa_settings(model: str = MODEL, profile: str | None = rate_profiles.DEFAULT_PROFILE, **answer) -> Settings:
    """Real PaperQA settings, built the way bootstrap.build_settings builds them, with answer changes."""
    settings = Settings(llm=model, summary_llm=model, agent=AgentSettings(agent_llm=model))
    if profile:
        rate_profiles.apply_profile(settings, profile)
    for name, value in answer.items():
        setattr(settings.answer, name, value)
    return settings


def _rate_settings(rate_limit: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        summary_llm=MODEL, summary_llm_config={"rate_limit": rate_limit} if rate_limit is not None else None
    )


def _estimate(**changes) -> estimate.Estimate:
    values = dict(
        passages=10,
        sources=5,
        answer_words=200,
        input_tokens=20000,
        output_tokens=2000,
        cost_usd=0.005,
        seconds=25,
        model=MODEL,
        local=False,
    )
    return estimate.Estimate(**(values | changes))


def test_get_tokens_per_minute_parses_the_configured_limit():
    assert estimate.get_tokens_per_minute(_rate_settings({MODEL: "30000 per 1 minute"})) == 30000


def test_get_tokens_per_minute_converts_other_periods_to_minutes():
    assert estimate.get_tokens_per_minute(_rate_settings({MODEL: "100 per 1 second"})) == 6000


def test_get_tokens_per_minute_returns_none_when_no_config_exists():
    assert estimate.get_tokens_per_minute(_rate_settings()) is None


def test_get_tokens_per_minute_returns_none_when_model_has_no_limit():
    assert estimate.get_tokens_per_minute(_rate_settings({"other": "1 per 1 minute"})) is None


def test_get_prices_returns_litellm_prices():
    prices = {MODEL: {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}}
    with mock.patch.object(litellm, "model_cost", prices):
        assert estimate.get_prices(MODEL) == prices[MODEL]


def test_get_prices_returns_none_for_an_unknown_model():
    with mock.patch.object(litellm, "model_cost", {}):
        assert estimate.get_prices(MODEL) is None


def test_get_prices_returns_none_when_a_price_is_missing():
    with mock.patch.object(litellm, "model_cost", {MODEL: {"input_cost_per_token": 1.0}}):
        assert estimate.get_prices(MODEL) is None


@pytest.mark.parametrize(
    ("phrase", "words"),
    [
        ("about 200 words, but can be longer", 200),
        ("50 to 100 words", 100),
        ("a short paragraph", estimate.FALLBACK_WORDS),
    ],
)
def test_count_words_takes_the_largest_number_in_the_phrase(phrase, words):
    assert estimate.count_words(phrase) == words


def test_call_cost_uses_the_model_prices():
    prices = {MODEL: {"input_cost_per_token": 0.5, "output_cost_per_token": 2.0}}
    with mock.patch.object(litellm, "model_cost", prices):
        assert estimate.call_cost(MODEL, 10, 3) == 11.0


def test_call_cost_is_unknown_without_prices():
    with mock.patch.object(litellm, "model_cost", {}):
        assert estimate.call_cost(MODEL, 10, 3) is None


def test_call_seconds_matches_the_measured_time_for_a_typical_reply():
    assert estimate.call_seconds(estimate.TYPICAL_OUTPUT_TOKENS) == pytest.approx(estimate.SECONDS_PER_CALL)


def test_paperqa_defaults_come_out_balanced():
    result = estimate.estimate_question(_paperqa_settings())

    assert (result.passages, result.sources, result.answer_words) == (10, 5, 200)
    assert result.cost_usd == pytest.approx(0.0046, abs=0.0005)
    assert estimate.label_for(result) == estimate.BALANCED


def test_fewer_passages_and_short_answers_come_out_fast():
    settings = _paperqa_settings(evidence_k=5, answer_length="50 to 100 words")

    assert estimate.label_for(estimate.estimate_question(settings)) == estimate.FAST


def test_thirty_passages_come_out_higher_cost():
    settings = _paperqa_settings(evidence_k=30, answer_max_sources=15)

    assert estimate.label_for(estimate.estimate_question(settings)) == estimate.HIGHER_COST


def test_low_speed_limits_come_out_slower_at_the_same_cost():
    usual = estimate.estimate_question(_paperqa_settings())
    low = estimate.estimate_question(_paperqa_settings(profile="Low limits"))

    assert low.cost_usd == usual.cost_usd
    assert low.seconds > usual.seconds
    assert estimate.label_for(low) == estimate.SLOWER


def test_a_dearer_model_comes_out_higher_cost():
    assert estimate.label_for(estimate.estimate_question(_paperqa_settings(model="gpt-4o"))) == estimate.HIGHER_COST


@pytest.mark.parametrize(
    "change",
    [{"evidence_k": 20}, {"answer_max_sources": 10}, {"answer_length": "about 400 words, but can be longer"}],
)
def test_each_answer_setting_raises_the_cost(change):
    usual = estimate.estimate_question(_paperqa_settings())

    assert estimate.estimate_question(_paperqa_settings(**change)).cost_usd > usual.cost_usd


def test_sources_cannot_exceed_passages():
    result = estimate.estimate_question(_paperqa_settings(evidence_k=3, answer_max_sources=5))

    assert result.sources == 3


def test_passages_go_out_in_batches_of_the_requests_sent_at_once():
    settings = _paperqa_settings(profile=None, evidence_k=10, max_concurrent_requests=5)
    summary_seconds = estimate.call_seconds(estimate.count_words("about 100 words") * estimate.TOKENS_PER_WORD)
    rest = estimate.call_seconds(200 * estimate.TOKENS_PER_WORD) + estimate.AGENT_CALLS * estimate.call_seconds(
        estimate.AGENT_OUTPUT_TOKENS
    )

    # Without a profile there is no token rate limit, so only the calls set the time.
    assert estimate.estimate_question(settings).seconds == pytest.approx(2 * summary_seconds + rest)


def test_the_token_rate_limit_can_set_the_time():
    settings = _paperqa_settings(profile="Low limits")
    result = estimate.estimate_question(settings)
    limited = (result.input_tokens * estimate.LIMITER_TOKEN_FACTOR + result.output_tokens) / 30000 * 60

    assert result.seconds == pytest.approx(limited, rel=0.001)


def test_a_local_model_has_no_cost():
    result = estimate.estimate_question(_paperqa_settings(), local=True)

    assert result.cost_usd is None
    assert result.local


def test_a_model_without_prices_has_no_cost():
    with mock.patch.object(litellm, "model_cost", {}):
        assert estimate.estimate_question(_paperqa_settings()).cost_usd is None


def test_label_is_higher_cost_from_the_cost_alone():
    assert estimate.label_for(_estimate(cost_usd=estimate.HIGHER_COST_USD, seconds=5)) == estimate.HIGHER_COST


def test_label_is_slower_from_the_time_alone():
    assert estimate.label_for(_estimate(cost_usd=0.001, seconds=estimate.SLOWER_SECONDS)) == estimate.SLOWER


def test_label_is_fast_when_quick_and_cheap():
    assert estimate.label_for(_estimate(cost_usd=0.001, seconds=10)) == estimate.FAST


def test_label_is_fast_from_the_time_when_the_cost_is_unknown():
    assert estimate.label_for(_estimate(cost_usd=None, seconds=10)) == estimate.FAST


def test_label_is_balanced_when_quick_but_not_cheap():
    assert estimate.label_for(_estimate(cost_usd=estimate.FAST_COST_USD, seconds=10)) == estimate.BALANCED


def test_label_is_balanced_when_cheap_but_not_quick():
    assert estimate.label_for(_estimate(cost_usd=0.001, seconds=estimate.FAST_SECONDS)) == estimate.BALANCED


@pytest.mark.parametrize(
    ("cost", "text"), [(0.004, "less than 1 cent"), (0.01, "up to about 1 cent"), (0.0231, "up to about 3 cents")]
)
def test_describe_cost_rounds_up_to_whole_cents(cost, text):
    assert estimate.describe_cost(cost) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(29, "under half a minute"), (30, "under a minute"), (59, "under a minute"), (61, "up to about 2 minutes")],
)
def test_describe_time_gives_a_rough_range(seconds, text):
    assert estimate.describe_time(seconds) == text


def test_summarize_gives_the_rough_cost_and_time():
    assert estimate.summarize(_estimate()) == "roughly less than 1 cent, under half a minute"


def test_summarize_says_when_the_cost_is_unknown():
    assert estimate.summarize(_estimate(cost_usd=None)) == "roughly cost not known, under half a minute"


def test_summarize_says_a_local_model_has_no_cost():
    summary = estimate.summarize(_estimate(cost_usd=None, local=True))

    assert summary == "roughly no cost for the answer model, under half a minute"


def test_describe_estimate_explains_what_is_counted():
    lines = estimate.describe_estimate(_estimate())

    assert lines == [
        "PaperQA reads 10 passages with the model, then writes an answer of up to about 200 words from at most 5 "
        "of them, plus a few short calls to plan the search.",
        "Cost: less than 1 cent.",
        "Time: under half a minute.",
        estimate.ROUGH_NOTE,
    ]


def test_describe_estimate_says_one_passage():
    assert estimate.describe_estimate(_estimate(passages=1, sources=1))[0].startswith("PaperQA reads 1 passage with")


def test_describe_estimate_says_when_the_cost_is_unknown():
    assert estimate.describe_estimate(_estimate(cost_usd=None))[1] == f"Cost: not known for {MODEL}."


def test_describe_estimate_explains_a_local_model():
    lines = estimate.describe_estimate(replace(_estimate(), cost_usd=None, local=True))

    assert lines[1] == estimate.LOCAL_COST_NOTE
    assert lines[2] == f"Time: under half a minute. {estimate.LOCAL_TIME_NOTE}"


@pytest.mark.parametrize("label", [estimate.FAST, estimate.BALANCED, estimate.SLOWER, estimate.HIGHER_COST])
def test_badge_colours_every_label(label):
    assert estimate.badge(label) == f":{estimate.BADGE_COLOURS[label]}-badge[{label}]"
