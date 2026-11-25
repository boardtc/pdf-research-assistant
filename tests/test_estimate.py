from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import litellm
import pytest
from paperqa import Settings
from paperqa.settings import AgentSettings, MultimodalOptions

from pdf_research_assistant import estimate, rate_profiles

MODEL = "gpt-4o-mini"
# A real question on a one-PDF index with images and tables on, at the app's picture resolution.
MEASURED_MEDIA_COST_USD = 0.0547
MEASURED_MEDIA_SECONDS = 90
# bootstrap.PICTURE_DPI; bootstrap is not imported here, since it needs PAPER_DIR.
PICTURE_DPI = 150


def _paperqa_settings(
    model: str = MODEL,
    profile: str | None = rate_profiles.DEFAULT_PROFILE,
    media: bool | MultimodalOptions = False,
    whole_page: bool = False,
    dpi: int | None = PICTURE_DPI,
    **answer,
) -> Settings:
    """Real PaperQA settings, built the way bootstrap.build_settings builds them, with answer changes."""
    settings = Settings(llm=model, summary_llm=model, agent=AgentSettings(agent_llm=model))
    settings.parsing.multimodal = media
    settings.parsing.reader_config = {"chunk_chars": 5000, "overlap": 250, "full_page": whole_page}
    if dpi is not None:
        settings.parsing.reader_config["dpi"] = dpi
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


def test_fewer_passages_and_short_answers_in_fixed_steps_come_out_fast():
    settings = _paperqa_settings(evidence_k=5, answer_length="50 to 100 words")
    settings.agent.agent_type = estimate.FIXED_STEPS_AGENT_TYPE

    assert estimate.label_for(estimate.estimate_question(settings)) == estimate.FAST


def test_fewer_passages_and_short_answers_are_not_fast_while_the_model_plans_the_search():
    # The model's own planning calls and its repeated evidence gathering take most of a minute by themselves.
    settings = _paperqa_settings(evidence_k=5, answer_length="50 to 100 words")

    assert estimate.label_for(estimate.estimate_question(settings)) == estimate.BALANCED


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

    # Without a profile there is no token rate limit, so only the start-up and the calls set the time. The
    # model gathers evidence a few times, and each time the passages go out in two batches.
    assert estimate.estimate_question(settings).seconds == pytest.approx(
        estimate.START_SECONDS + estimate.EVIDENCE_ROUNDS * 2 * summary_seconds + rest
    )


def test_the_token_rate_limit_can_set_the_time():
    settings = _paperqa_settings(profile="Low limits")
    settings.summary_llm_config["rate_limit"][MODEL] = "5000 per 1 minute"
    result = estimate.estimate_question(settings)
    limited = (result.input_tokens * estimate.LIMITER_TOKEN_FACTOR + result.output_tokens) / 5000 * 60

    # Starting the question and opening the index come before any call the limit can hold back.
    assert result.seconds == pytest.approx(estimate.START_SECONDS + limited, rel=0.001)


def test_the_calls_set_the_time_when_the_token_rate_limit_is_not_reached():
    limited = _paperqa_settings()
    unlimited = _paperqa_settings()
    unlimited.summary_llm_config = None

    # High limits is far above what one question sends, so the time is the same as with no limit at all.
    assert estimate.estimate_question(limited).seconds == pytest.approx(estimate.estimate_question(unlimited).seconds)


def test_a_question_counts_the_time_to_start_and_open_the_index():
    with mock.patch.object(estimate, "START_SECONDS", 0):
        without_start = estimate.estimate_question(_paperqa_settings()).seconds

    assert estimate.estimate_question(_paperqa_settings()).seconds == pytest.approx(
        without_start + estimate.START_SECONDS
    )


def _preset_settings(preset: str) -> Settings:
    from pdf_research_assistant import quality_presets

    settings = _paperqa_settings()
    quality_presets.apply_preset(settings, preset)
    return settings


def _pre_prompt_settings() -> Settings:
    settings = _paperqa_settings()
    settings.prompts.pre = "List the scientific terms in this question: {question}"
    return settings


# Real questions on a library of 119 PDFs with High limits (24 and 25 Nov 2026), timed from asking to the
# answer appearing. Where a setting was timed more than once, this is a typical time.
MEASURED_QUESTIONS = [
    ("Balanced", lambda: _paperqa_settings(), 80),
    ("Fast", lambda: _preset_settings("Fast"), 35),
    ("Debug", lambda: _preset_settings("Debug"), 60),
    ("High quality", lambda: _preset_settings("High quality"), 55),
    ("Long answer", lambda: _paperqa_settings(answer_length="about 400 words, but can be longer"), 74),
    ("30 passages", lambda: _paperqa_settings(evidence_k=30, answer_max_sources=15), 68),
    ("Before-answer prompt", _pre_prompt_settings, 85),
    ("Low limits", lambda: _paperqa_settings(profile="Low limits"), 93),
    ("Cut-out pictures", lambda: _paperqa_settings(media=True), 204),
    ("Whole-page pictures", lambda: _paperqa_settings(media=True, whole_page=True), 198),
]


@pytest.mark.parametrize(("name", "settings", "seconds"), MEASURED_QUESTIONS, ids=[q[0] for q in MEASURED_QUESTIONS])
def test_the_time_is_close_to_measured_questions(name, settings, seconds):
    result = estimate.estimate_question(settings())

    # One question can run a minute longer or shorter than the next, depending on how many searches the
    # model makes, so close means within about half again either way.
    assert seconds * 0.75 <= result.seconds <= seconds * 1.6


@pytest.mark.parametrize(("name", "settings", "seconds"), MEASURED_QUESTIONS, ids=[q[0] for q in MEASURED_QUESTIONS])
def test_no_measured_question_is_shown_as_under_half_a_minute(name, settings, seconds):
    assert estimate.describe_time(estimate.estimate_question(settings()).seconds) != "under half a minute"


# Pictures, sent with each passage when images and tables are on.


@pytest.mark.parametrize(
    ("width", "height", "tiles"),
    [
        # Small enough to send as it is.
        (504, 396, 1),
        # A short side under 768 pixels is not enlarged: 3 tiles across, 2 down.
        (1040, 616, 6),
        # A whole A4 page at 150 dpi shrinks to 768 x 1083: 2 across, 3 down.
        (1245, 1755, 6),
        # Too wide: first fitted within 2048 pixels, to 2048 x 683.
        (3000, 1000, 8),
    ],
)
def test_count_tiles_follows_openai_resizing(width, height, tiles):
    assert estimate.count_tiles(width, height) == tiles


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o-mini-2024-07-18", "openai/gpt-4o-mini"])
def test_get_picture_tokens_knows_gpt_4o_mini_bills_pictures_higher(model):
    assert estimate.get_picture_tokens(model) == estimate.PICTURE_TOKENS["gpt-4o-mini"]


def test_get_picture_tokens_uses_the_usual_figures_for_other_models():
    assert estimate.get_picture_tokens("gpt-4o") == estimate.DEFAULT_PICTURE_TOKENS


def test_count_picture_tokens_adds_a_base_and_each_tile():
    base, tile = estimate.PICTURE_TOKENS["gpt-4o-mini"]

    assert estimate.count_picture_tokens("gpt-4o-mini", 1040, 616) == base + 6 * tile


def test_count_pages_includes_the_page_a_passage_runs_into():
    assert estimate.count_pages(_paperqa_settings()) == 5000 / estimate.CHARACTERS_PER_PAGE + 1


def test_count_pictures_is_zero_with_images_and_tables_off():
    assert estimate.count_pictures(_paperqa_settings(media=False)) == 0


@pytest.mark.parametrize("media", [True, MultimodalOptions.ON_WITH_ENRICHMENT, MultimodalOptions.ON_WITHOUT_ENRICHMENT])
def test_count_pictures_guesses_from_a_typical_page_when_on(media):
    settings = _paperqa_settings(media=media)

    assert estimate.count_pictures(settings) == estimate.PICTURES_PER_PAGE * estimate.count_pages(settings)


def test_count_pictures_is_one_for_each_page_with_whole_page_pictures():
    settings = _paperqa_settings(media=True, whole_page=True)

    assert estimate.count_pictures(settings) == estimate.count_pages(settings)


def test_picture_pixels_for_a_cut_out_picture_follow_the_dpi():
    assert estimate.picture_pixels(_paperqa_settings(dpi=150)) == (1050, 825)


def test_picture_pixels_for_a_whole_page_are_an_a4_page():
    assert estimate.picture_pixels(_paperqa_settings(whole_page=True, dpi=150)) == (1245, 1755)


def test_picture_pixels_use_paperqa_resolution_without_a_dpi():
    assert estimate.picture_pixels(_paperqa_settings(dpi=None)) == (504, 396)


def test_count_limiter_tokens_counts_a_picture_as_base64_text():
    chars = 1000 * 500 * estimate.PNG_BYTES_PER_PIXEL * estimate.BASE64_CHARACTERS_PER_BYTE

    assert estimate.count_limiter_tokens(1000, 500) == pytest.approx(chars / estimate.CHARACTERS_PER_TOKEN)


def test_a_question_on_a_media_index_covers_the_measured_cost_and_time():
    result = estimate.estimate_question(_paperqa_settings(media=True))

    assert result.cost_usd >= MEASURED_MEDIA_COST_USD
    assert result.seconds >= MEASURED_MEDIA_SECONDS
    assert estimate.label_for(result) == estimate.HIGHER_COST


def test_images_and_tables_off_sends_no_pictures():
    result = estimate.estimate_question(_paperqa_settings(media=False))

    assert result.pictures == 0
    assert not result.whole_page


def test_pictures_are_counted_for_the_whole_question():
    settings = _paperqa_settings(media=True, whole_page=True)
    result = estimate.estimate_question(settings)

    assert result.pictures == round(10 * estimate.count_pictures(settings))
    assert result.whole_page


def test_whole_page_pictures_cost_more_than_cut_out_ones():
    cut_out = estimate.estimate_question(_paperqa_settings(media=True))
    whole_page = estimate.estimate_question(_paperqa_settings(media=True, whole_page=True))

    assert whole_page.cost_usd > cut_out.cost_usd
    assert whole_page.seconds > cut_out.seconds


def test_sharper_pictures_cost_more():
    sharp = estimate.estimate_question(_paperqa_settings(media=True, dpi=150))
    plain = estimate.estimate_question(_paperqa_settings(media=True, dpi=None))

    assert sharp.cost_usd > plain.cost_usd


def test_pictures_are_billed_at_the_summary_model_rate():
    mini = _paperqa_settings(media=True)
    other = _paperqa_settings(media=True)
    other.summary_llm = "gpt-4o"
    pictures = 10 * estimate.count_pictures(mini)
    width, height = estimate.picture_pixels(mini)
    difference = pictures * (
        estimate.count_picture_tokens("gpt-4o-mini", width, height)
        - estimate.count_picture_tokens("gpt-4o", width, height)
    )

    assert estimate.estimate_question(mini).input_tokens - estimate.estimate_question(other).input_tokens == (
        pytest.approx(difference, abs=1)
    )


def test_the_token_rate_limit_counts_pictures_as_text():
    settings = _paperqa_settings(media=True, profile="Low limits")
    pictures = 10 * estimate.count_pictures(settings)

    limited = pictures * estimate.count_limiter_tokens(*estimate.picture_pixels(settings)) / 30000 * 60

    assert estimate.estimate_question(settings).seconds > limited


def test_pictures_slow_a_local_model_too():
    text_only = estimate.estimate_question(_paperqa_settings(), local=True)
    with_pictures = estimate.estimate_question(_paperqa_settings(media=True), local=True)

    assert with_pictures.cost_usd is None
    assert with_pictures.seconds > text_only.seconds


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
        f"of them, plus about {estimate.AGENT_CALLS} short calls to plan the search.",
        "Cost: less than 1 cent.",
        f"Time: under half a minute. {estimate.START_NOTE} {estimate.EVIDENCE_ROUNDS_NOTE}",
        estimate.ROUGH_NOTE,
    ]


def test_start_note_and_evidence_rounds_note_give_the_figures_used():
    assert f"about {estimate.START_SECONDS} seconds" in estimate.START_NOTE
    assert f"about {estimate.EVIDENCE_ROUNDS} times" in estimate.EVIDENCE_ROUNDS_NOTE


def test_describe_estimate_explains_cut_out_pictures():
    lines = estimate.describe_estimate(_estimate(pictures=19))

    assert lines[1] == (
        "Images and tables are on, so passages are sent with their pictures: about 19 pictures in all, guessed "
        f"from a typical page. {estimate.PICTURE_NOTE}"
    )
    assert lines[2] == "Cost: less than 1 cent."


def test_describe_estimate_explains_whole_page_pictures():
    lines = estimate.describe_estimate(_estimate(pictures=27, whole_page=True))

    assert lines[1] == (
        "Images and tables are on with whole-page pictures, so passages are sent with a picture of each of "
        f"their pages: about 27 pictures in all. {estimate.PICTURE_NOTE}"
    )


def test_a_before_answer_prompt_adds_one_answer_model_call():
    usual = estimate.estimate_question(_paperqa_settings())
    settings = _paperqa_settings()
    settings.prompts.pre = "List the scientific terms in this question: {question}"

    result = estimate.estimate_question(settings)

    assert not usual.pre_prompt
    assert result.pre_prompt
    assert result.cost_usd > usual.cost_usd
    # The extra call runs before the answer, so it adds its own time.
    assert result.seconds == pytest.approx(usual.seconds + estimate.call_seconds(estimate.PRE_OUTPUT_TOKENS))
    assert result.output_tokens == usual.output_tokens + estimate.PRE_OUTPUT_TOKENS


def test_describe_estimate_explains_a_before_answer_prompt():
    lines = estimate.describe_estimate(_estimate(pre_prompt=True))

    assert lines[1] == estimate.PRE_PROMPT_NOTE
    assert lines[2] == "Cost: less than 1 cent."


def test_describe_estimate_says_one_passage():
    assert estimate.describe_estimate(_estimate(passages=1, sources=1))[0].startswith("PaperQA reads 1 passage with")


def test_describe_estimate_says_when_the_cost_is_unknown():
    assert estimate.describe_estimate(_estimate(cost_usd=None))[1] == f"Cost: not known for {MODEL}."


def test_describe_estimate_explains_a_local_model():
    lines = estimate.describe_estimate(replace(_estimate(), cost_usd=None, local=True))

    assert lines[1] == estimate.LOCAL_COST_NOTE
    assert lines[2] == (
        f"Time: under half a minute. {estimate.START_NOTE} {estimate.EVIDENCE_ROUNDS_NOTE} {estimate.LOCAL_TIME_NOTE}"
    )


@pytest.mark.parametrize("label", [estimate.FAST, estimate.BALANCED, estimate.SLOWER, estimate.HIGHER_COST])
def test_badge_colours_every_label(label):
    assert estimate.badge(label) == f":{estimate.BADGE_COLOURS[label]}-badge[{label}]"


def test_fixed_search_steps_make_two_short_calls_instead_of_the_agent_s_planning_calls():
    usual = estimate.estimate_question(_paperqa_settings())
    settings = _paperqa_settings()
    settings.agent.agent_type = "fake"

    result = estimate.estimate_question(settings)

    assert not usual.fixed_steps
    assert result.fixed_steps
    assert result.cost_usd < usual.cost_usd
    saved_calls = estimate.AGENT_CALLS - estimate.FIXED_STEPS_CALLS
    # Fixed steps also gather evidence only once, so the passages are read once instead of a few times.
    summary_seconds = estimate.call_seconds(estimate.count_words("about 100 words") * estimate.TOKENS_PER_WORD)
    saved_reads = (estimate.EVIDENCE_ROUNDS - 1) * summary_seconds
    assert result.seconds == pytest.approx(
        usual.seconds - saved_calls * estimate.call_seconds(estimate.AGENT_OUTPUT_TOKENS) - saved_reads
    )
    assert result.input_tokens == usual.input_tokens - (
        estimate.AGENT_CALLS * estimate.AGENT_INPUT_TOKENS - estimate.FIXED_STEPS_INPUT_TOKENS
    )


def test_fixed_search_steps_are_billed_at_the_answer_model_rate():
    settings = _paperqa_settings()
    settings.agent.agent_type = "fake"
    settings.agent.agent_llm = "some-model-without-prices"

    # PaperQA asks the answer model for the searches, so an agent model without prices does not matter.
    assert estimate.estimate_question(settings).cost_usd is not None


def test_describe_estimate_explains_fixed_search_steps():
    lines = estimate.describe_estimate(_estimate(fixed_steps=True))

    assert lines[0].endswith("plus two short calls to suggest searches and finish.")
    # Fixed steps gather evidence once, so only the start-up is mentioned.
    assert lines[-2] == f"Time: under half a minute. {estimate.START_NOTE}"


@pytest.mark.parametrize(
    ("preset", "label"),
    [
        ("Fast", estimate.FAST),
        ("Balanced", estimate.BALANCED),
        # The smallest run, but the model still plans its searches, which takes about a minute.
        ("Debug", estimate.BALANCED),
    ],
)
def test_quality_presets_come_out_with_the_label_their_name_suggests(preset, label):
    from pdf_research_assistant import quality_presets

    settings = _paperqa_settings()
    quality_presets.apply_preset(settings, preset)

    assert estimate.label_for(estimate.estimate_question(settings)) == label


def test_the_high_quality_preset_costs_more_and_takes_longer_than_balanced():
    from pdf_research_assistant import quality_presets

    balanced, high = _paperqa_settings(), _paperqa_settings()
    quality_presets.apply_preset(high, "High quality")

    usual, result = estimate.estimate_question(balanced), estimate.estimate_question(high)
    assert result.cost_usd > usual.cost_usd
    assert result.seconds > usual.seconds
    assert estimate.label_for(result) != estimate.FAST
