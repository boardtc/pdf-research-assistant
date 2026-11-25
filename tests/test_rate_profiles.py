import pytest
from paperqa import Settings
from paperqa.settings import AgentSettings

from pdf_research_assistant import rate_profiles, settings_groups


def _settings(model: str = "gpt-4o-mini", config: dict | None = None) -> Settings:
    config = config if config is not None else {}
    return Settings(
        llm=model,
        summary_llm=model,
        llm_config=dict(config),
        summary_llm_config=dict(config),
        agent=AgentSettings(agent_llm=model, agent_llm_config=dict(config)),
    )


def _configs(settings: Settings) -> list[dict]:
    return [settings.llm_config, settings.summary_llm_config, settings.agent.agent_llm_config]


def test_profiles_have_unique_names_and_the_default_is_one_of_them():
    names = [profile.name for profile in rate_profiles.PROFILES]

    assert len(names) == len(set(names))
    assert rate_profiles.DEFAULT_PROFILE in names


def test_profiles_get_faster_in_order():
    profiles = rate_profiles.PROFILES

    for slower, faster in zip(profiles, profiles[1:]):
        assert slower.tokens_per_minute < faster.tokens_per_minute
        assert slower.requests_at_once <= faster.requests_at_once
        assert slower.exhaustive_requests_at_once <= faster.exhaustive_requests_at_once


def test_the_fastest_profile_stays_under_the_measured_account_limits():
    fastest = rate_profiles.PROFILES[-1]

    # Measured for gpt-4o-mini on 2026-11-06: 2,000,000 tokens and 5,000 requests a minute.
    assert fastest.tokens_per_minute < 2_000_000
    assert fastest.requests_per_minute < 5_000


def test_get_profile_returns_the_named_profile():
    assert rate_profiles.get_profile("Standard").name == "Standard"


def test_get_profile_raises_for_an_unknown_name():
    with pytest.raises(KeyError):
        rate_profiles.get_profile("Turbo")


def test_apply_profile_sets_token_and_request_limits_for_each_model_role():
    settings = _settings()

    rate_profiles.apply_profile(settings, "Standard")

    profile = rate_profiles.get_profile("Standard")
    for config in _configs(settings):
        assert config["rate_limit"] == {"gpt-4o-mini": f"{profile.tokens_per_minute} per 1 minute"}
        assert config["request_limit"] == {"gpt-4o-mini": f"{profile.requests_per_minute} per 1 minute"}
    assert settings.answer.max_concurrent_requests == profile.requests_at_once


def test_apply_profile_keeps_each_role_a_separate_copy():
    settings = _settings()

    rate_profiles.apply_profile(settings, "Standard")

    assert len({id(config) for config in _configs(settings)}) == 3


def test_apply_profile_keeps_the_rest_of_a_local_model_config():
    model_list = [{"model_name": "ollama/llama3.2", "litellm_params": {"model": "ollama/llama3.2"}}]
    settings = _settings("ollama/llama3.2", {"model_list": model_list})

    rate_profiles.apply_profile(settings, "Low limits")

    for config in _configs(settings):
        assert config["model_list"] == model_list
        assert set(config["rate_limit"]) == {"ollama/llama3.2"}


def test_apply_profile_replaces_limits_for_other_models():
    settings = _settings(config={"rate_limit": {"gpt-4o": "1 per 1 minute"}})

    rate_profiles.apply_profile(settings, "Standard")

    assert set(settings.summary_llm_config["rate_limit"]) == {"gpt-4o-mini"}


def test_apply_profile_handles_a_missing_config():
    settings = _settings()
    settings.summary_llm_config = None

    rate_profiles.apply_profile(settings, "Standard")

    assert "gpt-4o-mini" in settings.summary_llm_config["rate_limit"]


def test_paperqa_uses_the_profile_limits_for_its_models():
    settings = _settings()

    rate_profiles.apply_profile(settings, "High limits")

    profile = rate_profiles.get_profile("High limits")
    model = settings.get_summary_llm()
    assert model.config["rate_limit"]["gpt-4o-mini"] == f"{profile.tokens_per_minute} per 1 minute"
    assert model.config["request_limit"]["gpt-4o-mini"] == f"{profile.requests_per_minute} per 1 minute"


@pytest.mark.parametrize("name", [profile.name for profile in rate_profiles.PROFILES])
def test_find_profile_names_the_profile_that_was_applied(name):
    settings = _settings()
    rate_profiles.apply_profile(settings, name)

    assert rate_profiles.find_profile(settings).name == name


def test_find_profile_returns_none_when_the_limits_match_no_profile():
    settings = _settings(config={"rate_limit": {"gpt-4o-mini": "123 per 1 minute"}})

    assert rate_profiles.find_profile(settings) is None


def test_find_profile_returns_none_when_requests_at_once_were_changed():
    settings = _settings()
    rate_profiles.apply_profile(settings, "Standard")
    settings.answer.max_concurrent_requests = 99

    assert rate_profiles.find_profile(settings) is None


def test_find_profile_returns_none_without_a_config():
    settings = _settings()
    settings.summary_llm_config = None

    assert rate_profiles.find_profile(settings) is None


def test_read_profile_name_names_an_applied_profile():
    settings = _settings()
    rate_profiles.apply_profile(settings, "Standard")

    assert rate_profiles.read_profile_name(settings) == "Standard"


def test_read_profile_name_says_when_no_profile_matches():
    assert rate_profiles.read_profile_name(_settings()) == rate_profiles.NO_PROFILE


def test_exhaustive_requests_at_once_follows_the_profile():
    settings = _settings()
    rate_profiles.apply_profile(settings, "Low limits")

    expected = rate_profiles.get_profile("Low limits").exhaustive_requests_at_once
    assert rate_profiles.exhaustive_requests_at_once(settings) == expected


def test_exhaustive_requests_at_once_keeps_the_setting_when_no_profile_matches():
    settings = _settings()
    settings.answer.max_concurrent_requests = 7

    assert rate_profiles.exhaustive_requests_at_once(settings) == 7


def test_describe_profile_mentions_its_limits():
    profile = rate_profiles.get_profile("Standard")

    text = rate_profiles.describe_profile(profile)

    assert profile.description in text
    assert f"{profile.tokens_per_minute:,}" in text
    assert str(profile.requests_at_once) in text


def test_the_profile_setting_is_saved_and_shown_per_answer():
    setting = settings_groups.get_setting("rate_profile")

    assert setting in settings_groups.INSTANT_SETTINGS
    assert setting.needs_reindex is False
    assert setting.check.values == tuple(profile.name for profile in rate_profiles.PROFILES)


def test_the_profile_setting_writes_and_reads_a_profile():
    setting = settings_groups.get_setting("rate_profile")
    settings = _settings()

    setting.write(settings, "Low limits")

    assert setting.read(settings) == "Low limits"
    assert settings.answer.max_concurrent_requests == rate_profiles.get_profile("Low limits").requests_at_once


def test_the_profile_setting_rejects_an_unknown_profile():
    with pytest.raises(ValueError):
        settings_groups.get_setting("rate_profile").check("Turbo")


def test_a_recorded_answer_names_the_profile():
    settings = _settings()
    rate_profiles.apply_profile(settings, "Standard")

    record = settings_groups.record_answer_settings(settings)

    values = {item["key"]: item["value"] for item in record["values"]}
    assert values["rate_profile"] == "Standard"


@pytest.fixture
def limiter():
    """PaperQA's shared rate limiter, put back to lmi's own wait after the test."""
    from lmi.rate_limiter import GLOBAL_LIMITER

    yield GLOBAL_LIMITER
    vars(GLOBAL_LIMITER).pop("try_acquire", None)


def _acquire_twice(limiter, key: str) -> None:
    """Ask the limiter for its whole allowance twice, so the second request must wait about a second."""
    import asyncio

    async def run():
        for _ in range(2):
            await limiter.try_acquire(("test", key), "10 per 1 second", weight=10)

    asyncio.run(run())


def test_requests_wait_for_the_rate_limiter_as_long_as_the_question_may_run(limiter):
    settings = _settings()
    settings.agent.timeout = 5

    rate_profiles.allow_limiter_wait(settings)

    _acquire_twice(limiter, "long-wait")


def test_a_request_still_stops_waiting_when_the_question_would_have_ended(limiter):
    settings = _settings()
    settings.agent.timeout = 5
    rate_profiles.allow_limiter_wait(settings)
    # A later question with a shorter time allowed replaces the wait instead of keeping the longer one.
    settings.agent.timeout = 0.2

    rate_profiles.allow_limiter_wait(settings)

    with pytest.raises(TimeoutError):
        _acquire_twice(limiter, "short-wait")


def test_the_rate_limiter_waits_only_a_minute_without_this(limiter):
    from lmi.rate_limiter import GLOBAL_RATE_LIMITER_TIMEOUT

    # lmi's own wait, which page pictures counted as base64 text can far outlast (#96).
    assert GLOBAL_RATE_LIMITER_TIMEOUT == 60
