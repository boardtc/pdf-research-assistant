"""Named rate-limit profiles: how fast PaperQA may send requests to the answer model.

PaperQA holds back its own requests using limits in each model config: `rate_limit` (tokens a
minute) and `request_limit` (requests a minute), both keyed by model name. It also sends at most
`answer.max_concurrent_requests` requests at once. A profile sets all of these together, for the
answer, summary and agent models, so a choice fits the limits of the user's OpenAI account.

If a profile allows more than the account does, OpenAI turns requests away and PaperQA waits and
tries again, which is slower than a profile that fits. If it allows less, nothing breaks, but
long runs such as exhaustive search take far longer than they need to.

The limiter weighs a request by its length in characters / 4, and a picture goes as base64 text, so a
passage sent with whole-page pictures can weigh several hundred thousand tokens. lmi lets a request
wait only 60 seconds for its turn, after which it fails; allow_limiter_wait lets it wait longer.
"""

import functools
from dataclasses import dataclass

from lmi import rate_limiter


@dataclass(frozen=True)
class RateProfile:
    name: str
    tokens_per_minute: int
    requests_per_minute: int
    # How many requests an ordinary question sends at once.
    requests_at_once: int
    # How many requests exhaustive search sends at once; it makes one per passage, so it needs more.
    exhaustive_requests_at_once: int
    description: str


PROFILES = (
    RateProfile(
        name="Low limits",
        tokens_per_minute=30_000,
        requests_per_minute=500,
        requests_at_once=4,
        exhaustive_requests_at_once=4,
        description=(
            "For a new or low-tier OpenAI account. Ordinary questions are fine, "
            "but exhaustive search can take hours."
        ),
    ),
    RateProfile(
        name="Standard",
        tokens_per_minute=200_000,
        requests_per_minute=500,
        requests_at_once=5,
        exhaustive_requests_at_once=10,
        description=(
            "PaperQA's own figures for gpt-4o-mini on OpenAI's first paid tier. "
            "Exhaustive search is several times faster than with Low limits."
        ),
    ),
    RateProfile(
        name="High limits",
        tokens_per_minute=1_500_000,
        requests_per_minute=4_000,
        requests_at_once=10,
        exhaustive_requests_at_once=40,
        description=(
            "For an account allowed about 2,000,000 tokens and 5,000 requests a minute, with some room to spare. "
            "Fastest, but an account with lower limits will see requests turned away and retried."
        ),
    ),
)

# Exhaustive search used these limits before profiles existed, so keeping them changes nothing for it.
DEFAULT_PROFILE = "High limits"

# Shown when the limits in use match no profile, for example after a settings file was edited by hand.
NO_PROFILE = "Not a named profile"


def get_profile(name: str) -> RateProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise KeyError(name)


def _limit(amount: int) -> str:
    return f"{amount} per 1 minute"


def _model_configs(settings: object) -> tuple[tuple[object, str, str], ...]:
    """Return each (owner, model attribute, config attribute) that PaperQA uses to call a model."""
    return (
        (settings, "llm", "llm_config"),
        (settings, "summary_llm", "summary_llm_config"),
        (settings.agent, "agent_llm", "agent_llm_config"),
    )


def apply_profile(settings: object, name: str) -> None:
    """Put a profile's limits on a PaperQA Settings object, for each model it calls."""
    profile = get_profile(name)
    for owner, model_attribute, config_attribute in _model_configs(settings):
        model = getattr(owner, model_attribute)
        # A fresh dict each time, so the three configs never share one copy.
        config = dict(getattr(owner, config_attribute) or {})
        config["rate_limit"] = {model: _limit(profile.tokens_per_minute)}
        config["request_limit"] = {model: _limit(profile.requests_per_minute)}
        setattr(owner, config_attribute, config)
    settings.answer.max_concurrent_requests = profile.requests_at_once


def allow_limiter_wait(settings: object) -> None:
    """Let each request wait for the rate limiter for as long as the question may run (`agent.timeout`).

    lmi fixes its wait when it is imported (60 seconds, or the RATE_LIMITER_TIMEOUT environment variable),
    so the wait is given to its shared limiter here instead. Without this, passages sent with whole-page
    pictures queue behind each other for more than a minute, fail, and leave no evidence to answer from.
    """
    limiter = rate_limiter.GLOBAL_LIMITER
    # Built from the class's method each time, so a second call replaces the wait rather than wrapping it.
    limiter.try_acquire = functools.partial(
        type(limiter).try_acquire, limiter, acquire_timeout=settings.agent.timeout
    )


def find_profile(settings: object) -> RateProfile | None:
    """Return the profile whose limits a PaperQA Settings object has, or None if none matches."""
    config = settings.summary_llm_config or {}
    model = settings.summary_llm
    tokens = config.get("rate_limit", {}).get(model)
    requests = config.get("request_limit", {}).get(model)
    for profile in PROFILES:
        if (
            tokens == _limit(profile.tokens_per_minute)
            and requests == _limit(profile.requests_per_minute)
            and settings.answer.max_concurrent_requests == profile.requests_at_once
        ):
            return profile
    return None


def read_profile_name(settings: object) -> str:
    """Return the name of the profile in use, or NO_PROFILE."""
    profile = find_profile(settings)
    return profile.name if profile else NO_PROFILE


def exhaustive_requests_at_once(settings: object) -> int:
    """Return how many requests exhaustive search should send at once with these settings."""
    profile = find_profile(settings)
    return profile.exhaustive_requests_at_once if profile else settings.answer.max_concurrent_requests


def describe_profile(profile: RateProfile) -> str:
    """Return a profile's purpose and limits in plain words."""
    return (
        f"{profile.description} Up to {profile.tokens_per_minute:,} tokens and "
        f"{profile.requests_per_minute:,} requests a minute; {profile.requests_at_once} requests at once "
        f"for a question, {profile.exhaustive_requests_at_once} for exhaustive search."
    )
