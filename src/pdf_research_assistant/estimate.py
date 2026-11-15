"""Rough cost and speed estimates, worked out from PaperQA settings before anything is sent.

Exhaustive search plans one summary call per passage in scope (exhaustive.estimate_run). An ordinary
question is smaller and its size is known from the settings alone, so it can be estimated on every
page load without reading the index:

- PaperQA's agent makes a few calls to choose its tools (search, gather evidence, answer).
- Gathering evidence sends `answer.evidence_k` passages to the summary model, at most
  `answer.max_concurrent_requests` at a time, and asks for `answer.evidence_summary_length` each.
- The answer model then writes `answer.answer_length` from up to `answer.answer_max_sources` summaries.
- With images and tables on (`parsing.multimodal`), each passage goes to the summary model with the
  pictures from its pages, cut out or of the whole page (`reader_config["full_page"]`), drawn at
  `reader_config["dpi"]`. How many a passage has is a guess, since counting them means reading the index.

The figures are deliberately rough. They become a label and plain-word ranges, never exact amounts.
"""

import math
import re
from dataclasses import dataclass

import limits
import litellm
from paperqa.settings import MultimodalOptions

# Median measured for gpt-4o-mini writing 200-word summaries (about 160 output tokens).
SECONDS_PER_CALL = 5.5
TYPICAL_OUTPUT_TOKENS = 160
# Part of each call's time is spent before the first word comes back; the rest grows with the reply.
CALL_START_SECONDS = 1.5
SECONDS_PER_OUTPUT_TOKEN = (SECONDS_PER_CALL - CALL_START_SECONDS) / TYPICAL_OUTPUT_TOKENS
CHARACTERS_PER_TOKEN = 4
TOKENS_PER_WORD = 4 / 3
# The rate limiter counts characters / 4, which runs about 10% above real input tokens.
LIMITER_TOKEN_FACTOR = 1.1
# Used when a length phrase names no number of words.
FALLBACK_WORDS = 200
# PaperQA's agent usually searches, gathers evidence, answers and finishes: about four short calls, each
# sent its tool descriptions and the conversation so far. A guess, not a measurement.
AGENT_CALLS = 4
AGENT_INPUT_TOKENS = 2000
AGENT_OUTPUT_TOKENS = 50
# Each passage is sent with its citation, such as "Smith 2020 pages 3-4: Smith, J. (2020) ...".
CITATION_TOKENS = 30

# Pictures and tables found per page, measured across a library of about 120 PDFs (2,127 over 3,163 pages).
PICTURES_PER_PAGE = 0.7
# A typical article page holds about 3,000 characters, so a passage usually runs on into one more page,
# and brings that page's pictures with it.
CHARACTERS_PER_PAGE = 3000
# Most cut-out pictures measured were figures across the width of the text; whole-page pictures are A4.
PICTURE_INCHES = (7, 5.5)
PAGE_INCHES = (8.3, 11.7)
# Without a dpi in reader_config, PaperQA draws pictures at 72 dpi.
DEFAULT_DPI = 72
# OpenAI fits a picture within 2048 x 2048 pixels, shrinks it until its short side is at most 768, then
# bills a base amount plus an amount for each 512-pixel tile.
MAX_PICTURE_SIDE = 2048
MAX_SHORT_SIDE = 768
TILE_SIDE = 512
# Tokens billed per picture and per tile. gpt-4o-mini bills about 33 times as many tokens as gpt-4o for the
# same picture, so at its lower price a picture costs about the same: often more than the passage's text.
PICTURE_TOKENS = {"gpt-4o-mini": (2833, 5667)}
DEFAULT_PICTURE_TOKENS = (85, 170)
# The rate limiter counts every character of a request, and a picture goes as base64 text: 4 characters
# for every 3 bytes of a PNG, which measured about 0.165 bytes per pixel.
PNG_BYTES_PER_PIXEL = 0.165
BASE64_CHARACTERS_PER_BYTE = 4 / 3
# Fitted to one measured question: 10 passages sent at once with 14 pictures, up to 5 in one passage, took
# about 90 seconds against an estimate of under 30 without pictures. A batch waits for its slowest passage,
# which has more pictures than the average, so this is well above the time to read one picture.
SECONDS_PER_PICTURE = 40

FAST = "Fast"
BALANCED = "Balanced"
SLOWER = "Slower"
HIGHER_COST = "Higher cost"
# Chosen so PaperQA's defaults with gpt-4o-mini come out Balanced, its fast preset's numbers (5 passages,
# short answers) Fast, and 30 passages Higher cost.
FAST_SECONDS = 22
FAST_COST_USD = 0.004
SLOWER_SECONDS = 45
HIGHER_COST_USD = 0.008

BADGE_COLOURS = {FAST: "green", BALANCED: "blue", SLOWER: "orange", HIGHER_COST: "orange"}
ROUGH_NOTE = (
    "A rough guide, not a quote: the real figures depend on the question, on how much the model writes, "
    "and on how many searches PaperQA decides to make."
)
LOCAL_COST_NOTE = (
    "Cost: none for the answer model, which runs on your computer. Finding passages still uses OpenAI "
    "embeddings, which cost very little."
)
LOCAL_TIME_NOTE = "The time is based on OpenAI's speed; a local model may be much slower."
PICTURE_NOTE = "Pictures often cost more than the text and make each passage slower to read."


@dataclass(frozen=True)
class Estimate:
    passages: int
    sources: int
    answer_words: int
    input_tokens: int
    output_tokens: int
    # None when the price is not known, such as for a local model.
    cost_usd: float | None
    seconds: float
    model: str
    local: bool = False
    # Pictures sent with all the passages together, and whether each is a whole page.
    pictures: int = 0
    whole_page: bool = False


def get_tokens_per_minute(settings: object) -> float | None:
    """Return the summary model's configured token rate limit, if it has one."""
    rate_limits = (settings.summary_llm_config or {}).get("rate_limit", {})
    rate_limit = rate_limits.get(settings.summary_llm)
    if rate_limit is None:
        return None
    parsed = limits.parse(rate_limit)
    return parsed.amount * 60 / parsed.get_expiry()


def get_prices(model: str) -> dict | None:
    """Return LiteLLM's per-token prices for a model, or None if it has none."""
    prices = litellm.model_cost.get(model)
    if not prices or "input_cost_per_token" not in prices or "output_cost_per_token" not in prices:
        return None
    return prices


def count_words(phrase: str) -> int:
    """Return the largest number of words a length phrase asks for, such as 100 in "50 to 100 words"."""
    numbers = [int(number) for number in re.findall(r"\d+", phrase)]
    return max(numbers) if numbers else FALLBACK_WORDS


def count_prompt_tokens(*templates: str) -> int:
    return sum(len(template) for template in templates) // CHARACTERS_PER_TOKEN


def call_seconds(output_tokens: float) -> float:
    """Return roughly how long one model call takes to write this many tokens."""
    return CALL_START_SECONDS + output_tokens * SECONDS_PER_OUTPUT_TOKEN


def call_cost(model: str, input_tokens: float, output_tokens: float) -> float | None:
    prices = get_prices(model)
    if prices is None:
        return None
    return input_tokens * prices["input_cost_per_token"] + output_tokens * prices["output_cost_per_token"]


def count_tiles(width: float, height: float) -> int:
    """Return how many 512-pixel tiles OpenAI bills for a picture, after its own resizing."""
    scale = min(1, MAX_PICTURE_SIDE / max(width, height))
    width, height = width * scale, height * scale
    scale = min(1, MAX_SHORT_SIDE / min(width, height))
    width, height = width * scale, height * scale
    return math.ceil(width / TILE_SIDE) * math.ceil(height / TILE_SIDE)


def get_picture_tokens(model: str) -> tuple[int, int]:
    """Return the tokens a model bills per picture and per tile, ignoring any provider prefix and date."""
    name = model.rsplit("/", 1)[-1]
    for known, tokens in PICTURE_TOKENS.items():
        if name.startswith(known):
            return tokens
    return DEFAULT_PICTURE_TOKENS


def count_picture_tokens(model: str, width: float, height: float) -> int:
    base, tile = get_picture_tokens(model)
    return base + count_tiles(width, height) * tile


def count_pages(settings: object) -> float:
    """Return how many pages a passage usually touches: its own text plus the page it runs into."""
    return settings.parsing.reader_config["chunk_chars"] / CHARACTERS_PER_PAGE + 1


def count_pictures(settings: object) -> float:
    """Return roughly how many pictures go with each passage, or 0 with images and tables off."""
    parse_media, _ = MultimodalOptions.from_value(settings.parsing.multimodal).should_parse_and_enrich_media
    if not parse_media:
        return 0
    if settings.parsing.reader_config.get("full_page", False):
        return count_pages(settings)
    return PICTURES_PER_PAGE * count_pages(settings)


def picture_pixels(settings: object) -> tuple[int, int]:
    """Return a typical picture's size in pixels at the index's picture resolution."""
    dpi = settings.parsing.reader_config.get("dpi") or DEFAULT_DPI
    inches = PAGE_INCHES if settings.parsing.reader_config.get("full_page", False) else PICTURE_INCHES
    return round(inches[0] * dpi), round(inches[1] * dpi)


def count_limiter_tokens(width: float, height: float) -> float:
    """Return how many tokens the rate limiter counts for a picture, which it sees as base64 text."""
    return width * height * PNG_BYTES_PER_PIXEL * BASE64_CHARACTERS_PER_BYTE / CHARACTERS_PER_TOKEN


def estimate_question(settings: object, local: bool = False) -> Estimate:
    """Estimate one ordinary question with these PaperQA settings. Makes no API calls."""
    answer = settings.answer
    prompts = settings.prompts
    passages = answer.evidence_k
    sources = min(answer.answer_max_sources, passages)
    answer_words = count_words(answer.answer_length)

    passage_tokens = settings.parsing.reader_config["chunk_chars"] // CHARACTERS_PER_TOKEN
    summary_input = (
        passage_tokens + CITATION_TOKENS + count_prompt_tokens(prompts.summary_json_system, prompts.summary_json)
    )
    summary_output = count_words(answer.evidence_summary_length) * TOKENS_PER_WORD
    answer_input = sources * summary_output + count_prompt_tokens(prompts.qa, prompts.system)
    answer_output = answer_words * TOKENS_PER_WORD

    pictures_per_passage = count_pictures(settings)
    pictures = passages * pictures_per_passage
    width, height = picture_pixels(settings)
    picture_tokens = pictures * count_picture_tokens(settings.summary_llm, width, height)

    calls = (
        (settings.summary_llm, passages * summary_input + picture_tokens, passages * summary_output),
        (settings.llm, answer_input, answer_output),
        (settings.agent.agent_llm, AGENT_CALLS * AGENT_INPUT_TOKENS, AGENT_CALLS * AGENT_OUTPUT_TOKENS),
    )
    input_tokens = sum(call[1] for call in calls)
    output_tokens = sum(call[2] for call in calls)
    costs = [call_cost(*call) for call in calls]
    cost_usd = None if local or None in costs else sum(costs)

    # Summaries go out a batch at a time; the answer and the agent's calls go one after another.
    batches = math.ceil(passages / answer.max_concurrent_requests)
    seconds = (
        batches * (call_seconds(summary_output) + pictures_per_passage * SECONDS_PER_PICTURE)
        + call_seconds(answer_output)
        + AGENT_CALLS * call_seconds(AGENT_OUTPUT_TOKENS)
    )
    # Unless the token rate limit holds the question back for longer. It counts pictures by their size as
    # base64 text, not by the tokens they are billed at.
    tokens_per_minute = get_tokens_per_minute(settings)
    if tokens_per_minute:
        limiter_tokens = (input_tokens - picture_tokens) * LIMITER_TOKEN_FACTOR + pictures * count_limiter_tokens(
            width, height
        )
        seconds = max(seconds, (limiter_tokens + output_tokens) / tokens_per_minute * 60)

    return Estimate(
        passages=passages,
        sources=sources,
        answer_words=answer_words,
        input_tokens=round(input_tokens),
        output_tokens=round(output_tokens),
        cost_usd=cost_usd,
        seconds=seconds,
        model=settings.llm,
        local=local,
        pictures=round(pictures),
        whole_page=bool(pictures) and settings.parsing.reader_config.get("full_page", False),
    )


def label_for(estimate: Estimate) -> str:
    """Return a one-word sense of the question's size. Without a known cost, only the time counts."""
    cost = estimate.cost_usd
    if cost is not None and cost >= HIGHER_COST_USD:
        return HIGHER_COST
    if estimate.seconds >= SLOWER_SECONDS:
        return SLOWER
    if estimate.seconds < FAST_SECONDS and (cost is None or cost < FAST_COST_USD):
        return FAST
    return BALANCED


def describe_cost(cost_usd: float) -> str:
    """Return a cost as a rough amount in cents."""
    if cost_usd < 0.01:
        return "less than 1 cent"
    cents = math.ceil(cost_usd * 100)
    return f"up to about {cents} cent{'' if cents == 1 else 's'}"


def describe_time(seconds: float) -> str:
    """Return a time as a rough range."""
    if seconds < 30:
        return "under half a minute"
    if seconds < 60:
        return "under a minute"
    minutes = math.ceil(seconds / 60)
    return f"up to about {minutes} minutes"


def summarize(estimate: Estimate) -> str:
    """Return the estimate's rough cost and time in a few words."""
    if estimate.local:
        cost = "no cost for the answer model"
    elif estimate.cost_usd is None:
        cost = "cost not known"
    else:
        cost = describe_cost(estimate.cost_usd)
    return f"roughly {cost}, {describe_time(estimate.seconds)}"


def describe_estimate(estimate: Estimate) -> list[str]:
    """Explain what the estimate is based on, in plain words."""
    lines = [
        f"PaperQA reads {estimate.passages} passage{'' if estimate.passages == 1 else 's'} with the model, "
        f"then writes an answer of up to about {estimate.answer_words} words from at most {estimate.sources} "
        f"of them, plus a few short calls to plan the search.",
    ]
    if estimate.whole_page:
        lines.append(
            "Images and tables are on with whole-page pictures, so passages are sent with a picture of each of "
            f"their pages: about {estimate.pictures} pictures in all. {PICTURE_NOTE}"
        )
    elif estimate.pictures:
        lines.append(
            "Images and tables are on, so passages are sent with their pictures: about "
            f"{estimate.pictures} pictures in all, guessed from a typical page. {PICTURE_NOTE}"
        )
    if estimate.local:
        lines.append(LOCAL_COST_NOTE)
    elif estimate.cost_usd is None:
        lines.append(f"Cost: not known for {estimate.model}.")
    else:
        lines.append(f"Cost: {describe_cost(estimate.cost_usd)}.")
    time_line = f"Time: {describe_time(estimate.seconds)}."
    if estimate.local:
        time_line += f" {LOCAL_TIME_NOTE}"
    lines.append(time_line)
    lines.append(ROUGH_NOTE)
    return lines


def badge(label: str) -> str:
    """Return a label as a coloured Streamlit badge."""
    return f":{BADGE_COLOURS[label]}-badge[{label}]"
