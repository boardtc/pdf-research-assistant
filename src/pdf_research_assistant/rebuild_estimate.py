"""A rough cost and time for a rebuild, worked out before it starts and without any paid call.

A rebuild indexes only the PDFs the index does not list yet, and those that failed before. For each one PaperQA:

- reads the PDF and splits its text into passages of `reader_config["chunk_chars"]` characters, each repeating
  the last `reader_config["overlap"]` of the one before, and sends them to the embedding model
  (`settings.embedding`) in one call;
- without a row in the paper list, asks the answer model (`settings.llm`) twice to read the PDF's citation
  from its first passage;
- with images and tables on with enrichment, asks `parsing.enrichment_llm` to describe each picture, sent with
  the text of the pages around it (`parsing.enrichment_page_radius`).

The PDFs are read locally. Reading every page of a large library takes minutes, and PDFs on a synced drive may
have to be fetched first, so the PDFs are read for a couple of seconds only, and the rest are assumed to be like
them. Most of that time goes on counting pages, which matters most, since PDFs differ far more in length than in
how much text a page holds; the text of a few pages of some of them then gives the text per page. (Scaling the PDFs
not read by their file size was tried on a library of 120 PDFs and was no closer, since some are mostly pictures.)
Pictures are not counted in the PDFs: they are guessed from a typical page, as for the question estimate.
"""

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import litellm
import pypdf
from paperqa.prompts import citation_prompt
from paperqa.settings import MultimodalOptions

from pdf_research_assistant import estimate

# How long the PDFs are read for before the estimate makes do with those read so far, and how long the page
# waits at most when one PDF is slow to open, such as one a synced drive must fetch first.
SAMPLE_SECONDS = 2.0
SAMPLE_WAIT_SECONDS = 4.0
# The share of that time spent opening PDFs to count their pages, before any text is read.
COUNTING_SHARE = 0.6
# Pages read from each PDF to measure how much text its pages hold.
PAGES_READ_PER_PDF = 3
# The citation PaperQA asks for is one line; reading it back as JSON is a short prompt and a short reply.
CITATION_OUTPUT_TOKENS = 60
STRUCTURED_CITATION_INPUT_TOKENS = 150
# PaperQA asks for "a detailed description" of each picture. A guess until it is measured.
ENRICHMENT_OUTPUT_TOKENS = 250
# Timed on one rebuild of 3 short PDFs from the app (27 Nov 2026, text only, 2 citation calls each): the rebuild
# process took 59 seconds from starting to its first write to the index, loading PaperQA and LiteLLM (2 of them
# were this estimate), 23 seconds for the 3 PDFs, and 22 more to save and close the index. Besides the citation
# calls and the time the app takes to read a PDF's text, each PDF took about 5 seconds: PaperQA's own reading of
# it, its call to the embedding model, and saving it to the index. A busy computer takes longer.
START_AND_FINISH_SECONDS = 80
PDF_SECONDS = 5.0

NOTHING_TO_INDEX_NOTE = (
    "No PDF needs indexing, so nothing is sent to the embedding model and the rebuild costs nothing"
)
LOCAL_NOTE = "no cost, as the model runs on your computer"
PICTURE_GUESS_NOTE = (
    f"This part is a guess: the pictures are counted from a typical page ({estimate.PICTURES_PER_PAGE} per page), "
    "not from these PDFs"
)
WHOLE_PAGE_NOTE = "Pictures are of whole pages, so every page is one picture"


@dataclass(frozen=True)
class PdfReading:
    pages: int
    open_seconds: float
    # Worked out for the whole PDF from the pages read; None until some of its text has been read.
    characters: int | None = None
    seconds_per_page: float | None = None

    @property
    def text_read(self) -> bool:
        return self.characters is not None and self.pages > 0


@dataclass(frozen=True)
class RebuildEstimate:
    pdfs: int
    # How many of them were read; the rest are assumed to be like them.
    read: int
    pages: int
    embedding_model: str
    embedding_tokens: int
    # None when the price is not known.
    embedding_cost: float | None
    citation_pdfs: int
    citation_model: str
    citation_cost: float | None
    # None when images and tables are not enriched.
    enrichment_model: str | None
    pictures: int
    enrichment_cost: float | None
    # None when no PDF could be read.
    seconds: float | None
    whole_page: bool = False
    local: bool = False

    @property
    def total_cost(self) -> float | None:
        parts = (self.embedding_cost, self.citation_cost, self.enrichment_cost)
        return None if None in parts else sum(parts)


def pages_to_read(pages: int, count: int = PAGES_READ_PER_PDF) -> list[int]:
    """Return the pages to read from a PDF: all of a short one, or a few spread through a long one."""
    if pages <= count:
        return list(range(pages))
    return [pages * (number + 1) // (count + 1) for number in range(count)]


def open_and_count(path: Path, open_pdf: Callable = pypdf.PdfReader, clock: Callable[[], float] = time.perf_counter):
    """Open a PDF and count its pages; return the reader too, or None for a PDF that cannot be opened.

    PaperQA cannot index a PDF that cannot be opened either, so it costs nothing and counts as no pages.
    """
    start = clock()
    try:
        reader = open_pdf(path)
        pages = len(reader.pages)
    except Exception:
        return None, PdfReading(pages=0, open_seconds=clock() - start)
    return reader, PdfReading(pages=pages, open_seconds=clock() - start)


def read_text(reader, reading: PdfReading, clock: Callable[[], float] = time.perf_counter) -> PdfReading:
    """Work out a PDF's text length, and how long its pages take to read, from a few of its pages."""
    chosen = pages_to_read(reading.pages)
    start = clock()
    characters = 0
    for number in chosen:
        try:
            characters += len(reader.pages[number].extract_text() or "")
        except Exception:
            continue
    return replace(
        reading,
        characters=round(characters / len(chosen) * reading.pages),
        seconds_per_page=(clock() - start) / len(chosen),
    )


def spread_order(paths: list[Path]) -> list[Path]:
    """Return the PDFs in a mixed order that is the same every time, so a sample comes from the whole library."""
    mixed = sorted(paths)
    random.Random(0).shuffle(mixed)
    return mixed


def read_sample(
    paths: list[Path],
    readings: list[PdfReading],
    seconds: float = SAMPLE_SECONDS,
    clock: Callable[[], float] = time.perf_counter,
    open_pdf: Callable = pypdf.PdfReader,
    order: Callable[[list[Path]], list[Path]] = spread_order,
) -> None:
    """Count the pages of PDFs into readings, then read some of their text, until the time is up.

    At least one PDF is counted and read. Readings are added and then completed in place, so what is in the list
    can be used at any moment.
    """
    start = clock()
    opened = []
    for path in order(list(paths)):
        if readings and clock() >= start + seconds * COUNTING_SHARE:
            break
        reader, reading = open_and_count(path, open_pdf, clock)
        readings.append(reading)
        if reading.pages:
            opened.append((len(readings) - 1, reader))
    for number, (index, reader) in enumerate(opened):
        if number and clock() >= start + seconds:
            return
        readings[index] = read_text(reader, readings[index], clock)


def read_sample_in_time(
    paths: list[Path],
    seconds: float = SAMPLE_SECONDS,
    wait: float = SAMPLE_WAIT_SECONDS,
    open_pdf: Callable = pypdf.PdfReader,
    order: Callable[[list[Path]], list[Path]] = spread_order,
) -> list[PdfReading]:
    """Read a sample of the PDFs, and return what was read by the time the wait is over.

    A PDF that is slow to open cannot be cut short, so the reading goes on in the background and is not waited for.
    """
    readings: list[PdfReading] = []
    worker = threading.Thread(
        target=read_sample,
        args=(paths, readings, seconds),
        kwargs={"open_pdf": open_pdf, "order": order},
        daemon=True,
    )
    worker.start()
    worker.join(wait)
    return list(readings)


def embedding_price(model: str) -> float | None:
    """Return LiteLLM's price for each token sent to an embedding model, or None if it has none."""
    return (litellm.model_cost.get(model) or {}).get("input_cost_per_token")


def embedded_characters(settings: object, characters: int) -> int:
    """Return how many characters of a PDF go to the embedding model: its text, with each passage's overlap."""
    if not characters:
        return 0
    chunk_chars = settings.parsing.reader_config["chunk_chars"]
    overlap = settings.parsing.reader_config["overlap"]
    passages = math.ceil(max(characters - overlap, 1) / (chunk_chars - overlap))
    # Each passage after the first repeats the end of the one before.
    return characters + (passages - 1) * overlap


def citation_input_tokens(first_passage_characters: float) -> int:
    """Return the tokens sent to read one PDF's citation: the prompt and the PDF's first passage."""
    return estimate.count_prompt_tokens(citation_prompt) + round(first_passage_characters) // estimate.CHARACTERS_PER_TOKEN


def enrichment_input_tokens(settings: object, characters_per_page: float, pages_per_pdf: float = 1) -> int:
    """Return the tokens sent to describe one picture: the picture, the prompt and the text of the pages near it."""
    radius = settings.parsing.enrichment_page_radius
    pages = pages_per_pdf if radius == -1 else 2 * radius + 1
    width, height = estimate.picture_pixels(settings)
    return (
        estimate.count_picture_tokens(settings.parsing.enrichment_llm, width, height)
        + estimate.count_prompt_tokens(settings.parsing.enrichment_prompt)
        + round(pages * characters_per_page) // estimate.CHARACTERS_PER_TOKEN
    )


def citation_cost(settings: object, pdfs: int, local: bool, first_passage_characters: float) -> float | None:
    """Return the cost of reading these PDFs' citations, whose first passages hold this many characters on average."""
    if not pdfs or local:
        return 0.0
    read = estimate.call_cost(settings.llm, citation_input_tokens(first_passage_characters), CITATION_OUTPUT_TOKENS)
    structure = estimate.call_cost(settings.llm, STRUCTURED_CITATION_INPUT_TOKENS, CITATION_OUTPUT_TOKENS)
    if read is None or structure is None:
        return None
    return pdfs * (read + structure)


def estimate_rebuild(
    settings: object,
    readings: list[PdfReading],
    pdfs: int,
    citation_pdfs: int,
    local: bool = False,
) -> RebuildEstimate:
    """Estimate a rebuild that indexes this many PDFs, from readings of some of them. Makes no API calls.

    citation_pdfs counts the PDFs without a row in the paper list; local says the answer model is local.
    """
    enrich = MultimodalOptions.from_value(settings.parsing.multimodal).should_parse_and_enrich_media[1]
    whole_page = enrich and settings.parsing.reader_config.get("full_page", False)
    empty = dict(
        pdfs=pdfs,
        read=len(readings),
        embedding_model=settings.embedding,
        citation_pdfs=citation_pdfs,
        citation_model=settings.llm,
        enrichment_model=settings.parsing.enrichment_llm if enrich else None,
        whole_page=whole_page,
        local=local,
    )
    if not pdfs:
        zero = dict(embedding_cost=0.0, citation_cost=0.0, enrichment_cost=0.0, seconds=0.0)
        return RebuildEstimate(**empty, **zero, pages=0, embedding_tokens=0, pictures=0)
    if not readings:
        unknown = dict(embedding_cost=None, citation_cost=None, enrichment_cost=None, seconds=None)
        return RebuildEstimate(**empty, **unknown, pages=0, embedding_tokens=0, pictures=0)

    scale = pdfs / len(readings)
    pages = sum(reading.pages for reading in readings) * scale
    # The PDFs whose pages were counted but whose text was not read are taken to hold as much text per page,
    # and to read as fast, as those that were read.
    text_read = [reading for reading in readings if reading.text_read]
    pages_read = sum(reading.pages for reading in text_read)
    characters_per_page = sum(reading.characters for reading in text_read) / pages_read if pages_read else 0
    seconds_per_page = (
        sum(reading.seconds_per_page * reading.pages for reading in text_read) / pages_read if pages_read else 0
    )
    pdf_characters = [
        reading.characters if reading.text_read else round(reading.pages * characters_per_page)
        for reading in readings
    ]
    characters = sum(pdf_characters) * scale
    embedding_tokens = (
        sum(embedded_characters(settings, count) for count in pdf_characters) * scale // estimate.CHARACTERS_PER_TOKEN
    )
    price = embedding_price(settings.embedding)
    embedding_cost = None if price is None else embedding_tokens * price

    # A PDF shorter than a passage is sent whole.
    chunk_chars = settings.parsing.reader_config["chunk_chars"]
    first_passage = sum(min(count, chunk_chars) for count in pdf_characters) / len(pdf_characters)

    seconds = (
        sum(reading.open_seconds + reading.pages * seconds_per_page for reading in readings) * scale
        + pdfs * PDF_SECONDS
        + citation_pdfs * 2 * estimate.call_seconds(CITATION_OUTPUT_TOKENS)
    )

    pictures = 0.0
    enrichment_cost = 0.0
    if enrich:
        pictures = pages * estimate.pictures_per_page(settings)
        per_call = estimate.call_cost(
            settings.parsing.enrichment_llm,
            enrichment_input_tokens(settings, characters / pages if pages else 0, pages / pdfs),
            ENRICHMENT_OUTPUT_TOKENS,
        )
        enrichment_cost = None if per_call is None else pictures * per_call
        # PaperQA describes a PDF's pictures all at once.
        seconds += pdfs * estimate.call_seconds(ENRICHMENT_OUTPUT_TOKENS)

    return RebuildEstimate(
        **empty,
        pages=round(pages),
        embedding_tokens=round(embedding_tokens),
        embedding_cost=embedding_cost,
        citation_cost=citation_cost(settings, citation_pdfs, local, first_passage),
        pictures=round(pictures),
        enrichment_cost=enrichment_cost,
        # PaperQA indexes this many PDFs side by side.
        seconds=seconds / max(settings.agent.index.concurrency, 1) + START_AND_FINISH_SECONDS,
    )


def estimate_files(
    settings: object, paper_dir: Path, files: list[str], listed: set[str], local: bool = False
) -> RebuildEstimate:
    """Estimate indexing these PDFs, as paths under the PDF folder; listed holds the paper list's paths."""
    paths = [Path(paper_dir) / file for file in files]
    readings = read_sample_in_time(paths) if files else []
    citation_pdfs = sum(1 for file in files if file not in listed)
    return estimate_rebuild(settings, readings, len(files), citation_pdfs, local)


def describe_cost(cost: float | None) -> str:
    if cost is None:
        return "cost not known"
    if cost == 0:
        return "nothing"
    if cost < 0.01:
        return "less than 1 cent"
    if cost < 1:
        return f"about {math.ceil(cost * 100)} cents"
    return f"about ${cost:.2f}"


def describe_time(seconds: float | None) -> str:
    if seconds is None:
        return "time not known"
    if seconds < 60:
        return "under a minute"
    if seconds < 3600:
        return f"about {math.ceil(seconds / 60)} minutes"
    return f"about {seconds / 3600:.1f} hours"


def describe_sample(read: int, pdfs: int) -> str:
    if read < pdfs:
        return (
            f"Worked out from {read} of the {pdfs} PDFs, counting their pages and reading a few pages of some, "
            "because reading them all would hold up the page. The rest are assumed to be like them"
        )
    return f"Worked out from all {pdfs} PDFs, counting their pages and reading a few pages of some"


def describe(result: RebuildEstimate) -> list[str]:
    """Explain the estimate in plain words, one line per part."""
    if not result.pdfs:
        return [NOTHING_TO_INDEX_NOTE]
    enrichment = (
        f"Images and tables are on with enrichment, so the model `{result.enrichment_model}` is also paid to "
        "describe every image and table"
    )
    if not result.read:
        lines = [
            f"Each PDF that is indexed is read, split into passages and sent to the embedding model "
            f"`{result.embedding_model}`, which costs money. None of the {result.pdfs} PDFs could be read quickly "
            "enough to estimate how much"
        ]
        return lines + [enrichment] if result.enrichment_model else lines
    lines = [
        f"{result.pdfs} PDFs will be indexed: about {result.pages:,} pages, sent to the embedding model "
        f"`{result.embedding_model}` as about {result.embedding_tokens:,} tokens. Embeddings: "
        f"{describe_cost(result.embedding_cost)}"
    ]
    if result.citation_pdfs:
        one = result.citation_pdfs == 1
        cost = LOCAL_NOTE if result.local else describe_cost(result.citation_cost)
        lines.append(
            f"{result.citation_pdfs} of these PDFs {'has' if one else 'have'} no row in the paper list, so "
            f"`{result.citation_model}` is asked twice for {'it' if one else 'each'} to read "
            f"{'its' if one else 'their'} citation: {cost}"
        )
    if result.enrichment_model:
        lines.append(
            f"{enrichment}: about {result.pictures:,} pictures, one call each: "
            f"{describe_cost(result.enrichment_cost)}. "
            f"{WHOLE_PAGE_NOTE if result.whole_page else PICTURE_GUESS_NOTE}"
        )
    lines.append(f"Rough total: {describe_cost(result.total_cost)}, {describe_time(result.seconds)}")
    lines.append(describe_sample(result.read, result.pdfs))
    return lines
