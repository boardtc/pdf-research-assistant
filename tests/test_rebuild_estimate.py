import math
import threading
from dataclasses import replace
from pathlib import Path

import litellm
import pytest
import tiktoken
from paperqa import Settings
from paperqa.settings import MultimodalOptions

from pdf_research_assistant import estimate, rebuild_estimate
from pdf_research_assistant.rebuild_estimate import PdfReading

TINY_PDF = Path(__file__).parent / "fixtures" / "tiny-enthusiasm.pdf"
EMBEDDING = "text-embedding-3-small"
MODEL = "gpt-4o-mini"


def _settings(
    media=False, whole_page=False, concurrency=1, radius=1, embedding=EMBEDDING, llm=MODEL, doc_details=True
) -> Settings:
    """Real PaperQA settings with the app's chunk size and picture resolution."""
    settings = Settings(llm=llm, summary_llm=llm, embedding=embedding)
    settings.parsing.use_doc_details = doc_details
    settings.parsing.multimodal = media
    settings.parsing.reader_config = {"chunk_chars": 5000, "overlap": 250, "dpi": 150, "full_page": whole_page}
    settings.parsing.enrichment_page_radius = radius
    settings.agent.index.concurrency = concurrency
    return settings


def _reading(pages=10, characters=30000, seconds=0.5, open_seconds=0.0, tokens=None) -> PdfReading:
    """A PDF whose text was read: seconds is how long all its pages take."""
    return PdfReading(
        pages=pages,
        open_seconds=open_seconds,
        characters=characters,
        seconds_per_page=seconds / pages if pages else 0,
        tokens=tokens,
    )


def _estimate(**changes) -> rebuild_estimate.RebuildEstimate:
    values = dict(
        pdfs=2,
        read=2,
        pages=20,
        embedding_model=EMBEDDING,
        embedding_tokens=15000,
        embedding_cost=0.0003,
        citation_pdfs=0,
        citation_model=MODEL,
        citation_calls=2,
        citation_cost=0.0,
        enrichment_model=None,
        pictures=0,
        enrichment_cost=0.0,
        seconds=40.0,
        whole_page=False,
        local=False,
    )
    return rebuild_estimate.RebuildEstimate(**(values | changes))


# pages_to_read


def test_pages_to_read_reads_the_middle_page_of_a_pdf():
    # One page from each of many PDFs gives the text per page more closely than a few pages from fewer PDFs.
    assert rebuild_estimate.pages_to_read(100) == [50]
    assert rebuild_estimate.pages_to_read(1) == [0]


def test_pages_to_read_reads_every_page_of_a_short_pdf():
    assert rebuild_estimate.pages_to_read(2, count=3) == [0, 1]


def test_pages_to_read_spreads_a_few_pages_through_a_long_pdf():
    assert rebuild_estimate.pages_to_read(100, count=3) == [25, 50, 75]


def test_pages_to_read_reads_nothing_of_a_pdf_without_pages():
    assert rebuild_estimate.pages_to_read(0) == []


# Reading the PDFs


class _Page:
    def __init__(self, text="x" * 1000, error=None):
        self.text, self.error = text, error

    def extract_text(self):
        if self.error:
            raise self.error
        return self.text


class _Reader:
    def __init__(self, pages):
        self.pages = pages


class _CountedReader:
    """A reader whose PDF records its page count, and whose pages would take time to list."""

    def __init__(self, count):
        self.trailer = {"/Root": {"/Pages": {"/Count": count}}}

    @property
    def pages(self):
        raise AssertionError("listed the pages")


def _ticks(*values):
    """A clock that gives these times, then the last one for ever."""
    values = list(values)
    return lambda: values.pop(0) if len(values) > 1 else values[0]


def test_open_and_count_counts_the_pages_of_a_real_pdf():
    reader, reading = rebuild_estimate.open_and_count(TINY_PDF)

    assert reading.pages == 1
    assert reading.open_seconds > 0
    assert reading.text_read is False
    assert len(reader.pages) == 1


def test_count_pages_reads_the_page_count_the_pdf_records():
    assert rebuild_estimate.count_pages(_CountedReader(7)) == 7


def test_count_pages_lists_the_pages_when_the_pdf_records_no_count():
    assert rebuild_estimate.count_pages(_Reader([_Page()] * 3)) == 3


def test_open_and_count_counts_a_pdf_that_cannot_be_opened_as_empty(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_text("not a pdf", encoding="utf-8")

    reader, reading = rebuild_estimate.open_and_count(broken)

    assert reader is None
    assert reading.pages == 0


def test_read_text_reads_a_real_pdfs_text():
    reader, reading = rebuild_estimate.open_and_count(TINY_PDF)

    reading = rebuild_estimate.read_text(reader, reading)

    # The fixture's five lines, as pypdf reads them.
    assert 150 < reading.characters < 200
    assert reading.seconds_per_page > 0
    assert reading.text_read is True


def test_read_text_scales_the_pages_it_read_to_the_whole_pdf():
    reading = rebuild_estimate.read_text(_Reader([_Page()] * 30), PdfReading(pages=30, open_seconds=0.1))

    assert reading.characters == 30000
    # Without a tokenizer, four characters make a token.
    assert reading.tokens == 30000 // estimate.CHARACTERS_PER_TOKEN


def test_read_text_counts_listing_the_pages_as_opening_the_pdf_not_reading_its_pages():
    # Listing a long PDF's pages takes a while once, the first time; it must not be charged to every page.
    reading = rebuild_estimate.read_text(
        _Reader([_Page()] * 100), PdfReading(pages=100, open_seconds=0.1), clock=_ticks(0, 5, 6)
    )

    assert reading.open_seconds == pytest.approx(5.1)
    assert reading.seconds_per_page == pytest.approx(1)


def test_read_text_reads_nothing_from_a_pdf_whose_pages_cannot_be_listed():
    reading = rebuild_estimate.read_text(_CountedReader(4), PdfReading(pages=4, open_seconds=0.1))

    assert (reading.characters, reading.tokens) == (0, 0)


def test_read_text_counts_tokens_with_the_tokenizer_given():
    reading = rebuild_estimate.read_text(
        _Reader([_Page("one two")] * 10), PdfReading(pages=10, open_seconds=0.1), count_tokens=lambda text: 2
    )

    assert reading.tokens == 20


def test_read_text_counts_a_page_whose_text_cannot_be_read_as_empty():
    reading = rebuild_estimate.read_text(_Reader([_Page(error=ValueError("bad font"))]), PdfReading(1, 0.1))

    assert reading.characters == 0
    assert reading.tokens == 0


# token_counter


def test_token_counter_uses_the_embedding_models_own_tokenizer():
    words = "Teacher enthusiasm and student motivation"
    expected = len(tiktoken.encoding_for_model(EMBEDDING).encode(words))

    assert rebuild_estimate.token_counter(EMBEDDING)(words) == expected
    # LiteLLM's provider prefix makes no difference.
    assert rebuild_estimate.token_counter(f"openai/{EMBEDDING}")(words) == expected


def test_token_counter_falls_back_to_openais_usual_tokenizer_for_other_models():
    words = "Teacher enthusiasm and student motivation"

    assert rebuild_estimate.token_counter("ollama/nomic-embed-text")(words) == len(
        tiktoken.get_encoding(rebuild_estimate.FALLBACK_ENCODING).encode(words)
    )


def test_token_counter_counts_text_that_looks_like_a_special_token():
    assert rebuild_estimate.token_counter(EMBEDDING)("<|endoftext|>") > 1


def test_a_pdf_without_pages_has_no_text_to_read():
    assert PdfReading(pages=0, open_seconds=0.1, characters=0).text_read is False


# read_sample


def test_read_sample_counts_and_reads_every_pdf_when_there_is_time():
    readings = []

    rebuild_estimate.read_sample(
        [Path(f"{n}.pdf") for n in range(5)], readings, open_pdf=lambda path: _Reader([_Page()] * 4)
    )

    assert len(readings) == 5
    assert all(reading.text_read for reading in readings)


def test_read_sample_counts_pages_first_then_reads_text_while_there_is_time():
    # Each look at the clock takes a second. Counting stops at 60% of the 12 seconds, after three PDFs; reading
    # text stops when all 12 have gone, after one.
    clock = _ticks(*range(100))
    readings = []

    rebuild_estimate.read_sample(
        [Path(f"{n}.pdf") for n in range(9)],
        readings,
        seconds=12,
        clock=clock,
        open_pdf=lambda path: _Reader([_Page()] * 2),
        order=list,
    )

    assert [reading.pages for reading in readings] == [2, 2, 2]
    assert [reading.text_read for reading in readings] == [True, False, False]


def test_read_sample_counts_tokens_with_the_tokenizer_given():
    readings = []

    rebuild_estimate.read_sample(
        [Path("a.pdf")], readings, open_pdf=lambda path: _Reader([_Page()] * 4), count_tokens=lambda text: 5
    )

    assert readings[0].tokens == 20


def test_read_sample_counts_and_reads_at_least_one_pdf():
    readings = []

    rebuild_estimate.read_sample(
        [Path("a.pdf"), Path("b.pdf")],
        readings,
        seconds=1,
        clock=_ticks(0, 5),
        open_pdf=lambda path: _Reader([_Page()]),
    )

    assert len(readings) == 1
    assert readings[0].text_read is True


def test_read_sample_reads_no_text_from_a_pdf_that_could_not_be_opened():
    def open_pdf(path):
        raise ValueError("not a pdf")

    readings = []

    rebuild_estimate.read_sample([Path("bad.pdf")], readings, open_pdf=open_pdf)

    assert readings == [PdfReading(pages=0, open_seconds=readings[0].open_seconds)]


def test_read_sample_reads_pdfs_from_across_the_library_not_just_the_first_ones():
    paths = [Path(f"{n:03}.pdf") for n in range(100)]

    def sample():
        opened = []
        rebuild_estimate.read_sample(
            paths, [], seconds=5, clock=_ticks(*range(100)), open_pdf=lambda path: opened.append(path) or _Reader([])
        )
        return opened

    first = sample()
    assert first != paths[: len(first)]
    # The same PDFs each time, so the estimate does not change from one page load to the next.
    assert sample() == first


def test_read_sample_in_time_returns_what_was_read_when_a_pdf_takes_too_long():
    release = threading.Event()

    def open_pdf(path):
        if path.name == "slow.pdf":
            release.wait(5)
        return _Reader([_Page()])

    paths = [Path("a.pdf"), Path("slow.pdf"), Path("b.pdf")]
    try:
        readings = rebuild_estimate.read_sample_in_time(paths, seconds=5, wait=0.3, open_pdf=open_pdf, order=list)
    finally:
        release.set()

    assert [reading.pages for reading in readings] == [1]


def test_read_sample_in_time_returns_every_reading_when_all_finish():
    readings = rebuild_estimate.read_sample_in_time([TINY_PDF, TINY_PDF])

    assert [reading.pages for reading in readings] == [1, 1]
    assert all(reading.text_read for reading in readings)


def test_read_sample_in_time_counts_tokens_with_the_tokenizer_given():
    readings = rebuild_estimate.read_sample_in_time([TINY_PDF], count_tokens=lambda text: 3)

    assert readings[0].tokens == 3


# Embedding price


def test_embedding_price_comes_from_litellms_price_list():
    assert rebuild_estimate.embedding_price(EMBEDDING) == litellm.model_cost[EMBEDDING]["input_cost_per_token"]


def test_embedding_price_is_none_for_a_model_litellm_does_not_price():
    assert rebuild_estimate.embedding_price("no-such-embedding") is None


# estimate_rebuild


def test_nothing_to_index_costs_nothing():
    result = rebuild_estimate.estimate_rebuild(_settings(), [], pdfs=0, citation_pdfs=0)

    assert result.total_cost == 0
    assert result.seconds == 0


def test_no_readings_leaves_the_cost_and_time_unknown():
    result = rebuild_estimate.estimate_rebuild(_settings(), [], pdfs=3, citation_pdfs=0)

    assert result.read == 0
    assert result.total_cost is None
    assert result.seconds is None


def test_embedding_tokens_count_each_passage_with_its_overlap():
    # 9,500 characters make two 5,000-character passages; the second repeats the last 250 of the first.
    result = rebuild_estimate.estimate_rebuild(_settings(), [_reading(characters=9500)], pdfs=1, citation_pdfs=0)

    assert result.embedding_tokens == round((9500 + 250) / estimate.CHARACTERS_PER_TOKEN)
    assert result.embedding_cost == pytest.approx(result.embedding_tokens * rebuild_estimate.embedding_price(EMBEDDING))


def test_embedding_tokens_use_the_tokens_counted_in_the_text_read():
    # 9,500 characters counted as 1,900 tokens: five characters a token, overlap included.
    result = rebuild_estimate.estimate_rebuild(
        _settings(), [_reading(characters=9500, tokens=1900)], pdfs=1, citation_pdfs=0
    )

    assert result.embedding_tokens == round((9500 + 250) / 5)


def test_a_pdf_shorter_than_a_passage_is_sent_once_without_any_overlap():
    result = rebuild_estimate.estimate_rebuild(_settings(), [_reading(characters=3000)], pdfs=1, citation_pdfs=0)

    assert result.embedding_tokens == 3000 // estimate.CHARACTERS_PER_TOKEN


def test_a_pdf_without_text_sends_nothing_to_the_embedding_model():
    result = rebuild_estimate.estimate_rebuild(_settings(), [_reading(characters=0)], pdfs=1, citation_pdfs=0)

    assert result.embedding_tokens == 0


def test_the_pdfs_not_read_are_assumed_to_be_like_those_read():
    one = rebuild_estimate.estimate_rebuild(_settings(), [_reading()], pdfs=1, citation_pdfs=0)
    many = rebuild_estimate.estimate_rebuild(_settings(), [_reading()], pdfs=10, citation_pdfs=0)
    fixed = rebuild_estimate.START_AND_FINISH_SECONDS

    assert many.read == 1
    assert many.pages == 10 * one.pages
    assert many.embedding_tokens == pytest.approx(10 * one.embedding_tokens, abs=10)
    # Starting and finishing the rebuild take the same time however many PDFs there are.
    assert many.seconds - fixed == pytest.approx(10 * (one.seconds - fixed))


def test_an_unpriced_embedding_model_leaves_the_cost_unknown():
    result = rebuild_estimate.estimate_rebuild(
        _settings(embedding="no-such-embedding"), [_reading()], pdfs=1, citation_pdfs=0
    )

    assert result.embedding_cost is None
    assert result.total_cost is None


def test_pdfs_without_a_paper_list_row_pay_the_answer_model_to_read_their_citation():
    without = rebuild_estimate.estimate_rebuild(_settings(), [_reading()], pdfs=4, citation_pdfs=0)
    result = rebuild_estimate.estimate_rebuild(_settings(), [_reading()], pdfs=4, citation_pdfs=3)

    assert without.citation_cost == 0
    # Each of the PDFs has 3,000 characters a page; PaperQA reads the citation from the first three pages, cut to
    # one passage of 5,000 characters: 1,250 tokens.
    one_pdf = estimate.call_cost(
        MODEL,
        rebuild_estimate.citation_input_tokens(1250),
        rebuild_estimate.CITATION_OUTPUT_TOKENS,
    ) + estimate.call_cost(
        MODEL, rebuild_estimate.STRUCTURED_CITATION_INPUT_TOKENS, rebuild_estimate.CITATION_OUTPUT_TOKENS
    )
    assert result.citation_cost == pytest.approx(3 * one_pdf)
    assert result.total_cost == pytest.approx(result.embedding_cost + result.citation_cost)
    assert result.seconds > without.seconds


def test_without_looking_citations_up_paperqa_asks_once_for_each():
    # The app turns PaperQA's use_doc_details off, so the citation is not read back as JSON to look it up.
    twice = rebuild_estimate.estimate_rebuild(_settings(), [_reading()], pdfs=4, citation_pdfs=3)
    once = rebuild_estimate.estimate_rebuild(_settings(doc_details=False), [_reading()], pdfs=4, citation_pdfs=3)
    read = estimate.call_cost(
        MODEL, rebuild_estimate.citation_input_tokens(1250), rebuild_estimate.CITATION_OUTPUT_TOKENS
    )

    assert (twice.citation_calls, once.citation_calls) == (2, 1)
    assert once.citation_cost == pytest.approx(3 * read)
    assert twice.seconds - once.seconds == pytest.approx(
        3 * estimate.call_seconds(rebuild_estimate.CITATION_OUTPUT_TOKENS)
    )


def test_citation_input_is_the_prompt_and_the_first_passage():
    assert rebuild_estimate.citation_input_tokens(500) == (
        estimate.count_prompt_tokens(rebuild_estimate.citation_prompt) + 500
    )


def _citation_cost(first_passage_tokens: int) -> float:
    return estimate.call_cost(
        MODEL, rebuild_estimate.citation_input_tokens(first_passage_tokens), rebuild_estimate.CITATION_OUTPUT_TOKENS
    ) + estimate.call_cost(
        MODEL, rebuild_estimate.STRUCTURED_CITATION_INPUT_TOKENS, rebuild_estimate.CITATION_OUTPUT_TOKENS
    )


def test_citation_reading_sends_only_the_first_three_pages():
    # 1,000 characters a page: the first three pages are 3,000 characters, 750 tokens.
    result = rebuild_estimate.estimate_rebuild(
        _settings(), [_reading(pages=10, characters=10000)], pdfs=1, citation_pdfs=1
    )

    assert result.citation_cost == pytest.approx(_citation_cost(750))


def test_citation_reading_sends_a_short_pdfs_whole_text():
    result = rebuild_estimate.estimate_rebuild(
        _settings(), [_reading(pages=1, characters=2000)], pdfs=1, citation_pdfs=1
    )

    assert result.citation_cost == pytest.approx(_citation_cost(500))


def test_citation_reading_counts_the_first_pages_of_pdfs_whose_text_was_not_read():
    counted = PdfReading(pages=40, open_seconds=0.25)

    result = rebuild_estimate.estimate_rebuild(
        _settings(), [_reading(pages=10, characters=10000), counted], pdfs=2, citation_pdfs=2
    )

    assert result.citation_cost == pytest.approx(2 * _citation_cost(750))


def test_reading_citations_with_a_local_model_costs_nothing():
    result = rebuild_estimate.estimate_rebuild(
        _settings(llm="ollama/llama3"), [_reading()], pdfs=1, citation_pdfs=1, local=True
    )

    assert result.citation_cost == 0
    assert result.local is True


def test_reading_citations_with_an_unpriced_model_leaves_the_cost_unknown():
    result = rebuild_estimate.estimate_rebuild(_settings(llm="no-such-model"), [_reading()], pdfs=1, citation_pdfs=1)

    assert result.citation_cost is None
    assert result.total_cost is None


@pytest.mark.parametrize("media", [False, True, MultimodalOptions.ON_WITHOUT_ENRICHMENT])
def test_no_enrichment_means_no_pictures_are_paid_for(media):
    result = rebuild_estimate.estimate_rebuild(_settings(media=media), [_reading()], pdfs=1, citation_pdfs=0)

    if media is True:
        # True turns enrichment on as well.
        assert result.enrichment_model is not None
    else:
        assert result.enrichment_model is None
        assert result.pictures == 0
        assert result.enrichment_cost == 0


def test_enrichment_pays_one_call_for_each_picture_guessed_from_a_typical_page():
    settings = _settings(media=MultimodalOptions.ON_WITH_ENRICHMENT)

    result = rebuild_estimate.estimate_rebuild(settings, [_reading(pages=10)], pdfs=1, citation_pdfs=0)

    assert result.enrichment_model == settings.parsing.enrichment_llm
    assert result.pictures == round(10 * estimate.PICTURES_PER_PAGE)
    per_call = estimate.call_cost(
        settings.parsing.enrichment_llm,
        rebuild_estimate.enrichment_input_tokens(settings, 3000, 10),
        rebuild_estimate.ENRICHMENT_OUTPUT_TOKENS,
    )
    assert result.enrichment_cost == pytest.approx(10 * estimate.PICTURES_PER_PAGE * per_call)
    assert result.total_cost == pytest.approx(result.embedding_cost + result.enrichment_cost)


def test_enrichment_with_whole_page_pictures_pays_for_every_page():
    settings = _settings(media=MultimodalOptions.ON_WITH_ENRICHMENT, whole_page=True)

    result = rebuild_estimate.estimate_rebuild(settings, [_reading(pages=10)], pdfs=1, citation_pdfs=0)

    assert result.pictures == 10
    assert result.whole_page is True


def test_enrichment_with_an_unpriced_model_leaves_the_cost_unknown():
    settings = _settings(media=MultimodalOptions.ON_WITH_ENRICHMENT)
    settings.parsing.enrichment_llm = "no-such-model"

    result = rebuild_estimate.estimate_rebuild(settings, [_reading()], pdfs=1, citation_pdfs=0)

    assert result.enrichment_cost is None
    assert result.total_cost is None


def test_enrichment_input_counts_the_picture_and_the_pages_around_it():
    settings = _settings(media=MultimodalOptions.ON_WITH_ENRICHMENT)
    width, height = estimate.picture_pixels(settings)

    tokens = rebuild_estimate.enrichment_input_tokens(settings, 2000, pages_per_pdf=10)

    picture = estimate.count_picture_tokens(settings.parsing.enrichment_llm, width, height)
    prompt = estimate.count_prompt_tokens(settings.parsing.enrichment_prompt)
    # A radius of one page: the picture's own page and one either side.
    assert tokens == picture + prompt + 3 * 2000 // estimate.CHARACTERS_PER_TOKEN


def test_enrichment_input_never_counts_more_pages_than_the_pdf_has():
    settings = _settings(media=MultimodalOptions.ON_WITH_ENRICHMENT)
    width, height = estimate.picture_pixels(settings)

    # PaperQA sends a one-page PDF's picture with that one page only.
    tokens = rebuild_estimate.enrichment_input_tokens(settings, 2000, pages_per_pdf=1)

    picture = estimate.count_picture_tokens(settings.parsing.enrichment_llm, width, height)
    prompt = estimate.count_prompt_tokens(settings.parsing.enrichment_prompt)
    assert tokens == picture + prompt + 2000 // estimate.CHARACTERS_PER_TOKEN


def test_enrichment_input_with_a_radius_of_all_pages_counts_the_whole_pdf():
    settings = _settings(media=MultimodalOptions.ON_WITH_ENRICHMENT, radius=-1)
    smaller = rebuild_estimate.enrichment_input_tokens(settings, 2000, pages_per_pdf=3)

    assert rebuild_estimate.enrichment_input_tokens(settings, 2000, pages_per_pdf=20) == (
        smaller + 17 * 2000 // estimate.CHARACTERS_PER_TOKEN
    )


def test_enrichment_adds_time_for_each_pdf_with_pictures():
    off = rebuild_estimate.estimate_rebuild(_settings(), [_reading()], pdfs=2, citation_pdfs=0)
    on = rebuild_estimate.estimate_rebuild(
        _settings(media=MultimodalOptions.ON_WITH_ENRICHMENT), [_reading()], pdfs=2, citation_pdfs=0
    )

    assert on.seconds == pytest.approx(off.seconds + 2 * estimate.call_seconds(rebuild_estimate.ENRICHMENT_OUTPUT_TOKENS))


def test_pdfs_only_counted_are_taken_to_hold_as_much_text_per_page_as_those_read():
    counted = PdfReading(pages=40, open_seconds=0.25)

    result = rebuild_estimate.estimate_rebuild(
        _settings(), [_reading(pages=10, characters=20000, seconds=1), counted], pdfs=2, citation_pdfs=0
    )
    both_read = rebuild_estimate.estimate_rebuild(
        _settings(),
        [_reading(pages=10, characters=20000, seconds=1), _reading(pages=40, characters=80000, seconds=4, open_seconds=0.25)],
        pdfs=2,
        citation_pdfs=0,
    )

    assert result.pages == 50
    assert result.embedding_tokens == both_read.embedding_tokens
    assert result.seconds == pytest.approx(both_read.seconds)


def test_without_any_text_read_the_counted_pages_send_nothing_to_embed():
    result = rebuild_estimate.estimate_rebuild(
        _settings(), [PdfReading(pages=10, open_seconds=0.1)], pdfs=1, citation_pdfs=0
    )

    assert result.pages == 10
    assert result.embedding_tokens == 0


def test_the_time_counts_opening_each_pdf_and_reading_its_pages():
    result = rebuild_estimate.estimate_rebuild(
        _settings(), [_reading(seconds=2, open_seconds=0.5)], pdfs=1, citation_pdfs=0
    )

    assert result.seconds == pytest.approx(2.5 + rebuild_estimate.PDF_SECONDS + rebuild_estimate.START_AND_FINISH_SECONDS)


def test_the_time_counts_reading_each_pdf_and_one_embedding_call():
    result = rebuild_estimate.estimate_rebuild(_settings(), [_reading(seconds=2)], pdfs=1, citation_pdfs=0)

    assert result.seconds == pytest.approx(2 + rebuild_estimate.PDF_SECONDS + rebuild_estimate.START_AND_FINISH_SECONDS)


def test_pdfs_indexed_side_by_side_take_less_time():
    one = rebuild_estimate.estimate_rebuild(_settings(), [_reading()], pdfs=4, citation_pdfs=0)
    two = rebuild_estimate.estimate_rebuild(_settings(concurrency=2), [_reading()], pdfs=4, citation_pdfs=0)
    fixed = rebuild_estimate.START_AND_FINISH_SECONDS

    assert two.seconds - fixed == pytest.approx((one.seconds - fixed) / 2)


# estimate_files


def test_estimate_files_reads_the_pdfs_to_index_and_counts_those_whose_citation_is_read(tmp_path):
    paper_dir = TINY_PDF.parent
    settings = _settings()
    settings.agent.index.manifest_file = _manifest(tmp_path, [str(Path(TINY_PDF.name))])

    result = rebuild_estimate.estimate_files(settings, paper_dir, [TINY_PDF.name, "missing.pdf"])

    assert result.pdfs == 2
    assert result.read == 2
    assert result.citation_pdfs == 1
    # The missing PDF is read as empty, so only the real one has pages.
    assert result.pages == 1
    # The tiny PDF's text was counted with the embedding model's tokenizer.
    assert 0 < result.embedding_tokens < 100


def test_estimate_files_reads_nothing_when_nothing_needs_indexing(monkeypatch):
    monkeypatch.setattr(rebuild_estimate, "read_sample_in_time", lambda *args, **kwargs: pytest.fail("read"))

    result = rebuild_estimate.estimate_files(_settings(), Path("papers"), [])

    assert result.pdfs == 0


# citation_pdfs


def _manifest(folder: Path, locations: list[str], citation: str = "Someone (2020)") -> Path:
    manifest = folder / "manifest.csv"
    rows = "".join(f'"{location}",A title,{citation},\n' for location in locations)
    manifest.write_text("file_location,title,citation,doi\n" + rows, encoding="utf-8")
    return manifest


def test_no_pdfs_means_no_citations_and_no_paper_list_read(monkeypatch, tmp_path):
    monkeypatch.setattr(rebuild_estimate.anyio, "run", lambda *args: pytest.fail("read the paper list"))
    settings = _settings()
    settings.agent.index.manifest_file = _manifest(tmp_path, ["a.pdf"])

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, []) == 0


def test_without_a_paper_list_every_pdfs_citation_is_read(tmp_path):
    settings = _settings()
    settings.agent.index.manifest_file = None

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["a.pdf", "sub/b.pdf"]) == 2


def test_a_pdf_whose_row_paperqa_finds_has_its_citation_already(tmp_path):
    settings = _settings()
    settings.agent.index.manifest_file = _manifest(tmp_path, ["a.pdf", "sub/b.pdf"])

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["a.pdf", "sub/b.pdf"]) == 0


def test_a_row_without_a_citation_has_its_citation_read(tmp_path):
    # The app leaves such rows out of what PaperQA is given, so PaperQA reads the citation from the PDF (#160).
    settings = _settings()
    settings.agent.index.manifest_file = _manifest(tmp_path, ["a.pdf"], citation="")

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["a.pdf"]) == 1


def test_rows_are_found_whichever_slashes_they_are_written_with(tmp_path):
    # PaperQA looks a PDF up by its path as the computer writes it, and the app keys the rows the same way (#160).
    settings = _settings()
    settings.agent.index.manifest_file = _manifest(tmp_path, ["a.pdf", "sub/b.pdf", "sub\\c.pdf"])

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["a.pdf", "sub/b.pdf", "sub/c.pdf"]) == 0


def test_the_paper_list_is_read_with_the_apps_reader_whatever_paperqa_has_been_given(monkeypatch, tmp_path):
    import paperqa.agents.search as search

    monkeypatch.setattr(search, "maybe_get_manifest", lambda *args: pytest.fail("used PaperQA's own reader"))
    settings = _settings()
    settings.agent.index.manifest_file = _manifest(tmp_path, ["sub/b.pdf"])

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["sub/b.pdf"]) == 0


def test_paperqa_also_finds_a_row_by_the_pdfs_full_path(tmp_path):
    settings = _settings()
    settings.agent.index.manifest_file = _manifest(tmp_path, [str(tmp_path.absolute() / "sub" / "b.pdf")])

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["sub/b.pdf", "c.pdf"]) == 1


def test_paperqa_finds_a_paper_list_named_from_the_pdf_folder(tmp_path):
    _manifest(tmp_path, ["a.pdf"])
    settings = _settings()
    settings.agent.index.paper_directory = str(tmp_path)
    settings.agent.index.manifest_file = "manifest.csv"

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["a.pdf"]) == 0


def test_a_paper_list_that_is_missing_gives_no_citations(tmp_path):
    settings = _settings()
    settings.agent.index.manifest_file = tmp_path / "missing.csv"

    assert rebuild_estimate.citation_pdfs(settings, tmp_path, ["a.pdf"]) == 1


# Wording


@pytest.mark.parametrize(
    "cost, words",
    [
        (None, "cost not known"),
        (0, "nothing"),
        (0.004, "less than 1 cent"),
        (0.012, "about 2 cents"),
        (0.5, "about 50 cents"),
        (3.456, "about $3.46"),
    ],
)
def test_describe_cost(cost, words):
    assert rebuild_estimate.describe_cost(cost) == words


@pytest.mark.parametrize(
    "seconds, words",
    [
        (None, "time not known"),
        (20, "under a minute"),
        (61, "about 2 minutes"),
        (3599, "about 60 minutes"),
        (5400, "about 1.5 hours"),
    ],
)
def test_describe_time(seconds, words):
    assert rebuild_estimate.describe_time(seconds) == words


def test_describe_says_nothing_is_paid_when_no_pdf_needs_indexing():
    lines = rebuild_estimate.describe(_estimate(pdfs=0, read=0))

    assert lines == [rebuild_estimate.NOTHING_TO_INDEX_NOTE]


def test_describe_says_when_no_pdf_could_be_read_in_time():
    lines = rebuild_estimate.describe(_estimate(read=0, embedding_cost=None, seconds=None))

    assert len(lines) == 1
    assert "`text-embedding-3-small`" in lines[0]
    assert "None of the 2 PDFs could be read quickly enough" in lines[0]


def test_describe_still_warns_about_enrichment_when_no_pdf_could_be_read_in_time():
    lines = rebuild_estimate.describe(
        _estimate(read=0, embedding_cost=None, seconds=None, enrichment_model="gpt-4o", enrichment_cost=None)
    )

    assert len(lines) == 2
    assert "`gpt-4o` is also paid to describe every image and table" in lines[1]


def test_describe_gives_the_embedding_cost_and_a_rough_total():
    lines = rebuild_estimate.describe(_estimate(pages=1234, embedding_tokens=987654))

    assert lines[0] == (
        "2 PDFs will be indexed: about 1,234 pages, sent to the embedding model `text-embedding-3-small` as about "
        "987,654 tokens. Embeddings: less than 1 cent"
    )
    assert "Rough total: less than 1 cent, under a minute" in lines
    assert lines[-1] == rebuild_estimate.describe_sample(2, 2)
    assert not any("citation" in line or "enrichment" in line for line in lines)


def test_describe_explains_the_citation_calls():
    lines = rebuild_estimate.describe(_estimate(citation_pdfs=1, citation_cost=0.0002))

    assert (
        "The paper list gives no citation for 1 of these PDFs, so `gpt-4o-mini` is asked twice for it to "
        "read its citation: less than 1 cent"
    ) in lines


def test_describe_explains_the_citation_calls_for_several_pdfs():
    lines = rebuild_estimate.describe(_estimate(citation_pdfs=2, citation_cost=0.0004))

    assert (
        "The paper list gives no citation for 2 of these PDFs, so `gpt-4o-mini` is asked twice for each to "
        "read its citation: less than 1 cent"
    ) in lines


def test_describe_says_when_each_citation_takes_one_call():
    lines = rebuild_estimate.describe(_estimate(citation_pdfs=2, citation_calls=1, citation_cost=0.0004))

    assert (
        "The paper list gives no citation for 2 of these PDFs, so `gpt-4o-mini` is asked once for each to "
        "read its citation: less than 1 cent"
    ) in lines


def test_describe_explains_the_citation_calls_of_a_local_model():
    lines = rebuild_estimate.describe(_estimate(citation_pdfs=2, citation_model="ollama/llama3", local=True))

    assert any(
        line.startswith("The paper list gives no citation for 2 of these PDFs")
        and line.endswith(rebuild_estimate.LOCAL_NOTE)
        for line in lines
    )


def test_describe_says_the_enrichment_part_is_a_guess():
    lines = rebuild_estimate.describe(_estimate(enrichment_model="gpt-4o", pictures=1400, enrichment_cost=2.5))

    line = next(line for line in lines if "enrichment" in line)
    assert "the model `gpt-4o` is also paid to describe every image and table" in line
    assert "about 1,400 pictures, one call each: about $2.50" in line
    assert line.endswith(rebuild_estimate.PICTURE_GUESS_NOTE)


def test_describe_says_whole_page_pictures_are_one_for_each_page():
    lines = rebuild_estimate.describe(
        _estimate(enrichment_model="gpt-4o", pictures=20, enrichment_cost=0.1, whole_page=True)
    )

    line = next(line for line in lines if "enrichment" in line)
    assert line.endswith(rebuild_estimate.WHOLE_PAGE_NOTE)


def test_describe_sample_says_when_only_some_pdfs_were_read():
    assert rebuild_estimate.describe_sample(12, 140) == (
        "Worked out from 12 of the 140 PDFs, counting their pages and reading a page of some, because "
        "reading them all would hold up the page. The rest are assumed to be like them"
    )


def test_describe_sample_says_when_every_pdf_was_read():
    assert rebuild_estimate.describe_sample(3, 3) == "Worked out from all 3 PDFs, counting their pages and reading a page of some"


def test_total_cost_adds_up_the_parts():
    assert _estimate(embedding_cost=0.1, citation_cost=0.2, enrichment_cost=0.3).total_cost == pytest.approx(0.6)
