import importlib
import sys
from types import ModuleType
from unittest import mock

import pytest


@pytest.fixture
def pdf_parser_module():
    sys.modules.pop("pdf_research_assistant.pdf_parser", None)
    module = importlib.import_module("pdf_research_assistant.pdf_parser")
    yield module
    sys.modules.pop("pdf_research_assistant.pdf_parser", None)


def test_should_fallback_pdf_parse_returns_true_for_symbol_set_lookup_error(pdf_parser_module):
    error = LookupError("unknown encoding: /SymbolSetEncoding")

    assert pdf_parser_module.should_fallback_pdf_parse(error) is True


def test_should_fallback_pdf_parse_returns_false_for_other_lookup_errors(pdf_parser_module):
    error = LookupError("unknown encoding: /OtherEncoding")

    assert pdf_parser_module.should_fallback_pdf_parse(error) is False


def test_should_fallback_pdf_parse_returns_false_for_non_lookup_errors(pdf_parser_module):
    error = ValueError("bad metadata")

    assert pdf_parser_module.should_fallback_pdf_parse(error) is False


def test_should_fallback_pdf_parse_returns_true_for_crop_exceeds_page_dimensions_value_error(
    pdf_parser_module,
):
    error = ValueError("Crop exceeds page dimensions")

    assert pdf_parser_module.should_fallback_pdf_parse(error) is True


def test_resolve_pdfplumber_page_indexes_returns_all_pages_when_page_range_is_none(pdf_parser_module):
    assert pdf_parser_module.resolve_pdfplumber_page_indexes(None, total_pages=3) == [0, 1, 2]


def test_resolve_pdfplumber_page_indexes_returns_single_page_when_page_range_is_an_int(pdf_parser_module):
    assert pdf_parser_module.resolve_pdfplumber_page_indexes(2, total_pages=4) == [1]


def test_resolve_pdfplumber_page_indexes_returns_inclusive_range_when_page_range_is_a_tuple(pdf_parser_module):
    assert pdf_parser_module.resolve_pdfplumber_page_indexes((2, 3), total_pages=5) == [1, 2]


def test_parse_pdf_with_pdfplumber_raises_import_error_when_pdfplumber_is_unavailable(pdf_parser_module):
    with mock.patch.dict(sys.modules, {"pdfplumber": None}):
        with pytest.raises(ImportError, match="pdfplumber"):
            pdf_parser_module.parse_pdf_with_pdfplumber("example.pdf")


def test_parse_pdf_with_pdfplumber_returns_requested_page_texts(pdf_parser_module):
    pdfplumber_module = ModuleType("pdfplumber")
    paperqa_types_module = ModuleType("paperqa.types")
    page_one = mock.Mock()
    page_one.extract_text.return_value = "first page"
    page_two = mock.Mock()
    page_two.extract_text.return_value = "second page"
    fake_pdf = mock.MagicMock()
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False
    fake_pdf.pages = [page_one, page_two]
    pdfplumber_module.open = mock.Mock(return_value=fake_pdf)
    paperqa_types_module.ParsedMetadata = lambda **kwargs: kwargs
    paperqa_types_module.ParsedText = lambda **kwargs: mock.Mock(**kwargs)

    with mock.patch.dict(sys.modules, {"pdfplumber": pdfplumber_module, "paperqa.types": paperqa_types_module}):
        parsed_pages = pdf_parser_module.parse_pdf_with_pdfplumber("example.pdf", page_range=(1, 2))

    assert parsed_pages.content == {"1": "first page", "2": "second page"}
    assert parsed_pages.metadata["parsing_libraries"] == ["pdfplumber"]
    assert parsed_pages.metadata["total_parsed_text_length"] == len("first page") + len("second page")


def test_parse_pdf_with_pdfplumber_raises_value_error_when_page_size_limit_is_exceeded(pdf_parser_module):
    pdfplumber_module = ModuleType("pdfplumber")
    paperqa_types_module = ModuleType("paperqa.types")
    page = mock.Mock()
    page.extract_text.return_value = "too long"
    fake_pdf = mock.MagicMock()
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False
    fake_pdf.pages = [page]
    pdfplumber_module.open = mock.Mock(return_value=fake_pdf)
    paperqa_types_module.ParsedMetadata = lambda **kwargs: kwargs
    paperqa_types_module.ParsedText = lambda **kwargs: mock.Mock(**kwargs)

    with mock.patch.dict(sys.modules, {"pdfplumber": pdfplumber_module, "paperqa.types": paperqa_types_module}):
        with pytest.raises(ValueError, match="exceeds the 3 char limit"):
            pdf_parser_module.parse_pdf_with_pdfplumber("example.pdf", page_size_limit=3)


def test_parse_pdf_with_fallback_returns_primary_parser_result_when_pypdf_succeeds(pdf_parser_module):
    paperqa_pypdf_module = ModuleType("paperqa_pypdf")
    paperqa_pypdf_module.parse_pdf_to_pages = mock.Mock(return_value={"1": "primary parser"})

    with mock.patch.dict(sys.modules, {"paperqa_pypdf": paperqa_pypdf_module}):
        parsed_pages = pdf_parser_module.parse_pdf_with_fallback("example.pdf")

    assert parsed_pages == {"1": "primary parser"}


def test_parse_pdf_with_fallback_uses_pdfplumber_when_pypdf_hits_symbol_set_encoding_lookup_error(
    pdf_parser_module,
):
    paperqa_pypdf_module = ModuleType("paperqa_pypdf")
    paperqa_pypdf_module.parse_pdf_to_pages = mock.Mock(
        side_effect=LookupError("unknown encoding: /SymbolSetEncoding")
    )

    with mock.patch.dict(sys.modules, {"paperqa_pypdf": paperqa_pypdf_module}):
        with mock.patch.object(
            pdf_parser_module,
            "parse_pdf_with_pdfplumber",
            return_value={"1": "fallback parser"},
        ) as fallback_parser:
            parsed_pages = pdf_parser_module.parse_pdf_with_fallback("example.pdf", page_range=2)

    assert parsed_pages == {"1": "fallback parser"}
    fallback_parser.assert_called_once_with("example.pdf", page_range=2)


CROP_ERROR = ValueError("Crop exceeds page dimensions")


def _parse_with_primary_results(pdf_parser_module, results, **kwargs):
    """Parse example.pdf with the primary parser giving each result in turn; return the pages and both mocks."""
    paperqa_pypdf_module = ModuleType("paperqa_pypdf")
    paperqa_pypdf_module.parse_pdf_to_pages = mock.Mock(side_effect=results)

    with mock.patch.dict(sys.modules, {"paperqa_pypdf": paperqa_pypdf_module}):
        with mock.patch.object(
            pdf_parser_module,
            "parse_pdf_with_pdfplumber",
            return_value={"1": "fallback parser"},
        ) as fallback_parser:
            parsed_pages = pdf_parser_module.parse_pdf_with_fallback("example.pdf", page_range=2, **kwargs)
    return parsed_pages, paperqa_pypdf_module.parse_pdf_to_pages, fallback_parser


def test_parse_pdf_with_fallback_keeps_whole_page_pictures_when_cutting_out_a_figure_fails(pdf_parser_module):
    parsed_pages, primary_parser, fallback_parser = _parse_with_primary_results(
        pdf_parser_module, [CROP_ERROR, {"1": "whole pages"}], parse_media=True, dpi=150
    )

    assert parsed_pages == {"1": "whole pages"}
    assert primary_parser.call_args_list[1] == mock.call(
        "example.pdf", page_range=2, parse_media=True, dpi=150, full_page=True
    )
    fallback_parser.assert_not_called()


def test_parse_pdf_with_fallback_treats_unset_media_as_on_like_the_primary_parser(pdf_parser_module):
    parsed_pages, primary_parser, _fallback_parser = _parse_with_primary_results(
        pdf_parser_module, [CROP_ERROR, {"1": "whole pages"}]
    )

    assert parsed_pages == {"1": "whole pages"}
    assert primary_parser.call_args_list[1] == mock.call("example.pdf", page_range=2, full_page=True)


def test_parse_pdf_with_fallback_reads_text_only_when_whole_page_pictures_also_fail(pdf_parser_module):
    parsed_pages, _primary_parser, fallback_parser = _parse_with_primary_results(
        pdf_parser_module, [CROP_ERROR, CROP_ERROR], parse_media=True
    )

    assert parsed_pages == {"1": "fallback parser"}
    fallback_parser.assert_called_once_with("example.pdf", page_range=2, parse_media=False)


def test_parse_pdf_with_fallback_reraises_other_errors_from_the_whole_page_retry(pdf_parser_module):
    with pytest.raises(RuntimeError, match="render failed"):
        _parse_with_primary_results(pdf_parser_module, [CROP_ERROR, RuntimeError("render failed")], parse_media=True)


def test_parse_pdf_with_fallback_reads_text_only_when_whole_page_pictures_were_already_asked_for(
    pdf_parser_module,
):
    parsed_pages, primary_parser, fallback_parser = _parse_with_primary_results(
        pdf_parser_module, [CROP_ERROR], parse_media=True, full_page=True
    )

    assert parsed_pages == {"1": "fallback parser"}
    primary_parser.assert_called_once()
    fallback_parser.assert_called_once_with("example.pdf", page_range=2, parse_media=False, full_page=True)


def test_parse_pdf_with_fallback_reads_text_only_after_a_crop_error_with_media_off(pdf_parser_module):
    parsed_pages, primary_parser, fallback_parser = _parse_with_primary_results(
        pdf_parser_module, [CROP_ERROR], parse_media=False
    )

    assert parsed_pages == {"1": "fallback parser"}
    primary_parser.assert_called_once()
    fallback_parser.assert_called_once_with("example.pdf", page_range=2, parse_media=False)


def test_parse_pdf_with_fallback_reraises_lookup_errors_that_do_not_match_symbol_set_encoding(
    pdf_parser_module,
):
    paperqa_pypdf_module = ModuleType("paperqa_pypdf")
    paperqa_pypdf_module.parse_pdf_to_pages = mock.Mock(side_effect=LookupError("unknown encoding: /OtherEncoding"))

    with mock.patch.dict(sys.modules, {"paperqa_pypdf": paperqa_pypdf_module}):
        with pytest.raises(LookupError, match="/OtherEncoding"):
            pdf_parser_module.parse_pdf_with_fallback("example.pdf")


def test_parse_pdf_with_fallback_reraises_non_lookup_errors_from_primary_parser(pdf_parser_module):
    paperqa_pypdf_module = ModuleType("paperqa_pypdf")
    paperqa_pypdf_module.parse_pdf_to_pages = mock.Mock(side_effect=RuntimeError("parser exploded"))

    with mock.patch.dict(sys.modules, {"paperqa_pypdf": paperqa_pypdf_module}):
        with pytest.raises(RuntimeError, match="parser exploded"):
            pdf_parser_module.parse_pdf_with_fallback("example.pdf")


# Issue 142: pypdf's default text mode moves italic words and citation years to the end of the line in some
# PDFs (ProQuest Ebook Central chapters). Layout mode reads those lines in the right order.
# A page of ordinary lines, so the few words split differently ("). " vs "2020).") are a small share as on real pages.
BODY = "\n".join(f"line {n} of the ordinary text on this page reads the same in both modes" for n in range(6))
MOVED_PLAIN = f"of student motivation (Blum ). I see a role for grades to 2020\n{BODY}"
MOVED_LAYOUT = "   of student motivation (Blum 2020). I see a role for grades to   \n\n" + BODY.replace("\n", "   \n  ")
FIXED_TEXT = f"of student motivation (Blum 2020). I see a role for grades to\n{BODY}"


CHAPTERMAKER = {"/Producer": "ProQuest Ebook Central™ (ChapterMaker)"}


def test_from_moved_word_producer_is_true_for_proquest_ebook_central_chapters(pdf_parser_module):
    assert pdf_parser_module.from_moved_word_producer(CHAPTERMAKER) is True


def test_from_moved_word_producer_is_false_for_other_producers(pdf_parser_module):
    assert pdf_parser_module.from_moved_word_producer({"/Producer": "Adobe PDF Library 15.0"}) is False


def test_from_moved_word_producer_is_false_without_a_producer(pdf_parser_module):
    assert pdf_parser_module.from_moved_word_producer({"/Creator": "Word"}) is False


def test_from_moved_word_producer_is_false_without_document_information(pdf_parser_module):
    assert pdf_parser_module.from_moved_word_producer(None) is False


def test_collapse_layout_text_squeezes_spaces_and_drops_blank_lines(pdf_parser_module):
    assert pdf_parser_module.collapse_layout_text("  a    b  \n\n   \n c  d ") == "a b\nc d"


def test_moved_word_lines_returns_lines_whose_words_layout_mode_orders_differently(pdf_parser_module):
    assert pdf_parser_module.moved_word_lines(MOVED_PLAIN, FIXED_TEXT) == [MOVED_PLAIN.splitlines()[0]]


def test_moved_word_lines_ignores_lines_with_the_same_order(pdf_parser_module):
    assert pdf_parser_module.moved_word_lines(FIXED_TEXT, FIXED_TEXT) == []


def test_moved_word_lines_ignores_lines_layout_mode_has_no_match_for(pdf_parser_module):
    assert pdf_parser_module.moved_word_lines("left column text right column text", "left column text") == []


def test_moved_word_lines_ignores_words_layout_mode_only_splits_or_joins(pdf_parser_module):
    assert pdf_parser_module.moved_word_lines("Short videos are pervasive", "Shor t videos are per vasive") == []


def test_moved_word_lines_ignores_short_lines(pdf_parser_module):
    assert pdf_parser_module.moved_word_lines("b a", "a b") == []


def test_holds_same_words_accepts_a_reordering(pdf_parser_module):
    assert pdf_parser_module.holds_same_words(MOVED_PLAIN, FIXED_TEXT) is True


def test_holds_same_words_rejects_text_split_into_letters(pdf_parser_module):
    assert pdf_parser_module.holds_same_words("Chapter Three Teaching", "C h a p t e r Three Teaching") is False


def test_holds_same_words_accepts_empty_text(pdf_parser_module):
    assert pdf_parser_module.holds_same_words("", "") is True


def test_layout_text_if_words_moved_returns_the_layout_text_for_a_moved_word_page(pdf_parser_module):
    assert pdf_parser_module.layout_text_if_words_moved(MOVED_PLAIN, MOVED_LAYOUT) == FIXED_TEXT


def test_layout_text_if_words_moved_returns_none_when_no_line_moved(pdf_parser_module):
    assert pdf_parser_module.layout_text_if_words_moved(FIXED_TEXT, FIXED_TEXT) is None


def test_layout_text_if_words_moved_returns_none_when_layout_mode_changes_the_words(pdf_parser_module):
    plain = MOVED_PLAIN + "\nChapter Three Teaching"
    layout = FIXED_TEXT + "\nC h a p t e r T h r e e T e a c h i n g"

    assert pdf_parser_module.layout_text_if_words_moved(plain, layout) is None


def _parsed(content):
    paperqa_types = importlib.import_module("paperqa.types")
    return paperqa_types.ParsedText(
        content=content,
        metadata=paperqa_types.ParsedMetadata(
            parsing_libraries=["pypdf"],
            total_parsed_text_length=sum(len(v if isinstance(v, str) else v[0]) for v in content.values()),
            count_parsed_media=0,
        ),
    )


def _fake_pypdf(layout_by_page, metadata=CHAPTERMAKER):
    """A pypdf stand-in whose pages give the layout text in the list (or raise it, if it is an exception)."""
    pages = []
    for layout in layout_by_page:
        page = mock.Mock()
        if isinstance(layout, Exception):
            page.extract_text.side_effect = layout
        else:
            page.extract_text.return_value = layout
        pages.append(page)
    module = ModuleType("pypdf")
    module.PdfReader = mock.Mock(return_value=mock.Mock(pages=pages, metadata=metadata))
    return module, pages


def test_fix_moved_words_replaces_moved_word_pages_and_keeps_their_media(pdf_parser_module):
    paperqa_types = importlib.import_module("paperqa.types")
    media_one = [paperqa_types.ParsedMedia(index=0, data=b"picture one", info={"page_num": 1})]
    media_two = [paperqa_types.ParsedMedia(index=0, data=b"picture two", info={"page_num": 2})]
    parsed = _parsed({"1": (FIXED_TEXT, media_one), "2": (MOVED_PLAIN, media_two)})
    fake_pypdf, pages = _fake_pypdf([FIXED_TEXT, MOVED_LAYOUT])

    with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        result = pdf_parser_module.fix_moved_words("example.pdf", parsed)

    assert result.content == {"1": (FIXED_TEXT, media_one), "2": (FIXED_TEXT, media_two)}
    assert result.metadata.parsing_libraries == ["pypdf", "pypdf (layout mode)"]
    assert result.metadata.total_parsed_text_length == 2 * len(FIXED_TEXT)
    pages[1].extract_text.assert_called_once_with(extraction_mode="layout")


def test_fix_moved_words_replaces_plain_text_pages(pdf_parser_module):
    parsed = _parsed({"3": MOVED_PLAIN})
    fake_pypdf, _pages = _fake_pypdf([FIXED_TEXT, FIXED_TEXT, MOVED_LAYOUT])

    with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        result = pdf_parser_module.fix_moved_words("example.pdf", parsed)

    assert result.content == {"3": FIXED_TEXT}


def test_fix_moved_words_leaves_pdfs_from_other_producers_alone(pdf_parser_module):
    parsed = _parsed({"1": MOVED_PLAIN})
    fake_pypdf, pages = _fake_pypdf([MOVED_LAYOUT], metadata={"/Producer": "Adobe PDF Library 15.0"})

    with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        result = pdf_parser_module.fix_moved_words("example.pdf", parsed)

    assert result.content == {"1": MOVED_PLAIN}
    assert result.metadata.parsing_libraries == ["pypdf"]
    pages[0].extract_text.assert_not_called()


def test_fix_moved_words_keeps_the_default_text_when_nothing_moved(pdf_parser_module):
    parsed = _parsed({"1": FIXED_TEXT})
    fake_pypdf, _pages = _fake_pypdf([FIXED_TEXT])

    with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        result = pdf_parser_module.fix_moved_words("example.pdf", parsed)

    assert result.content == {"1": FIXED_TEXT}
    assert result.metadata.parsing_libraries == ["pypdf"]


def test_fix_moved_words_keeps_the_default_text_when_layout_mode_fails(pdf_parser_module):
    parsed = _parsed({"1": MOVED_PLAIN})
    fake_pypdf, _pages = _fake_pypdf([RuntimeError("layout failed")])

    with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        result = pdf_parser_module.fix_moved_words("example.pdf", parsed)

    assert result.content == {"1": MOVED_PLAIN}


def test_fix_moved_words_returns_results_without_page_content_unchanged(pdf_parser_module):
    assert pdf_parser_module.fix_moved_words("example.pdf", {"1": "text"}) == {"1": "text"}


def test_parse_pdf_with_fallback_fixes_moved_words_after_the_primary_parser(pdf_parser_module):
    paperqa_pypdf_module = ModuleType("paperqa_pypdf")
    paperqa_pypdf_module.parse_pdf_to_pages = mock.Mock(return_value=mock.sentinel.parsed)

    with mock.patch.dict(sys.modules, {"paperqa_pypdf": paperqa_pypdf_module}):
        with mock.patch.object(pdf_parser_module, "fix_moved_words", return_value=mock.sentinel.fixed) as fix:
            parsed_pages = pdf_parser_module.parse_pdf_with_fallback("example.pdf")

    assert parsed_pages is mock.sentinel.fixed
    fix.assert_called_once_with("example.pdf", mock.sentinel.parsed)


def test_parse_pdf_with_fallback_fixes_moved_words_after_the_whole_page_retry(pdf_parser_module):
    with mock.patch.object(pdf_parser_module, "fix_moved_words", return_value=mock.sentinel.fixed) as fix:
        parsed_pages, _primary, _fallback = _parse_with_primary_results(
            pdf_parser_module, [CROP_ERROR, mock.sentinel.whole_pages], parse_media=True
        )

    assert parsed_pages is mock.sentinel.fixed
    fix.assert_called_once_with("example.pdf", mock.sentinel.whole_pages)
