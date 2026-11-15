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
