"""PDF parser helpers with a targeted fallback for known pypdf font-encoding failures."""

from __future__ import annotations

import importlib
from pathlib import Path


def is_crop_error(error: BaseException) -> bool:
    """Return whether cutting a figure or table out of a page failed because it reached past the page."""
    return isinstance(error, ValueError) and "Crop exceeds page dimensions" in str(error)


def should_fallback_pdf_parse(error: BaseException) -> bool:
    """Return whether a parser error matches the known SymbolSetEncoding or figure-crop failure."""
    return isinstance(error, LookupError) and "/SymbolSetEncoding" in str(error) or is_crop_error(error)


def resolve_pdfplumber_page_indexes(
    page_range: int | tuple[int, int] | None, total_pages: int
) -> list[int]:
    """Convert PaperQA-style page ranges into zero-based pdfplumber page indexes."""
    if page_range is None:
        return list(range(total_pages))
    if isinstance(page_range, int):
        return [page_range - 1]
    start_page, end_page = page_range
    return list(range(start_page - 1, end_page))


def parse_pdf_with_pdfplumber(
    path: str | Path,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    **_: object,
) -> object:
    """Parse PDF page text with pdfplumber as a fallback when pypdf cannot decode fonts."""
    try:
        pdfplumber = importlib.import_module("pdfplumber")
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for the fallback PDF parser. Install it with `pip install pdfplumber`."
        ) from exc
    paperqa_types = importlib.import_module("paperqa.types")

    parsed_pages: dict[str, str] = {}
    with pdfplumber.open(str(path)) as pdf:
        for page_index in resolve_pdfplumber_page_indexes(page_range, len(pdf.pages)):
            text = pdf.pages[page_index].extract_text() or ""
            if page_size_limit and len(text) > page_size_limit:
                raise ValueError(
                    f"The text in page {page_index} of {len(pdf.pages)} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path!r}."
                )
            parsed_pages[str(page_index + 1)] = text
    return paperqa_types.ParsedText(
        content=parsed_pages,
        metadata=paperqa_types.ParsedMetadata(
            parsing_libraries=["pdfplumber"],
            total_parsed_text_length=sum(len(text) for text in parsed_pages.values()),
            count_parsed_media=0,
        ),
    )


def parse_pdf_with_fallback(path: str | Path, **kwargs: object) -> dict[str, str]:
    """Parse PDFs with PaperQA's default pypdf parser, falling back for SymbolSetEncoding and figure-crop errors.

    When a figure or table cannot be cut out of a page, the PDF is read again with a picture of each whole
    page, so a figure-heavy PDF keeps its pictures. Only if that fails too is the text read on its own.
    """
    paperqa_pypdf = importlib.import_module("paperqa_pypdf")
    try:
        return paperqa_pypdf.parse_pdf_to_pages(path, **kwargs)
    except Exception as exc:
        if not should_fallback_pdf_parse(exc):
            raise
        if not is_crop_error(exc):
            return parse_pdf_with_pdfplumber(path, **kwargs)
    # PaperQA's reader reads media unless told not to, so a missing parse_media means on.
    if kwargs.get("parse_media", True) and not kwargs.get("full_page", False):
        try:
            return paperqa_pypdf.parse_pdf_to_pages(path, **(kwargs | {"full_page": True}))
        except ValueError as exc:
            if not is_crop_error(exc):
                raise
    return parse_pdf_with_pdfplumber(path, **(kwargs | {"parse_media": False}))

