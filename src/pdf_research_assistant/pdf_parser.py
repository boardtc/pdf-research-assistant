"""PDF parser helpers with targeted fixes for known pypdf failures: font encodings, figure crops and moved words."""

from __future__ import annotations

import importlib
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# pypdf's default text mode moves some words (italics, citation years) to the end of their line in ProQuest
# Ebook Central chapter PDFs (#142). Layout mode reads them in place but is twice as slow and misreads some other
# PDFs, so only PDFs from this producer are read again.
MOVED_WORD_PRODUCER = "ProQuest Ebook Central"
# Lines shorter than this are too common (headings, page numbers) to show a reordering.
MIN_MOVED_LINE_WORDS = 3
# Layout mode may split or join words differently; more than this share of changed words means it misread the page.
MAX_CHANGED_WORD_SHARE = 0.05
LAYOUT_LIBRARY = "pypdf (layout mode)"


def from_moved_word_producer(metadata: object) -> bool:
    """Return whether a PDF's document information names the producer whose PDFs pypdf reads with moved words."""
    producer = metadata.get("/Producer") if metadata else None
    return MOVED_WORD_PRODUCER in str(producer or "")


def collapse_layout_text(text: str) -> str:
    """Squeeze the column-filling spaces out of pypdf layout-mode text and drop blank lines."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def moved_word_lines(plain: str, layout: str) -> list[str]:
    """Return default-mode lines whose characters layout mode puts on one line in a different order.

    Lines are compared without spaces, so "(Blum ). … to 2020" matches "(Blum 2020). … to", and layout mode
    splitting or joining a word ("Shor t", "Isnot") does not count as a move.
    """
    layout_lines = {}
    for line in layout.splitlines():
        if len(line.split()) >= MIN_MOVED_LINE_WORDS:
            squeezed = "".join(line.split())
            layout_lines.setdefault("".join(sorted(squeezed)), squeezed)
    moved = []
    for line in plain.splitlines():
        if len(line.split()) < MIN_MOVED_LINE_WORDS:
            continue
        squeezed = "".join(line.split())
        layout_squeezed = layout_lines.get("".join(sorted(squeezed)))
        if layout_squeezed is not None and layout_squeezed != squeezed:
            moved.append(line)
    return moved


def holds_same_words(plain: str, layout: str) -> bool:
    """Return whether two readings of a page hold nearly the same words, whatever their order."""
    plain_words, layout_words = Counter(plain.split()), Counter(layout.split())
    changed = sum(((plain_words - layout_words) + (layout_words - plain_words)).values())
    return changed <= MAX_CHANGED_WORD_SHARE * max(1, sum(plain_words.values()))


def layout_text_if_words_moved(plain: str, layout: str) -> str | None:
    """Return the tidied layout-mode text when it puts moved words back without changing the words, else None."""
    layout = collapse_layout_text(layout)
    if moved_word_lines(plain, layout) and holds_same_words(plain, layout):
        return layout
    return None


def fix_moved_words(path: str | Path, parsed: object) -> object:
    """Re-read, in pypdf's layout mode, pages whose default-mode text has words moved to the line end.

    Only PDFs from the producer known for this are read again, and a page keeps its default text unless layout
    mode has the same words with some line in a different order. Pictures stay as they were read.
    """
    content = getattr(parsed, "content", None)
    if not isinstance(content, dict):
        return parsed
    reader = importlib.import_module("pypdf").PdfReader(str(path))
    if not from_moved_word_producer(reader.metadata):
        return parsed
    fixed = False
    for key, value in content.items():
        plain = value if isinstance(value, str) else value[0]
        try:
            layout = reader.pages[int(key) - 1].extract_text(extraction_mode="layout")
        except Exception as exc:  # A failed second reading must not lose the first one.
            logger.warning("Layout-mode reading of page %s of %s failed: %s", key, path, exc)
            continue
        replacement = layout_text_if_words_moved(plain, layout)
        if replacement is None:
            continue
        content[key] = replacement if isinstance(value, str) else (replacement, value[1])
        fixed = True
    if fixed:
        parsed.metadata.parsing_libraries.append(LAYOUT_LIBRARY)
        parsed.metadata.total_parsed_text_length = sum(
            len(value if isinstance(value, str) else value[0]) for value in content.values()
        )
    return parsed


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
    Pages that pypdf reads with words moved to the line end are read again in layout mode (#142).
    """
    paperqa_pypdf = importlib.import_module("paperqa_pypdf")
    try:
        return fix_moved_words(path, paperqa_pypdf.parse_pdf_to_pages(path, **kwargs))
    except Exception as exc:
        if not should_fallback_pdf_parse(exc):
            raise
        if not is_crop_error(exc):
            return parse_pdf_with_pdfplumber(path, **kwargs)
    # PaperQA's reader reads media unless told not to, so a missing parse_media means on.
    if kwargs.get("parse_media", True) and not kwargs.get("full_page", False):
        try:
            return fix_moved_words(path, paperqa_pypdf.parse_pdf_to_pages(path, **(kwargs | {"full_page": True})))
        except ValueError as exc:
            if not is_crop_error(exc):
                raise
    return parse_pdf_with_pdfplumber(path, **(kwargs | {"parse_media": False}))

