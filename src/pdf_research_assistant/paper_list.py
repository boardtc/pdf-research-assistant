"""How PaperQA reads the paper list (`manifest.csv`) when it indexes PDFs (#160).

PaperQA looks each PDF up in the paper list it is given (`agent.index.manifest_file`) by the PDF's path under the
PDF folder as the computer writes it, so on Windows with backslashes. Its own reader keys the rows by
`file_location` exactly as written, which is with forward slashes, so on Windows it found no row for a PDF in a
subfolder and asked the answer model for that PDF's citation. Its reader also opens the file in the computer's own
code page rather than UTF-8.

A row PaperQA does find goes through its DocDetails, which by default throws the row's citation away and builds one
from the row's fields: "Unknown authors. <Title>. Unknown journal, Unknown year." So each row given to PaperQA here
keeps its citation as written, with a document name made from it the way PaperQA makes one from a citation it reads,
and rows without a citation are left out so PaperQA still reads those citations itself.

The scope (which PDFs are indexed) does not come from here: bootstrap.only_manifest decides that.
"""

import csv
import importlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The DocDetails fields PaperQA may still fill in itself: only the document key, from the row's DOI or title. Leaving
# out "citation", "key" and "docname" keeps the row's citation and the document name given here.
FIELDS_TO_OVERWRITE = "doc_id"


def rows_for_paperqa(path: Path, sep: str = os.sep) -> dict[str, dict[str, str]]:
    """Return the paper list's rows that have a citation, keyed by their path written with `sep`, as PaperQA looks
    PDFs up; empty cells are left out so PaperQA uses its own defaults for them."""
    from paperqa.utils import citation_to_docname

    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {key: value.strip() for key, value in raw.items() if key and isinstance(value, str) and value.strip()}
            if not row.get("file_location") or not row.get("citation"):
                continue
            location = row["file_location"].replace("\\", "/").replace("/", sep)
            rows[location] = row | {
                "file_location": location,
                "docname": citation_to_docname(row["citation"]),
                "fields_to_overwrite_from_metadata": FIELDS_TO_OVERWRITE,
            }
    return rows


async def read_for_paperqa(filename: Any = None) -> dict[str, dict[str, str]]:
    """Read the paper list for PaperQA in place of its own maybe_get_manifest, with the same signature and the same
    empty result for no paper list, a missing one or an unreadable one."""
    if not filename:
        return {}
    path = Path(filename)
    try:
        return rows_for_paperqa(path)
    except FileNotFoundError:
        logger.warning(f"Manifest file at {path} could not be found.")
    except Exception:
        logger.exception(f"Error reading manifest file {path}.")
    return {}


def use_with_paperqa() -> None:
    """Make PaperQA read the paper list with read_for_paperqa when it indexes PDFs."""
    importlib.import_module("paperqa.agents.search").maybe_get_manifest = read_for_paperqa
