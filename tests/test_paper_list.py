import asyncio
import logging
import ntpath
import posixpath
from pathlib import Path

import pytest

from pdf_research_assistant import paper_list

HEADER = "file_location,title,citation,doi\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_rows_are_keyed_with_backslashes_on_windows(tmp_path):
    path = _write(tmp_path, HEADER + "Resources/Sub/a.pdf,A title,Smith (2020) A title,\n")

    rows = paper_list.rows_for_paperqa(path, sep=ntpath.sep)

    assert list(rows) == ["Resources\\Sub\\a.pdf"]
    assert rows["Resources\\Sub\\a.pdf"]["file_location"] == "Resources\\Sub\\a.pdf"


def test_rows_written_with_backslashes_are_keyed_with_forward_slashes_elsewhere(tmp_path):
    path = _write(tmp_path, HEADER + "Resources\\a.pdf,A title,Smith (2020) A title,\n")

    assert list(paper_list.rows_for_paperqa(path, sep=posixpath.sep)) == ["Resources/a.pdf"]


def test_rows_use_the_computers_own_separator_by_default(tmp_path):
    path = _write(tmp_path, HEADER + "Resources/a.pdf,A title,Smith (2020) A title,\n")

    assert list(paper_list.rows_for_paperqa(path)) == [str(Path("Resources/a.pdf"))]


def test_a_row_keeps_its_citation_and_gets_a_document_name_from_it(tmp_path):
    path = _write(tmp_path, HEADER + "a.pdf,A Research Agenda, Bretag (2020) Edward Elgar Publishing ,10.1/x\n")

    row = paper_list.rows_for_paperqa(path)["a.pdf"]

    assert row["citation"] == "Bretag (2020) Edward Elgar Publishing"
    assert row["docname"] == "Bretag2020"
    assert row["title"] == "A Research Agenda"
    assert row["doi"] == "10.1/x"
    assert row["fields_to_overwrite_from_metadata"] == paper_list.FIELDS_TO_OVERWRITE


def test_rows_without_a_citation_or_a_file_are_left_out(tmp_path):
    path = _write(tmp_path, HEADER + "a.pdf,Only a title,,\nb.pdf,Blank citation,   ,\n,No file,Smith (2020),\n")

    assert paper_list.rows_for_paperqa(path) == {}


def test_empty_cells_and_extra_cells_are_left_out_of_a_row(tmp_path):
    path = _write(tmp_path, HEADER + "a.pdf,,Smith (2020) A title,,surplus\n")

    row = paper_list.rows_for_paperqa(path)["a.pdf"]

    assert "title" not in row
    assert "doi" not in row
    assert None not in row


def test_a_byte_order_mark_and_non_ascii_citations_are_read_as_utf8(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text(HEADER + "a.pdf,T,\"Bearman, M. (2024) 'Title', pp. 893–905\",\n", encoding="utf-8-sig")

    assert paper_list.rows_for_paperqa(path)["a.pdf"]["citation"] == "Bearman, M. (2024) 'Title', pp. 893–905"


def test_paperqa_keeps_the_rows_citation_document_name_and_title():
    from paperqa.types import DocDetails

    row = {
        "file_location": "Resources\\a.pdf",
        "title": "A Research Agenda",
        "citation": "Bretag (2020) Edward Elgar Publishing",
        "docname": "Bretag2020",
        "fields_to_overwrite_from_metadata": paper_list.FIELDS_TO_OVERWRITE,
    }

    details = DocDetails(**row).model_dump()

    assert details["citation"] == "Bretag (2020) Edward Elgar Publishing"
    assert details["docname"] == "Bretag2020"
    assert details["title"] == "A Research Agenda"
    assert details["dockey"]
    assert details["dockey"] == details["doc_id"]


def test_read_for_paperqa_gives_paperqa_the_rows_by_the_path_it_looks_up(tmp_path):
    from paperqa.agents.search import fetch_kwargs_from_manifest

    path = _write(tmp_path, HEADER + "Resources/a.pdf,A title,Smith (2020) A title,\n")
    lookup = str(Path("Resources") / "a.pdf")

    manifest = asyncio.run(paper_list.read_for_paperqa(path))
    kwargs = fetch_kwargs_from_manifest(lookup, manifest, str(tmp_path / lookup))

    assert kwargs["citation"] == "Smith (2020) A title"
    assert kwargs["file_location"] == lookup


def test_read_for_paperqa_gives_nothing_without_a_paper_list():
    assert asyncio.run(paper_list.read_for_paperqa(None)) == {}


def test_read_for_paperqa_gives_nothing_for_a_missing_paper_list(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=paper_list.__name__):
        assert asyncio.run(paper_list.read_for_paperqa(tmp_path / "missing.csv")) == {}

    assert "could not be found" in caplog.text


def test_read_for_paperqa_gives_nothing_for_an_unreadable_paper_list(tmp_path, caplog):
    path = tmp_path / "manifest.csv"
    path.write_bytes(HEADER.encode() + b"a.pdf,T,\xff\xfe broken,\n")

    with caplog.at_level(logging.ERROR, logger=paper_list.__name__):
        assert asyncio.run(paper_list.read_for_paperqa(path)) == {}

    assert "Error reading" in caplog.text


def test_use_with_paperqa_makes_paperqa_read_the_paper_list_this_way(monkeypatch):
    import paperqa.agents.search as search

    monkeypatch.setattr(search, "maybe_get_manifest", search.maybe_get_manifest)

    paper_list.use_with_paperqa()

    assert search.maybe_get_manifest is paper_list.read_for_paperqa


def test_build_settings_makes_paperqa_read_the_paper_list_this_way(monkeypatch):
    import paperqa.agents.search as search

    from pdf_research_assistant import bootstrap

    monkeypatch.setattr(search, "maybe_get_manifest", search.maybe_get_manifest)

    bootstrap.build_settings()

    assert search.maybe_get_manifest is paper_list.read_for_paperqa


@pytest.mark.parametrize("sep", [ntpath.sep, posixpath.sep])
def test_a_top_level_pdf_is_keyed_by_its_name_on_every_platform(tmp_path, sep):
    path = _write(tmp_path, HEADER + "a.pdf,A title,Smith (2020) A title,\n")

    assert list(paper_list.rows_for_paperqa(path, sep=sep)) == ["a.pdf"]
