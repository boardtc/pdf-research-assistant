from pathlib import Path

import pytest

from pdf_research_assistant import scope_change


def _touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_list_pdfs_finds_pdfs_in_subfolders_by_default_in_the_manifest_form(tmp_path):
    _touch(tmp_path, "a.pdf", "B.PDF", "notes.txt", "sub/c.pdf")

    assert scope_change.list_pdfs(tmp_path, recurse=True) == {"a.pdf", "B.PDF", "sub/c.pdf"}


def test_list_pdfs_leaves_subfolders_out_when_not_searching_them(tmp_path):
    _touch(tmp_path, "a.pdf", "sub/c.pdf")

    assert scope_change.list_pdfs(tmp_path, recurse=False) == {"a.pdf"}


def test_list_pdfs_is_empty_for_a_folder_that_does_not_exist(tmp_path):
    assert scope_change.list_pdfs(tmp_path / "missing", recurse=True) == set()


def test_a_folder_named_like_a_pdf_is_not_a_pdf(tmp_path):
    (tmp_path / "folder.pdf").mkdir()

    assert scope_change.list_pdfs(tmp_path, recurse=True) == set()


def test_pdfs_in_scope_are_all_pdfs_without_a_paper_list(tmp_path):
    _touch(tmp_path, "a.pdf", "sub/b.pdf")

    assert scope_change.pdfs_in_scope(tmp_path, set(), recurse=True) == {"a.pdf", "sub/b.pdf"}


def test_pdfs_in_scope_are_only_the_listed_pdfs_that_exist_with_a_paper_list(tmp_path):
    _touch(tmp_path, "a.pdf", "sub/b.pdf")

    scope = scope_change.pdfs_in_scope(tmp_path, {"a.pdf", "missing.pdf"}, recurse=True)

    assert scope == {"a.pdf"}


def test_pdfs_in_scope_leave_out_listed_pdfs_in_subfolders_when_not_searching_them(tmp_path):
    _touch(tmp_path, "a.pdf", "sub/b.pdf")

    assert scope_change.pdfs_in_scope(tmp_path, {"a.pdf", "sub/b.pdf"}, recurse=False) == {"a.pdf"}


def test_preview_counts_pdfs_that_would_be_added_and_removed_from_the_index():
    preview = scope_change.preview_change(
        scope_now={"a.pdf", "b.pdf"}, scope_new={"b.pdf", "c.pdf", "d.pdf"}, indexed={"a.pdf", "b.pdf", "d.pdf"}
    )

    # c.pdf is new and not indexed, d.pdf is new but already in the index, a.pdf leaves the scope while indexed.
    assert preview == {"added": ["c.pdf"], "removed": ["a.pdf"], "already_indexed": ["d.pdf"]}


def test_preview_ignores_index_entries_that_were_already_out_of_scope():
    preview = scope_change.preview_change(scope_now={"a.pdf"}, scope_new={"a.pdf"}, indexed={"a.pdf", "stale.pdf"})

    assert preview == {"added": [], "removed": [], "already_indexed": []}


def test_preview_is_sorted():
    preview = scope_change.preview_change(scope_now=set(), scope_new={"z.pdf", "a.pdf"}, indexed=set())

    assert preview["added"] == ["a.pdf", "z.pdf"]


@pytest.mark.parametrize(
    ("preview", "expected"),
    [
        ({"added": [], "removed": [], "already_indexed": []}, False),
        ({"added": ["a.pdf"], "removed": [], "already_indexed": []}, True),
        ({"added": [], "removed": ["a.pdf"], "already_indexed": []}, True),
        ({"added": [], "removed": [], "already_indexed": ["a.pdf"]}, False),
    ],
)
def test_a_preview_changes_the_index_only_when_it_adds_or_removes_pdfs(preview, expected):
    assert scope_change.changes_index(preview) is expected


def test_describe_preview_gives_the_counts_and_names_a_few_pdfs():
    preview = {"added": ["c.pdf", "d.pdf"], "removed": ["a.pdf"], "already_indexed": ["e.pdf"]}

    lines = scope_change.describe_preview(preview)

    assert lines == [
        "2 PDFs would be added to the index and read, which costs money: `c.pdf`, `d.pdf`",
        "1 PDF would be removed from the index: `a.pdf`",
        "1 PDF coming into scope is already in the index and costs nothing: `e.pdf`",
    ]


def test_describe_preview_shortens_a_long_list_of_names():
    preview = {"added": [f"p{i:02}.pdf" for i in range(12)], "removed": [], "already_indexed": []}

    (line,) = scope_change.describe_preview(preview)

    assert line.startswith("12 PDFs would be added")
    assert "`p04.pdf`" in line
    assert "`p05.pdf`" not in line
    assert line.endswith("and 7 more")


def test_describe_preview_says_nothing_about_a_part_that_is_empty():
    lines = scope_change.describe_preview({"added": [], "removed": ["a.pdf"], "already_indexed": []})

    assert lines == ["1 PDF would be removed from the index: `a.pdf`"]


def test_nothing_is_left_to_search_when_the_paper_list_names_only_pdfs_in_subfolders():
    assert scope_change.leaves_nothing_to_search({"sub/a.pdf"}, use_manifest=True, recurse=False) is True


@pytest.mark.parametrize(
    ("manifest_paths", "use_manifest", "recurse"),
    [
        ({"sub/a.pdf"}, True, True),
        ({"a.pdf", "sub/a.pdf"}, True, False),
        ({"sub/a.pdf"}, False, False),
        (set(), True, False),
    ],
)
def test_something_is_left_to_search_in_every_other_case(manifest_paths, use_manifest, recurse):
    assert scope_change.leaves_nothing_to_search(manifest_paths, use_manifest, recurse) is False


def test_describe_preview_uses_the_plural_for_several_pdfs_already_in_the_index():
    lines = scope_change.describe_preview({"added": [], "removed": [], "already_indexed": ["a.pdf", "b.pdf"]})

    assert lines == ["2 PDFs coming into scope are already in the index and cost nothing: `a.pdf`, `b.pdf`"]


def test_manifest_paths_in_scope_are_all_the_listed_paths_when_the_list_is_used():
    assert scope_change.manifest_paths_in_scope({"a.pdf", "sub/b.pdf"}, "manifest", True) == {"a.pdf", "sub/b.pdf"}


def test_manifest_paths_in_scope_drop_subfolder_paths_when_subfolders_are_not_searched():
    assert scope_change.manifest_paths_in_scope({"a.pdf", "sub/b.pdf"}, "manifest", False) == {"a.pdf"}


def test_manifest_paths_in_scope_are_empty_when_the_list_is_left_out():
    assert scope_change.manifest_paths_in_scope({"a.pdf"}, "all", True) == set()
