"""What changing the manifest scope or the subfolder search does to the index in use.

Neither setting is part of PaperQA's index name, so the index PaperQA reads stays the same. On the next
question PaperQA lists the PDFs in scope and syncs the index to match: it indexes the ones that are missing,
which costs money, and removes the ones that are no longer in scope. This module works out those PDFs from
the folder, the paper list and the index's own file list, so the app can say so before the change is saved.

It reads folders only and does not import PaperQA.
"""

from pathlib import Path, PurePosixPath

from pdf_research_assistant import settings_groups

PDF_SUFFIX = ".pdf"
# Names shown in a warning before it says how many more there are.
NAMES_SHOWN = 5


def normalize(path: str | Path) -> str:
    """Return a relative path with forward slashes, as the paper list and the index list it."""
    return str(PurePosixPath(str(path).replace("\\", "/").strip()))


def manifest_paths_in_scope(manifest_paths: set[str], manifest_scope: str, recurse: bool) -> set[str]:
    """Return the paper list's PDF paths that a scope allows; empty when the list is left out or has no paths.

    With subfolders off, paths inside a subfolder are dropped, because PaperQA never looks at them.
    """
    if manifest_scope != settings_groups.MANIFEST_SCOPE_USE:
        return set()
    return manifest_paths if recurse else {path for path in manifest_paths if "/" not in path}


def list_pdfs(paper_dir: Path, recurse: bool) -> set[str]:
    """Return the PDFs PaperQA would find in the folder, as paths inside it, with or without subfolders."""
    paper_dir = Path(paper_dir)
    if not paper_dir.is_dir():
        return set()
    paths = paper_dir.rglob("*") if recurse else paper_dir.iterdir()
    return {
        normalize(path.relative_to(paper_dir))
        for path in paths
        if path.suffix.lower() == PDF_SUFFIX and path.is_file()
    }


def pdfs_in_scope(paper_dir: Path, manifest_paths: set[str], recurse: bool) -> set[str]:
    """Return the PDFs in scope: those on disk, narrowed to the paper list when it has paths (empty means none)."""
    on_disk = list_pdfs(paper_dir, recurse)
    return on_disk & manifest_paths if manifest_paths else on_disk


def leaves_nothing_to_search(manifest_paths: set[str], use_manifest: bool, recurse: bool) -> bool:
    """Return whether the paper list names PDFs only in subfolders, which would leave none in scope without them."""
    return bool(use_manifest and not recurse and manifest_paths and all("/" in path for path in manifest_paths))


def preview_change(scope_now: set[str], scope_new: set[str], indexed: set[str]) -> dict[str, list[str]]:
    """Return what the next question would do to the index because the scope changes.

    "added" are PDFs entering the scope that are not indexed, so they would be read and embedded. "removed" are
    indexed PDFs leaving the scope. "already_indexed" are PDFs entering the scope that the index already holds.
    Index entries that were already out of scope are not counted, because they go whatever is chosen.
    """
    entering = scope_new - scope_now
    return {
        "added": sorted(entering - indexed),
        "removed": sorted((scope_now & indexed) - scope_new),
        "already_indexed": sorted(entering & indexed),
    }


def changes_index(preview: dict[str, list[str]]) -> bool:
    """Return whether the preview adds or removes any PDF; PDFs the index already holds change nothing."""
    return bool(preview["added"] or preview["removed"])


def _count(number: int) -> str:
    return f"{number} PDF" if number == 1 else f"{number} PDFs"


def _names(paths: list[str]) -> str:
    shown = ", ".join(f"`{path}`" for path in paths[:NAMES_SHOWN])
    hidden = len(paths) - NAMES_SHOWN
    return f"{shown} and {hidden} more" if hidden > 0 else shown


def describe_preview(preview: dict[str, list[str]]) -> list[str]:
    """Return one line of Markdown for each part of the preview that is not empty."""
    lines = []
    if preview["added"]:
        lines.append(
            f"{_count(len(preview['added']))} would be added to the index and read, which costs money: "
            f"{_names(preview['added'])}"
        )
    if preview["removed"]:
        lines.append(f"{_count(len(preview['removed']))} would be removed from the index: {_names(preview['removed'])}")
    if preview["already_indexed"]:
        is_or_are, costs = ("is", "costs") if len(preview["already_indexed"]) == 1 else ("are", "cost")
        lines.append(
            f"{_count(len(preview['already_indexed']))} coming into scope {is_or_are} already in the index and "
            f"{costs} nothing: {_names(preview['already_indexed'])}"
        )
    return lines
