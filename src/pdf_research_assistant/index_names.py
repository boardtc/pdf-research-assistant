"""Friendly names the user gives PaperQA indexes, so an index can be told apart without its folder name.

PaperQA names each index folder after a hash of its index settings (Settings.get_index_name), such as
pqa_index_1a2b.... The app keeps the names in index_names.json beside the index folders, keyed by folder
name, so a name survives a restart and stays with its folder: changing a setting points PaperQA at another
folder, which has its own name or none, and putting the setting back brings the old folder and its name back.
"""

import json
from pathlib import Path

NAMES_FILE = "index_names.json"
MAX_LENGTH = 60


def _names_path(index_root: Path) -> Path:
    return Path(index_root) / NAMES_FILE


def load(index_root: Path) -> dict[str, str]:
    """Return each saved name by index folder name, skipping anything unusable."""
    try:
        saved = json.loads(_names_path(index_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    return {
        folder: name.strip()
        for folder, name in saved.items()
        if isinstance(folder, str) and isinstance(name, str) and name.strip()
    }


def get_name(index_dir: Path) -> str | None:
    """Return the name given to this index folder, or None when it has none."""
    index_dir = Path(index_dir)
    return load(index_dir.parent).get(index_dir.name)


def clean(name: str) -> str:
    """Return the name on one line with single spaces, or raise ValueError when it is too long."""
    cleaned = " ".join(name.split())
    if len(cleaned) > MAX_LENGTH:
        raise ValueError(f"Use at most {MAX_LENGTH} characters for an index name.")
    return cleaned


def save_name(index_dir: Path, name: str) -> None:
    """Give this index folder a name; an empty name removes it.

    Raises ValueError for a name that is too long or that another index already has, ignoring capitals,
    and OSError when the file cannot be written.
    """
    index_dir = Path(index_dir)
    cleaned = clean(name)
    names = load(index_dir.parent)
    for folder, other in names.items():
        if folder != index_dir.name and other.casefold() == cleaned.casefold() and cleaned:
            raise ValueError(f"“{cleaned}” is already the name of the index {folder}. Choose another name.")
    if cleaned:
        names[index_dir.name] = cleaned
    elif index_dir.name in names:
        del names[index_dir.name]
    else:
        return
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    _names_path(index_dir.parent).write_text(json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
