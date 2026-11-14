"""Settings chosen in the sidebar: saved by the app, applied whenever PaperQA settings are built.

A question runs in a separate process (query_once.py) that builds its settings afresh, so a
choice made in the sidebar is saved to saved_settings.json in the project folder, which Git
ignores, and bootstrap.build_settings applies it. Any setting in settings_groups that has
`write` and `check` can be saved here, with no change to this module.

A saved value that is unknown or no longer allowed is skipped, so PaperQA's default is used instead.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from pdf_research_assistant import settings_groups

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "saved_settings.json"


def _read_file(path: Path) -> dict:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return values if isinstance(values, dict) else {}


def _write_file(values: dict, path: Path) -> None:
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def _usable(key: str, value: object) -> bool:
    try:
        setting = settings_groups.get_setting(key)
    except KeyError:
        return False
    if setting.write is None:
        return False
    try:
        setting.check(value)
    except ValueError:
        return False
    return True


def load(path: Path | None = None) -> dict:
    """Return the saved values that can still be used, by setting key."""
    return {key: value for key, value in _read_file(path or SETTINGS_PATH).items() if _usable(key, value)}


def save_value(key: str, value: object, path: Path | None = None) -> None:
    """Save one chosen value, keeping the others; raise ValueError if it cannot be used."""
    setting = settings_groups.get_setting(key)
    if setting.write is None:
        raise ValueError(f"The setting {key} cannot be changed from the sidebar.")
    settings_path = path or SETTINGS_PATH
    values = load(settings_path)
    values[key] = setting.check(value)
    _write_file(values, settings_path)


def forget(keys: Iterable[str], path: Path | None = None) -> bool:
    """Remove the saved values for these settings, so PaperQA's defaults apply; return whether any were saved."""
    settings_path = path or SETTINGS_PATH
    values = load(settings_path)
    forgotten = set(keys)
    remaining = {key: value for key, value in values.items() if key not in forgotten}
    if remaining == values:
        return False
    _write_file(remaining, settings_path)
    return True


def apply(settings: object, path: Path | None = None) -> None:
    """Put each saved value onto a PaperQA Settings object."""
    for key, value in load(path).items():
        settings_groups.get_setting(key).write(settings, value)
