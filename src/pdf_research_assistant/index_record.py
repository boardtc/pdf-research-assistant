"""Which library and index settings the last built index used, so a change to them can be pointed out.

PaperQA names each index after its index settings (settings_groups.REINDEX_SETTINGS), so the
index folder name alone cannot say which setting changed. Whenever the active index holds PDFs,
the app writes its name and those settings to last_index.json, beside the index folders. When a
setting later changes, PaperQA looks for an index under a new name that is still empty, and
find_changes compares the settings in use with the recorded ones.

Putting a setting back brings back the recorded index name, and building the new index records
it instead, so either one ends the warning. Query-only settings are never recorded.
"""

import json
from pathlib import Path

from pdf_research_assistant import settings_groups

RECORD_NAME = "last_index.json"


def _record_path(active_index_dir: Path) -> Path:
    return Path(active_index_dir).parent / RECORD_NAME


def describe_index_settings(settings: object) -> dict[str, str]:
    """Return each library and index setting in use as readable text, by key."""
    return {
        setting.key: settings_groups.describe_value(setting.read(settings))
        for setting in settings_groups.REINDEX_SETTINGS
    }


def load(active_index_dir: Path) -> dict | None:
    """Return the recorded index and its settings, or None when nothing usable was recorded."""
    try:
        record = json.loads(_record_path(active_index_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("index"), str):
        return None
    if not isinstance(record.get("values"), dict):
        return None
    return record


def remember(settings: object, active_index_dir: Path) -> bool:
    """Record the active index and its settings; call only when it holds PDFs. Return whether the file changed."""
    record = {"index": Path(active_index_dir).name, "values": describe_index_settings(settings)}
    if load(active_index_dir) == record:
        return False
    try:
        _record_path(active_index_dir).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def find_changes(settings: object, active_index_dir: Path) -> dict | None:
    """Return how the settings in use differ from the last built index; call only when the active index is empty.

    None means there is nothing to warn about: no index was recorded, or the recorded one is the active one.
    Otherwise "index" names the recorded index and "settings" lists each changed setting with its old and
    new value; the list may be empty if the record predates a setting.
    """
    record = load(active_index_dir)
    if record is None or record["index"] == Path(active_index_dir).name:
        return None
    now = describe_index_settings(settings)
    return {
        "index": record["index"],
        "settings": [
            {"key": setting.key, "label": setting.label, "was": record["values"][setting.key], "now": now[setting.key]}
            for setting in settings_groups.REINDEX_SETTINGS
            if setting.key in record["values"] and record["values"][setting.key] != now[setting.key]
        ],
    }
