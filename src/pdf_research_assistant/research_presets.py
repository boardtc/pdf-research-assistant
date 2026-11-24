"""Named research presets: saved combinations of the choices made in the sidebar.

A preset is a copy of the choices saved in saved_settings.json and local_llm.json at the time it was
saved, so picking it writes those same choices back and every PaperQA setting built afterwards uses them.
It has up to three parts:

- "answer": the saved answer settings (quality preset, sources, answer length, template, grounding and
  before-answer prompt). Always included. A setting with no saved choice uses PaperQA's default, so picking
  the preset forgets any choice saved for it since.
- "index": the saved library and index settings. Optional, because changing them means indexing the PDFs
  again.
- "model": the local LLM connection, or None for OpenAI. Optional.

Speed limits are left out: they follow the OpenAI account, not the research task.

The presets live in research_presets.json in the project folder, which Git ignores.

Presets can be exported to a presets file and imported from one, to back them up or use them on another
computer. The file holds the same choices, marked with EXPORT_FORMAT and EXPORT_VERSION, and is checked
as a whole before any preset in it is saved. It never holds API keys or folder paths; a local model's
server address is the one detail tied to this computer, and is left out unless asked for.
"""

import json
from pathlib import Path

from pdf_research_assistant import local_llm, saved_settings, settings_groups

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS_PATH = PROJECT_ROOT / "research_presets.json"
MAX_NAME_LENGTH = 60
# A presets file made with Export says what it is, so import can turn down any other JSON file.
EXPORT_FORMAT = "pdf-research-assistant research presets"
EXPORT_VERSION = 1
# Far more than any real presets file, small enough that a wrong file is turned down before it is read.
MAX_IMPORT_BYTES = 1_000_000
NOT_EXPORTED_NOTE = "The file is not a presets file exported from this app."

ANSWER = "answer"
INDEX = "index"
MODEL = "model"
# Settings that follow the account rather than the research task, so presets leave them as they are.
LEFT_OUT = ("rate_profile",)
ANSWER_KEYS = tuple(
    setting.key
    for setting in settings_groups.INSTANT_SETTINGS
    if setting.write is not None and setting.key not in LEFT_OUT
)
INDEX_KEYS = tuple(setting.key for setting in settings_groups.REINDEX_SETTINGS if setting.write is not None)
PART_KEYS = {ANSWER: ANSWER_KEYS, INDEX: INDEX_KEYS}


def _pick(values: dict, keys: tuple[str, ...]) -> dict:
    """Return the values for these keys that can be used, in the order the keys are listed."""
    return {key: values[key] for key in keys if key in values and settings_groups.usable(key, values[key])}


def _clean_model(config: object) -> dict | None:
    """Return a local LLM connection with only its fields, or None for OpenAI; raise ValueError if unusable."""
    if config is None:
        return None
    if local_llm.find_field_problems(config):
        raise ValueError("The local model details cannot be used.")
    return {field: config[field] for field in local_llm.FIELDS}


def make_preset(saved: dict, local_config: dict | None, include_index: bool, include_model: bool) -> dict:
    """Return a preset made from the saved choices: the answer settings, and the index and model if asked."""
    preset = {ANSWER: _pick(saved, ANSWER_KEYS)}
    if include_index:
        preset[INDEX] = _pick(saved, INDEX_KEYS)
    if include_model:
        preset[MODEL] = _clean_model(local_config)
    return preset


def _clean_preset(preset: object) -> dict | None:
    """Return a preset read from the file with anything unusable dropped, or None if it is not a preset."""
    if not isinstance(preset, dict) or not isinstance(preset.get(ANSWER), dict):
        return None
    cleaned = {ANSWER: _pick(preset[ANSWER], ANSWER_KEYS)}
    if isinstance(preset.get(INDEX), dict):
        cleaned[INDEX] = _pick(preset[INDEX], INDEX_KEYS)
    if MODEL in preset:
        try:
            cleaned[MODEL] = _clean_model(preset[MODEL])
        except (ValueError, TypeError):
            # A model part that cannot be used is left out, so picking the preset keeps the model in use.
            pass
    return cleaned


def load(path: Path | None = None) -> dict[str, dict]:
    """Return each saved preset by name, in name order, skipping anything that is not a preset."""
    try:
        saved = json.loads((path or PRESETS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    presets = {}
    for name, preset in saved.items():
        cleaned = _clean_preset(preset)
        if isinstance(name, str) and name.strip() and cleaned is not None:
            presets[name] = cleaned
    return dict(sorted(presets.items(), key=lambda item: item[0].casefold()))


def _write(presets: dict, path: Path | None) -> None:
    (path or PRESETS_PATH).write_text(json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_name(name: str) -> str:
    """Return the name on one line with single spaces; raise ValueError when it is empty or too long."""
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ValueError("Type a name for the preset.")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise ValueError(f"Use at most {MAX_NAME_LENGTH} characters for a preset name.")
    return cleaned


def find_name(presets: dict, name: str) -> str | None:
    """Return the saved name that matches this one, ignoring capitals, or None."""
    return next((saved for saved in presets if saved.casefold() == name.casefold()), None)


def save_preset(name: str, preset: dict, path: Path | None = None) -> tuple[str, bool]:
    """Save a preset under a name, replacing one with the same name; return the name and whether it replaced one.

    Raises ValueError for an unusable name.
    """
    cleaned = clean_name(name)
    presets = load(path)
    replaced = find_name(presets, cleaned)
    if replaced is not None:
        del presets[replaced]
    presets[cleaned] = preset
    _write(presets, path)
    return cleaned, replaced is not None


def rename_preset(old: str, new: str, path: Path | None = None) -> str:
    """Give a preset a new name and return it.

    Raises ValueError for an unusable name, a name another preset has, or a preset that no longer exists.
    """
    cleaned = clean_name(new)
    presets = load(path)
    if old not in presets:
        raise ValueError(f"There is no preset called “{old}” any more.")
    other = find_name(presets, cleaned)
    if other is not None and other != old:
        raise ValueError(f"There is already a preset called “{other}”. Choose another name.")
    presets[cleaned] = presets.pop(old)
    _write(presets, path)
    return cleaned


def delete_preset(name: str, path: Path | None = None) -> bool:
    """Remove a preset; return whether there was one to remove."""
    presets = load(path)
    if name not in presets:
        return False
    del presets[name]
    _write(presets, path)
    return True


def matches(preset: dict, saved: dict, local_config: dict | None) -> bool:
    """Return whether every part the preset includes is what is in use now."""
    for part, keys in PART_KEYS.items():
        if part in preset and preset[part] != _pick(saved, keys):
            return False
    return MODEL not in preset or preset[MODEL] == _clean_model(local_config)


def find_active(presets: dict, saved: dict, local_config: dict | None) -> list[str]:
    """Return the names of the presets that match what is in use now."""
    return [name for name, preset in presets.items() if matches(preset, saved, local_config)]


def changes_index(preset: dict, saved: dict) -> bool:
    """Return whether picking the preset would change the library and index settings in use."""
    return INDEX in preset and preset[INDEX] != _pick(saved, INDEX_KEYS)


def pick_preset(preset: dict, settings_path: Path | None = None, model_path: Path | None = None) -> None:
    """Put a preset's choices in place of the saved ones, for each part it includes.

    Choices the preset leaves out (speed limits, and the index settings or model when not included)
    stay as they are.
    """
    values = saved_settings.load(settings_path)
    for part, keys in PART_KEYS.items():
        if part in preset:
            values = {key: value for key, value in values.items() if key not in keys} | preset[part]
    saved_settings.replace_all(values, settings_path)
    if MODEL in preset:
        if preset[MODEL] is None:
            local_llm.clear_config(model_path)
        else:
            local_llm.save_config(preset[MODEL], model_path)


def export_text(presets: dict, names: list[str], include_model_details: bool) -> str:
    """Return the chosen presets as the text of a presets file, in the order the names are given.

    Without model details, a preset's local model is left out, so the server address on this computer
    never leaves it and the preset keeps whatever model is in use where it is imported. A preset that
    uses OpenAI holds no address, so it keeps its model part.
    """
    chosen = {}
    for name in names:
        if name not in presets:
            continue
        preset = dict(presets[name])
        if not include_model_details and preset.get(MODEL) is not None:
            del preset[MODEL]
        chosen[name] = preset
    content = {"format": EXPORT_FORMAT, "version": EXPORT_VERSION, "presets": chosen}
    return json.dumps(content, indent=2, ensure_ascii=False) + "\n"


def has_model_details(presets: dict, names: list[str]) -> bool:
    """Return whether any of the chosen presets holds a local model, whose address belongs to this computer."""
    return any(presets[name].get(MODEL) is not None for name in names if name in presets)


def _check_part(name: str, part: str, values: dict, problems: list[str]) -> dict:
    """Return a part's values, adding a problem for each one a preset cannot hold or use."""
    keys = PART_KEYS[part]
    for key, value in values.items():
        if key not in keys:
            problems.append(f"“{name}”: “{key}” is not a setting a preset can hold.")
        elif not settings_groups.usable(key, value):
            problems.append(f"“{name}”: the value for {settings_groups.get_setting(key).label} cannot be used.")
    return dict(values)


def _check_imported_preset(name: str, preset: object, problems: list[str]) -> dict:
    """Return a preset from a presets file, adding a problem for anything in it that cannot be used."""
    if not isinstance(preset, dict) or not isinstance(preset.get(ANSWER), dict):
        problems.append(f"“{name}”: it has no answer settings.")
        return {}
    for part in preset:
        if part not in (ANSWER, INDEX, MODEL):
            problems.append(f"“{name}”: “{part}” is not a part a preset can hold.")
    checked = {ANSWER: _check_part(name, ANSWER, preset[ANSWER], problems)}
    if INDEX in preset:
        if isinstance(preset[INDEX], dict):
            checked[INDEX] = _check_part(name, INDEX, preset[INDEX], problems)
        else:
            problems.append(f"“{name}”: its library and index settings are not in the expected format.")
    if MODEL in preset:
        try:
            checked[MODEL] = _clean_model(preset[MODEL])
        except ValueError:
            problems.append(f"“{name}”: its local model details cannot be used.")
    return checked


def _read_presets_file(data: bytes) -> dict:
    """Return the presets a presets file holds by name; raise ValueError when it is not a presets file."""
    if len(data) > MAX_IMPORT_BYTES:
        raise ValueError("The file is too large to be a presets file.")
    try:
        content = json.loads(data.decode("utf-8"))
    except ValueError:
        # UnicodeDecodeError is a ValueError too.
        raise ValueError("The file is not a presets file: it is not JSON text.") from None
    if not isinstance(content, dict) or content.get("format") != EXPORT_FORMAT:
        raise ValueError(NOT_EXPORTED_NOTE)
    version = content.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("The file does not say which version wrote it.")
    if version > EXPORT_VERSION:
        raise ValueError("The file was exported by a newer version of the app. Update the app to import it.")
    presets = content.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError("The file holds no presets.")
    return presets


def check_import(data: bytes) -> tuple[dict[str, dict], list[str]]:
    """Check a presets file before anything in it is saved.

    Returns the presets by tidied name in the file's order and an empty list, or no presets and a
    plain-language problem for everything that cannot be used. A file with any problem is refused as a
    whole, so a preset never arrives with a setting quietly dropped.
    """
    try:
        presets = _read_presets_file(data)
    except ValueError as exc:
        return {}, [str(exc)]
    problems: list[str] = []
    checked: dict[str, dict] = {}
    for typed, preset in presets.items():
        try:
            name = clean_name(typed)
        except ValueError as exc:
            problems.append("A preset has no name." if not typed.strip() else f"“{typed}”: {exc}")
            continue
        if find_name(checked, name) is not None:
            problems.append(f"“{name}”: the file has another preset with this name.")
            continue
        checked[name] = _check_imported_preset(name, preset, problems)
    return ({}, problems) if problems else (checked, [])


def find_clashes(presets: dict, imported: dict) -> list[str]:
    """Return the imported names that a saved preset already has, ignoring capitals."""
    return [name for name in imported if find_name(presets, name) is not None]


def import_presets(
    imported: dict, replace: bool, path: Path | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Save checked presets from a file; return the names added, the names that replaced one, and those skipped.

    A preset with the name of a saved one replaces it only when asked; otherwise it is skipped.
    """
    presets = load(path)
    added, replaced, skipped = [], [], []
    for name, preset in imported.items():
        saved = find_name(presets, name)
        if saved is not None and not replace:
            skipped.append(name)
            continue
        if saved is not None:
            del presets[saved]
            replaced.append(name)
        else:
            added.append(name)
        presets[name] = preset
    if added or replaced:
        _write(dict(sorted(presets.items(), key=lambda item: item[0].casefold())), path)
    return added, replaced, skipped


def describe_model(config: dict | None) -> str:
    """Return the answer model a preset uses, in a few words."""
    if config is None:
        return "OpenAI"
    return f"{config['model']} ({local_llm.PROVIDERS[config['provider']]['label']} at {config['address']})"


def describe_choice(key: str, value: object) -> str:
    """Return one Markdown line for a saved choice, using the name the sidebar shows for it where there is one."""
    setting = settings_groups.get_setting(key)
    check = setting.check
    name = check.name_for(value) if isinstance(check, settings_groups.OneOf) else None
    # An empty before-answer prompt is saved as "" and means it is off.
    text = name or settings_groups.describe_value(value if value != "" else None)
    return settings_groups.describe_line(setting.label, text)


def describe_preset(preset: dict) -> list[str]:
    """Return one Markdown line for each choice the preset holds, and a line for each part it leaves out."""
    lines = [describe_choice(key, value) for key, value in preset[ANSWER].items()]
    if not preset[ANSWER]:
        lines.append("Answer settings: PaperQA's defaults.")
    if INDEX in preset:
        lines += [describe_choice(key, value) for key, value in preset[INDEX].items()]
        if not preset[INDEX]:
            lines.append("Library and index settings: the defaults.")
    else:
        lines.append("Library and index settings: not included, so they stay as they are.")
    if MODEL in preset:
        lines.append(settings_groups.describe_line("Answer model", describe_model(preset[MODEL])))
    else:
        lines.append("Answer model: not included, so it stays as it is.")
    return lines
