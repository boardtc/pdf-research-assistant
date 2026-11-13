"""Local LLM connection settings: saved by the app, read when PaperQA settings are built.

The settings live in local_llm.json in the project folder, which Git ignores. When that
file is missing or unusable, the app uses its hosted OpenAI model. Only the model that
reads passages and writes answers changes; embeddings and the index stay as they are.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "local_llm.json"
CHECK_TIMEOUT_SECONDS = 5
# OpenAI-compatible servers ignore the key, but the client library refuses to send a request without one.
PLACEHOLDER_API_KEY = "not-needed"

OLLAMA = "ollama"
OPENAI_COMPATIBLE = "openai_compatible"
PROVIDERS = {
    OLLAMA: {
        "label": "Ollama",
        "default_address": "http://localhost:11434",
        "models_path": "/api/tags",
        "models_key": "models",
        "name_key": "name",
        "litellm_prefix": "ollama/",
    },
    OPENAI_COMPATIBLE: {
        "label": "OpenAI-compatible server",
        "default_address": "http://localhost:1234/v1",
        "models_path": "/models",
        "models_key": "data",
        "name_key": "id",
        "litellm_prefix": "openai/",
    },
}
FIELDS = ("provider", "address", "model")


class LocalServerError(Exception):
    """The local model server could not be reached or did not answer as expected."""


def clean_config(provider: str, address: str, model: str) -> dict:
    """Return the settings as typed, without stray spaces or a trailing slash on the address."""
    return {"provider": provider.strip(), "address": address.strip().rstrip("/"), "model": model.strip()}


def find_field_problems(config: object) -> list[str]:
    """Return a plain-language problem for each setting that cannot work, or an empty list."""
    if not isinstance(config, dict):
        return ["The saved local LLM settings are not in the expected format."]
    problems = []
    if config.get("provider") not in PROVIDERS:
        problems.append("Choose a local server type from the list.")
    address = urlparse(str(config.get("address") or ""))
    if address.scheme not in ("http", "https") or not address.netloc:
        problems.append(
            "The server address must start with http:// or https://, for example http://localhost:11434."
        )
    if not str(config.get("model") or "").strip():
        problems.append("Enter the name of the model to use.")
    return problems


def load_config(path: Path | None = None) -> dict | None:
    """Return the saved local LLM settings, or None when there are none that can be used."""
    config_path = path or CONFIG_PATH
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if find_field_problems(config):
        return None
    return {field: config[field] for field in FIELDS}


def save_config(config: dict, path: Path | None = None) -> None:
    (path or CONFIG_PATH).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def clear_config(path: Path | None = None) -> bool:
    """Remove the saved settings so the app uses its hosted model; return whether any were saved."""
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return False
    config_path.unlink()
    return True


def list_models(config: dict) -> list[str]:
    """Ask the local server which models it has."""
    provider = PROVIDERS[config["provider"]]
    url = config["address"] + provider["models_path"]
    try:
        with urllib.request.urlopen(url, timeout=CHECK_TIMEOUT_SECONDS) as response:
            body = response.read()
    except (OSError, ValueError) as exc:
        reason = getattr(exc, "reason", None) or exc
        raise LocalServerError(str(reason)) from exc
    try:
        return [str(item[provider["name_key"]]) for item in json.loads(body)[provider["models_key"]]]
    except (ValueError, KeyError, TypeError) as exc:
        raise LocalServerError(f"{url} did not answer like a model server.") from exc


def model_is_listed(model: str, names: list[str]) -> bool:
    """Return whether the server has the model; Ollama lists "name:latest" for a model typed as "name"."""
    return model in names or f"{model}:latest" in names


def check_connection(config: dict) -> tuple[bool, str]:
    """Check the settings against the running server and say, in plain language, whether they are ready."""
    problems = find_field_problems(config)
    if problems:
        return False, " ".join(problems)
    label = PROVIDERS[config["provider"]]["label"]
    try:
        names = list_models(config)
    except LocalServerError as exc:
        return False, (
            f"Could not reach {label} at {config['address']}. "
            f"Check that it is running and that the address is right. Details: {exc}"
        )
    if not model_is_listed(config["model"], names):
        available = f"Available models: {', '.join(names)}." if names else "No models are installed there yet."
        return False, f'Connected to {label}, but the model "{config["model"]}" is not available there. {available}'
    return True, f'Ready: {label} at {config["address"]} has the model "{config["model"]}".'


def litellm_model(config: dict) -> tuple[str, dict]:
    """Return the model name and model config PaperQA needs to send requests to the local server."""
    name = PROVIDERS[config["provider"]]["litellm_prefix"] + config["model"]
    params = {"model": name, "api_base": config["address"]}
    if config["provider"] == OPENAI_COMPATIBLE:
        params["api_key"] = PLACEHOLDER_API_KEY
    return name, {"model_list": [{"model_name": name, "litellm_params": params}]}


def describe_answer_model(config: dict | None, hosted_model: str) -> str:
    """Say in one line which model currently writes the answers."""
    if config is None:
        return f"Answers come from OpenAI ({hosted_model})."
    label = PROVIDERS[config["provider"]]["label"]
    return f"Answers come from the local model {config['model']} ({label} at {config['address']})."
