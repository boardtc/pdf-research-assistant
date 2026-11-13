import io
import json
import urllib.error
from unittest import mock

import pytest

from pdf_research_assistant import local_llm

OLLAMA_CONFIG = {"provider": "ollama", "address": "http://localhost:11434", "model": "llama3.2"}
COMPATIBLE_CONFIG = {"provider": "openai_compatible", "address": "http://localhost:1234/v1", "model": "qwen2.5"}


def fake_server(payload):
    """Stand in for urlopen, answering every request with `payload` as JSON."""
    return mock.patch.object(
        local_llm.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode("utf-8"))
    )


def test_clean_config_trims_spaces_and_the_trailing_slash():
    assert local_llm.clean_config(" ollama ", " http://localhost:11434/ ", " llama3.2 ") == OLLAMA_CONFIG


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (OLLAMA_CONFIG, []),
        (COMPATIBLE_CONFIG, []),
        (OLLAMA_CONFIG | {"provider": "other"}, ["Choose a local server type from the list."]),
        (
            OLLAMA_CONFIG | {"address": "localhost:11434"},
            ["The server address must start with http:// or https://, for example http://localhost:11434."],
        ),
        (
            OLLAMA_CONFIG | {"address": "http://"},
            ["The server address must start with http:// or https://, for example http://localhost:11434."],
        ),
        (OLLAMA_CONFIG | {"model": ""}, ["Enter the name of the model to use."]),
        ({"provider": "ollama"}, 2),
        ("not a dict", ["The saved local LLM settings are not in the expected format."]),
    ],
    ids=["ollama", "compatible", "bad-provider", "no-scheme", "no-host", "no-model", "missing-fields", "not-a-dict"],
)
def test_find_field_problems_explains_each_bad_field(config, expected):
    problems = local_llm.find_field_problems(config)

    if isinstance(expected, int):
        assert len(problems) == expected
    else:
        assert problems == expected


def test_save_then_load_returns_the_same_config(tmp_path):
    path = tmp_path / "local_llm.json"

    local_llm.save_config(OLLAMA_CONFIG, path)

    assert local_llm.load_config(path) == OLLAMA_CONFIG


def test_load_config_is_none_when_nothing_is_saved(tmp_path):
    assert local_llm.load_config(tmp_path / "local_llm.json") is None


@pytest.mark.parametrize("content", ["{not json", json.dumps({"provider": "ollama"}), json.dumps(["a list"])])
def test_load_config_is_none_when_the_saved_file_cannot_be_used(tmp_path, content):
    path = tmp_path / "local_llm.json"
    path.write_text(content, encoding="utf-8")

    assert local_llm.load_config(path) is None


def test_load_config_uses_the_default_path_when_none_is_given(tmp_path):
    path = tmp_path / "local_llm.json"
    with mock.patch.object(local_llm, "CONFIG_PATH", path):
        local_llm.save_config(COMPATIBLE_CONFIG)

        assert local_llm.load_config() == COMPATIBLE_CONFIG
        assert local_llm.clear_config() is True
    assert not path.exists()


def test_clear_config_reports_when_there_was_nothing_to_remove(tmp_path):
    assert local_llm.clear_config(tmp_path / "local_llm.json") is False


def test_list_models_reads_ollama_model_names():
    with fake_server({"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]}) as urlopen:
        names = local_llm.list_models(OLLAMA_CONFIG)

    assert names == ["llama3.2:latest", "qwen2.5:7b"]
    assert urlopen.call_args.args[0] == "http://localhost:11434/api/tags"
    assert urlopen.call_args.kwargs == {"timeout": local_llm.CHECK_TIMEOUT_SECONDS}


def test_list_models_reads_openai_compatible_model_ids():
    with fake_server({"data": [{"id": "qwen2.5"}, {"id": "phi-4"}]}) as urlopen:
        names = local_llm.list_models(COMPATIBLE_CONFIG)

    assert names == ["qwen2.5", "phi-4"]
    assert urlopen.call_args.args[0] == "http://localhost:1234/v1/models"


def test_list_models_says_when_the_server_cannot_be_reached():
    failure = urllib.error.URLError("connection refused")
    with mock.patch.object(local_llm.urllib.request, "urlopen", side_effect=failure):
        with pytest.raises(local_llm.LocalServerError, match="connection refused"):
            local_llm.list_models(OLLAMA_CONFIG)


@pytest.mark.parametrize("payload", ["plain text", {"unexpected": []}, {"models": [{"no_name": 1}]}])
def test_list_models_says_when_the_reply_is_not_a_model_list(payload):
    with fake_server(payload):
        with pytest.raises(local_llm.LocalServerError, match="did not answer like"):
            local_llm.list_models(OLLAMA_CONFIG)


def test_list_models_says_when_the_reply_is_not_json():
    with mock.patch.object(local_llm.urllib.request, "urlopen", return_value=io.BytesIO(b"<html>")):
        with pytest.raises(local_llm.LocalServerError, match="did not answer like"):
            local_llm.list_models(OLLAMA_CONFIG)


@pytest.mark.parametrize(
    ("model", "names", "expected"),
    [
        ("llama3.2", ["llama3.2"], True),
        ("llama3.2", ["llama3.2:latest"], True),
        ("llama3.2:latest", ["llama3.2:latest"], True),
        ("llama3.2", ["llama3.2:1b"], False),
        ("llama3.2", [], False),
    ],
    ids=["exact", "latest-tag-implied", "tag-given", "different-tag", "none-installed"],
)
def test_model_is_listed_accepts_the_default_latest_tag(model, names, expected):
    assert local_llm.model_is_listed(model, names) is expected


def test_check_connection_reports_field_problems_without_calling_the_server():
    with mock.patch.object(local_llm.urllib.request, "urlopen") as urlopen:
        ready, message = local_llm.check_connection(OLLAMA_CONFIG | {"model": ""})

    assert (ready, message) == (False, "Enter the name of the model to use.")
    urlopen.assert_not_called()


def test_check_connection_reports_an_unreachable_server():
    failure = urllib.error.URLError("connection refused")
    with mock.patch.object(local_llm.urllib.request, "urlopen", side_effect=failure):
        ready, message = local_llm.check_connection(OLLAMA_CONFIG)

    assert ready is False
    assert message.startswith("Could not reach Ollama at http://localhost:11434.")
    assert "connection refused" in message


def test_check_connection_lists_installed_models_when_the_chosen_one_is_missing():
    with fake_server({"models": [{"name": "qwen2.5:7b"}, {"name": "phi4:latest"}]}):
        ready, message = local_llm.check_connection(OLLAMA_CONFIG)

    assert ready is False
    assert message == (
        'Connected to Ollama, but the model "llama3.2" is not available there. Available models: qwen2.5:7b, phi4:latest.'
    )


def test_check_connection_says_when_no_model_is_installed():
    with fake_server({"models": []}):
        ready, message = local_llm.check_connection(OLLAMA_CONFIG)

    assert ready is False
    assert message == (
        'Connected to Ollama, but the model "llama3.2" is not available there. No models are installed there yet.'
    )


def test_check_connection_is_ready_when_the_server_has_the_model():
    with fake_server({"models": [{"name": "llama3.2:latest"}]}):
        ready, message = local_llm.check_connection(OLLAMA_CONFIG)

    assert (ready, message) == (True, 'Ready: Ollama at http://localhost:11434 has the model "llama3.2".')


def test_litellm_model_for_ollama_points_litellm_at_the_local_server():
    assert local_llm.litellm_model(OLLAMA_CONFIG) == (
        "ollama/llama3.2",
        {
            "model_list": [
                {
                    "model_name": "ollama/llama3.2",
                    "litellm_params": {"model": "ollama/llama3.2", "api_base": "http://localhost:11434"},
                }
            ]
        },
    )


def test_litellm_model_for_an_openai_compatible_server_adds_a_placeholder_key():
    name, config = local_llm.litellm_model(COMPATIBLE_CONFIG)

    assert name == "openai/qwen2.5"
    assert config["model_list"][0]["litellm_params"] == {
        "model": "openai/qwen2.5",
        "api_base": "http://localhost:1234/v1",
        "api_key": local_llm.PLACEHOLDER_API_KEY,
    }


def test_describe_answer_model_names_the_local_model_or_the_hosted_default():
    assert local_llm.describe_answer_model(OLLAMA_CONFIG, "gpt-4o-mini") == (
        "Answers come from the local model llama3.2 (Ollama at http://localhost:11434)."
    )
    assert local_llm.describe_answer_model(None, "gpt-4o-mini") == "Answers come from OpenAI (gpt-4o-mini)."
