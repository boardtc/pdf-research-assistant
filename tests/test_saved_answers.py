import asyncio
from unittest import mock

import pytest
from paperqa.agents.main import index_search
from paperqa.agents.models import AgentStatus, AnswerResponse
from paperqa.agents.search import SearchIndex
from paperqa.types import PQASession

from pdf_research_assistant import saved_answers


def _response(question="Why is the sky blue?", answer="Rayleigh scattering of light.", cost=0.01):
    session = PQASession(question=question, answer=answer, formatted_answer=f"{answer}\n\nReferences", cost=cost)
    return AnswerResponse(session=session, status=AgentStatus.SUCCESS)


def _record(name="Paper p1"):
    return saved_answers.build_record([{"name": name}], {"index": "pqa_index_x", "values": []}, "2026-11-24T14:05:00+00:00")


def _save(directory, response, record=None):
    """Keep one answer, failing the test with the reason if it could not be kept."""
    assert saved_answers.save_answer(directory, response, record or _record()) is None


@pytest.fixture
def failing_saves(monkeypatch):
    """Make the next given number of PaperQA saves fail the way Windows now and then fails them.

    With after_commit, each failing save commits the answer first and fails afterwards.
    """
    real_save = SearchIndex.save_index
    plan = {"left": 0, "after_commit": False, "tried": 0}

    async def save_index(self):
        plan["tried"] += 1
        if plan["left"] > 0:
            plan["left"] -= 1
            if plan["after_commit"]:
                await real_save(self)
            raise ValueError("An IO error occurred: 'Access is denied. (os error 5)'")
        await real_save(self)

    monkeypatch.setattr(SearchIndex, "save_index", save_index)
    monkeypatch.setattr(saved_answers, "RETRY_PAUSE_SECONDS", 0)
    return plan


def test_build_record_keeps_the_passages_settings_and_time_given():
    record = saved_answers.build_record([{"name": "P"}], {"index": "i"}, "2026-11-24T14:05:00+00:00")

    assert record == {"asked_at": "2026-11-24T14:05:00+00:00", "contexts": [{"name": "P"}], "settings": {"index": "i"}}


def test_build_record_stamps_the_time_now_when_none_is_given():
    record = saved_answers.build_record([], None)

    # A local time with its offset, to the second, such as 2026-11-24T14:05:00+00:00.
    assert record["asked_at"][10] == "T"
    assert record["asked_at"][19] in "+-"


def test_answers_folder_is_paperqas_answers_index_inside_the_index_folder(tmp_path):
    assert saved_answers.answers_folder(tmp_path) == tmp_path / "answers"


def test_a_saved_answer_is_found_again_with_the_apps_record(tmp_path):
    response = _response()

    assert saved_answers.save_answer(tmp_path, response, _record()) is None
    found = saved_answers.search(tmp_path, "sky")

    assert found == {
        "text": "sky",
        "searched": 1,
        "answers": [
            {
                "id": str(response.session.id),
                "question": "Why is the sky blue?",
                "content": "Rayleigh scattering of light.\n\nReferences",
                "cost": 0.01,
                "asked_at": "2026-11-24T14:05:00+00:00",
                "contexts": [{"name": "Paper p1"}],
                "settings": {"index": "pqa_index_x", "values": []},
            }
        ],
    }
    assert (tmp_path / "answers" / "files.zip").exists()


def test_search_looks_through_answers_as_well_as_questions_best_match_first(tmp_path):
    _save(tmp_path, _response("What is photosynthesis?", "Plants turn light into sugar."))
    _save(tmp_path, _response("Why is the sky blue?", "Scattering of light by air."))
    _save(tmp_path, _response("Who wrote it?", "Nobody knows."))

    found = saved_answers.search(tmp_path, "light sugar")

    assert [answer["question"] for answer in found["answers"]] == ["What is photosynthesis?", "Why is the sky blue?"]
    assert found["searched"] == 3


def test_search_finds_nothing_when_no_answer_has_been_kept(tmp_path):
    assert saved_answers.search(tmp_path, "sky") == {"text": "sky", "answers": [], "searched": 0}


def test_search_shows_the_best_matches_only(tmp_path):
    for number in range(3):
        _save(tmp_path, _response(f"Question {number} about light?", f"Answer {number}."))

    assert len(saved_answers.search(tmp_path, "light", top_n=2)["answers"]) == 2


def test_two_answers_with_the_same_text_come_back_once(tmp_path):
    # PaperQA names each stored file after the answer text, so the later answer replaces the earlier one's file.
    _save(tmp_path, _response("First?", "I cannot answer."))
    second = _response("Second?", "I cannot answer.")
    _save(tmp_path, second)

    found = saved_answers.search(tmp_path, "cannot")

    assert [answer["id"] for answer in found["answers"]] == [str(second.session.id)]


def test_search_raises_when_paperqa_cannot_read_the_search(tmp_path):
    _save(tmp_path, _response())

    with pytest.raises(ValueError):
        saved_answers.search(tmp_path, "sky AND")


def test_paperqas_own_search_still_reads_the_answers_kept_here(tmp_path):
    response = _response()
    _save(tmp_path, response)

    results = asyncio.run(index_search("sky", index_name="answers", index_directory=tmp_path))

    assert [(found.session.question, name) for found, name in results] == [
        ("Why is the sky blue?", str(response.session.id))
    ]


def test_answers_kept_by_paperqa_itself_are_found_without_the_apps_record(tmp_path):
    response = _response()

    # PaperQA stores the answer alone. Saving through the app keeps its retry for Windows' failed saves.
    with mock.patch.object(saved_answers, "StoredAnswer", lambda **_data: response):
        _save(tmp_path, response)
    [answer] = saved_answers.search(tmp_path, "sky")["answers"]

    assert (answer["question"], answer["asked_at"], answer["contexts"], answer["settings"]) == (
        "Why is the sky blue?",
        None,
        [],
        None,
    )


def test_a_save_that_fails_is_tried_again_and_the_answer_kept(tmp_path, failing_saves):
    failing_saves["left"] = 1

    _save(tmp_path, _response())

    # At least twice: a real failure on Windows can come on top of the one made here.
    assert failing_saves["tried"] >= 2
    assert saved_answers.search(tmp_path, "sky")["searched"] == 1


def test_a_failed_save_lets_go_of_the_folders_writer_lock(tmp_path, failing_saves, monkeypatch):
    # A writer whose save failed kept the lock, so the next save in the folder waited on it for hours.
    # With one try for the lock, a lock still held fails the next save at once instead.
    monkeypatch.setattr(saved_answers, "LOCK_ATTEMPTS", 1)
    _save(tmp_path, _response("First?", "First answer."))
    failing_saves["left"] = saved_answers.SAVE_ATTEMPTS
    assert saved_answers.save_answer(tmp_path, _response("Second?", "Second answer."), _record()) is not None

    _save(tmp_path, _response("Third?", "Third answer."))

    assert saved_answers.search(tmp_path, "answer")["searched"] == 2


def test_a_save_that_fails_after_committing_keeps_the_answer_once(tmp_path, failing_saves):
    failing_saves.update(left=1, after_commit=True)
    response = _response()

    _save(tmp_path, response)

    assert saved_answers.search(tmp_path, "sky")["searched"] == 1
    # PaperQA's list of kept files names the answer, so PaperQA would not add it again.
    assert str(response.session.id) in asyncio.run(saved_answers.open_index(tmp_path).index_files)


def test_a_save_that_keeps_failing_gives_up_and_says_why(tmp_path, failing_saves):
    failing_saves["left"] = saved_answers.SAVE_ATTEMPTS

    error = saved_answers.save_answer(tmp_path, _response(), _record())

    assert error == "ValueError: An IO error occurred: 'Access is denied. (os error 5)'"
    assert failing_saves["tried"] == saved_answers.SAVE_ATTEMPTS
    assert saved_answers.search(tmp_path, "sky")["searched"] == 0


def test_save_answer_says_why_an_answer_could_not_be_kept_without_raising(tmp_path):
    with mock.patch.object(saved_answers, "add_answer", side_effect=OSError("disk full")):
        assert saved_answers.save_answer(tmp_path, _response(), _record()) == "OSError: disk full"


@pytest.mark.parametrize("stored", [None, "text", {}, {"session": "text"}])
def test_describe_stored_skips_anything_that_is_not_a_paperqa_answer(stored):
    assert saved_answers.describe_stored(stored) is None


def test_describe_stored_falls_back_to_the_plain_answer_and_defaults():
    described = saved_answers.describe_stored({"session": {"answer": "Plain"}, "app": "not a record"})

    assert described == {
        "id": "",
        "question": "",
        "content": "Plain",
        "cost": 0.0,
        "asked_at": None,
        "contexts": [],
        "settings": None,
    }


def test_search_skips_stored_files_that_are_not_answers(tmp_path):
    async def found(*_args):
        return [{"other": 1}, {"session": {"id": "a", "answer": "A"}}], 2

    with mock.patch.object(saved_answers, "find_answers", side_effect=found):
        result = saved_answers.search(tmp_path, "a")

    assert [answer["id"] for answer in result["answers"]] == ["a"]
