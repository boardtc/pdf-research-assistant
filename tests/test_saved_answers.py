import asyncio
import base64
import hashlib
from pathlib import Path
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
    monkeypatch.setattr(saved_answers, "RETRY_PAUSE_SECONDS", 0)
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


def test_a_save_that_fails_again_while_finishing_after_committing_keeps_the_answer_once(tmp_path, failing_saves):
    # Windows can fail the save that finishes a committed answer too; it used to give up on the answer then.
    failing_saves.update(left=2, after_commit=True)
    response = _response()

    _save(tmp_path, response)

    assert failing_saves["tried"] >= 3
    assert saved_answers.search(tmp_path, "sky")["searched"] == 1
    assert str(response.session.id) in asyncio.run(saved_answers.open_index(tmp_path).index_files)


def test_a_save_that_keeps_failing_gives_up_and_says_why(tmp_path, failing_saves, monkeypatch):
    monkeypatch.setattr(saved_answers, "RETRY_PAUSE_SECONDS", 0)
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


PICTURE = b"\x89PNG picture bytes"
PICTURE_TEXT = base64.b64encode(PICTURE).decode("ascii")
PICTURE_NAME = hashlib.sha256(PICTURE).hexdigest()


def _picture_record(*media_lists):
    contexts = [{"name": f"Paper p{n}", "media": media} for n, media in enumerate(media_lists, start=1)]
    return saved_answers.build_record(contexts, {"index": "pqa_index_x", "values": []}, "2026-11-24T14:05:00+00:00")


def test_pictures_folder_is_inside_the_answers_folder(tmp_path):
    assert saved_answers.pictures_folder(tmp_path) == tmp_path / "answers" / "pictures"


def test_keep_pictures_stores_each_distinct_picture_once_and_names_it_in_the_record(tmp_path):
    other = b"another page"
    record = _picture_record(
        [{"page": 1, "image": PICTURE_TEXT, "url": ""}, {"page": 2, "image": base64.b64encode(other).decode()}],
        [{"page": 1, "image": PICTURE_TEXT}],
    )

    kept = saved_answers.keep_pictures(tmp_path, record)

    folder = saved_answers.pictures_folder(tmp_path)
    assert sorted(path.name for path in folder.iterdir()) == sorted([PICTURE_NAME, hashlib.sha256(other).hexdigest()])
    assert (folder / PICTURE_NAME).read_bytes() == PICTURE
    assert kept["contexts"][0]["media"][0] == {"page": 1, "url": "", "picture": PICTURE_NAME}
    assert kept["contexts"][1]["media"][0] == {"page": 1, "picture": PICTURE_NAME}
    assert kept["asked_at"] == record["asked_at"]
    assert kept["settings"] == record["settings"]
    # The record shown under the question keeps its pictures.
    assert record["contexts"][0]["media"][0]["image"] == PICTURE_TEXT


def test_keep_pictures_leaves_media_without_a_readable_picture_as_it_is(tmp_path):
    media = [{"kind": "table", "text": "| a |", "image": ""}, {"url": "https://example.org/f.png"}, {"image": "not base64!"}]
    record = _picture_record(media, [])
    record["contexts"] += [{"name": "No media list"}, {"name": "Empty media list", "media": None}]

    kept = saved_answers.keep_pictures(tmp_path, record)

    assert kept["contexts"] == [
        {"name": "Paper p1", "media": [{"kind": "table", "text": "| a |"}, {"url": "https://example.org/f.png"}, {"image": "not base64!"}]},
        {"name": "Paper p2", "media": []},
        {"name": "No media list"},
        {"name": "Empty media list", "media": []},
    ]
    assert not saved_answers.pictures_folder(tmp_path).exists()


def test_keep_pictures_does_not_write_a_picture_that_is_already_kept(tmp_path):
    folder = saved_answers.pictures_folder(tmp_path)
    folder.mkdir(parents=True)
    (folder / PICTURE_NAME).write_bytes(PICTURE)

    with mock.patch.object(saved_answers.os, "replace") as replace:
        saved_answers.keep_pictures(tmp_path, _picture_record([{"image": PICTURE_TEXT}]))

    replace.assert_not_called()
    assert [path.name for path in folder.iterdir()] == [PICTURE_NAME]


def test_keep_pictures_accepts_a_picture_another_save_kept_at_the_same_moment(tmp_path):
    folder = saved_answers.pictures_folder(tmp_path)

    def other_save_won(_source, target):
        # Another process kept the same picture first and Windows refuses to replace it while it is open.
        Path(target).write_bytes(PICTURE)
        raise PermissionError("Access is denied")

    with mock.patch.object(saved_answers.os, "replace", side_effect=other_save_won):
        kept = saved_answers.keep_pictures(tmp_path, _picture_record([{"image": PICTURE_TEXT}]))

    assert kept["contexts"][0]["media"] == [{"picture": PICTURE_NAME}]
    assert [path.name for path in folder.iterdir()] == [PICTURE_NAME]


def test_keep_pictures_raises_when_a_picture_cannot_be_written_and_leaves_no_part_file(tmp_path):
    with mock.patch.object(saved_answers.os, "replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            saved_answers.keep_pictures(tmp_path, _picture_record([{"image": PICTURE_TEXT}]))

    assert list(saved_answers.pictures_folder(tmp_path).iterdir()) == []


def test_a_saved_answer_keeps_its_pictures_beside_it_and_is_found_with_their_names(tmp_path):
    _save(tmp_path, _response(), _picture_record([{"page": 1, "image": PICTURE_TEXT}]))

    found = saved_answers.search(tmp_path, "sky")

    assert found["answers"][0]["contexts"] == [{"name": "Paper p1", "media": [{"page": 1, "picture": PICTURE_NAME}]}]
    assert saved_answers.load_picture(tmp_path, PICTURE_NAME) == PICTURE
    stored = next((saved_answers.answers_folder(tmp_path) / "docs").iterdir()).read_text(encoding="utf-8")
    assert PICTURE_TEXT not in stored


def test_an_answer_whose_picture_cannot_be_kept_is_not_saved(tmp_path):
    with mock.patch.object(saved_answers.os, "replace", side_effect=OSError("disk full")):
        error = saved_answers.save_answer(tmp_path, _response(), _picture_record([{"image": PICTURE_TEXT}]))

    assert error == "OSError: disk full"
    assert saved_answers.search(tmp_path, "sky")["searched"] == 0


def test_an_answer_saved_with_its_pictures_inline_still_comes_back_with_them(tmp_path):
    # Answers saved before pictures were kept apart carry each picture in the record itself.
    record = _picture_record([{"page": 1, "image": PICTURE_TEXT}])

    with mock.patch.object(saved_answers, "keep_pictures", side_effect=lambda _directory, kept: kept):
        _save(tmp_path, _response(), record)

    assert saved_answers.search(tmp_path, "sky")["answers"][0]["contexts"][0]["media"] == [{"page": 1, "image": PICTURE_TEXT}]


@pytest.mark.parametrize("name", ["../secret", "a" * 63, "A" * 64, "", None])
def test_load_picture_refuses_a_name_that_is_not_a_kept_picture(tmp_path, name):
    with pytest.raises(ValueError):
        saved_answers.load_picture(tmp_path, name)


def test_load_picture_raises_when_the_picture_is_gone(tmp_path):
    with pytest.raises(FileNotFoundError):
        saved_answers.load_picture(tmp_path, PICTURE_NAME)


def _stored_files(directory):
    docs = saved_answers.answers_folder(directory) / "docs"
    return sorted(path.name for path in docs.iterdir()) if docs.exists() else []


def _kept_pictures(directory):
    return sorted(path.name for path in saved_answers.pictures_folder(directory).iterdir())


OTHER_PICTURE = b"another picture"
OTHER_PICTURE_NAME = hashlib.sha256(OTHER_PICTURE).hexdigest()


def test_deleting_an_answer_takes_it_out_of_paperqas_index_and_leaves_the_others(tmp_path):
    gone = _response("Why is the sky blue?", "Scattering of light.")
    kept = _response("Why is grass green?", "Chlorophyll reflects green light.")
    _save(tmp_path, gone)
    _save(tmp_path, kept)
    files_before = _stored_files(tmp_path)

    assert saved_answers.delete_answer(tmp_path, str(gone.session.id)) is None

    found = saved_answers.search(tmp_path, "light")
    assert [answer["id"] for answer in found["answers"]] == [str(kept.session.id)]
    assert found["searched"] == 1
    results = asyncio.run(index_search("light", index_name="answers", index_directory=tmp_path))
    assert [name for _found, name in results] == [str(kept.session.id)]
    assert str(gone.session.id) not in asyncio.run(saved_answers.open_index(tmp_path).index_files)
    assert len(_stored_files(tmp_path)) == len(files_before) - 1


def test_paperqas_own_delete_leaves_an_answer_counted_which_is_why_the_app_deletes_by_phrase(tmp_path):
    # If this starts failing, PaperQA's remove_from_index works on its own and delete_from_search can go.
    response = _response()
    _save(tmp_path, response)
    _save(tmp_path, _response("Other?", "Another answer."))
    index = saved_answers.open_index(tmp_path)

    async def remove():
        await index.remove_from_index(str(response.session.id))
        await index.save_index()

    asyncio.run(remove())

    assert asyncio.run(saved_answers.open_index(tmp_path).count) == 2


def test_paperqas_own_save_cannot_save_an_empty_list_which_is_why_the_app_saves_it_itself(tmp_path):
    # If this starts failing, save_empty_list can go.
    response = _response()
    _save(tmp_path, response)
    index = saved_answers.open_index(tmp_path)

    async def remove():
        await index.remove_from_index(str(response.session.id))
        await index.save_index()

    with pytest.raises(Exception, match="decompressing"):
        asyncio.run(remove())


def test_deleting_the_last_answer_leaves_an_empty_index_that_can_be_searched_and_saved_to(tmp_path):
    response = _response()
    _save(tmp_path, response)

    assert saved_answers.delete_answer(tmp_path, str(response.session.id)) is None

    assert saved_answers.search(tmp_path, "sky") == {"text": "sky", "answers": [], "searched": 0}
    assert _stored_files(tmp_path) == []
    _save(tmp_path, _response("Again?", "Sky again."))
    assert saved_answers.search(tmp_path, "sky")["searched"] == 1


def test_deleting_an_answer_kept_by_paperqa_itself_works_too(tmp_path):
    response = _response()
    with mock.patch.object(saved_answers, "StoredAnswer", lambda **_data: response):
        _save(tmp_path, response)

    assert saved_answers.delete_answer(tmp_path, str(response.session.id)) is None

    assert saved_answers.search(tmp_path, "sky")["searched"] == 0


def test_deleting_an_answer_that_is_no_longer_kept_changes_nothing(tmp_path):
    response = _response()
    _save(tmp_path, response)

    assert saved_answers.delete_answer(tmp_path, "not-a-kept-answer") is None

    assert saved_answers.search(tmp_path, "sky")["searched"] == 1


def test_deleting_one_of_two_answers_with_the_same_text_deletes_both(tmp_path):
    # They share one stored file, so the other could not be shown once that file is gone.
    first = _response("First?", "I cannot answer.")
    second = _response("Second?", "I cannot answer.")
    other = _response("Other?", "Cannot say for sure.")
    for response in (first, second, other):
        _save(tmp_path, response)

    assert saved_answers.delete_answer(tmp_path, str(second.session.id)) is None

    found = saved_answers.search(tmp_path, "cannot")
    assert [answer["id"] for answer in found["answers"]] == [str(other.session.id)]
    assert found["searched"] == 1
    index_files = asyncio.run(saved_answers.open_index(tmp_path).index_files)
    assert set(index_files) == {str(other.session.id)}


def test_count_sharing_counts_the_other_answers_with_the_same_text(tmp_path):
    first = _response("First?", "I cannot answer.")
    second = _response("Second?", "I cannot answer.")
    alone = _response("Alone?", "Something else.")
    for response in (first, second, alone):
        _save(tmp_path, response)

    assert saved_answers.count_sharing(tmp_path, str(first.session.id)) == 1
    assert saved_answers.count_sharing(tmp_path, str(second.session.id)) == 1
    assert saved_answers.count_sharing(tmp_path, str(alone.session.id)) == 0
    assert saved_answers.count_sharing(tmp_path, "not-a-kept-answer") == 0


def test_count_sharing_is_nought_when_no_answer_has_been_kept(tmp_path):
    assert saved_answers.count_sharing(tmp_path, "a") == 0


def test_deleting_an_answer_removes_its_own_pictures_and_keeps_those_another_answer_shows(tmp_path):
    other_text = base64.b64encode(OTHER_PICTURE).decode("ascii")
    gone = _response("Why is the sky blue?", "Scattering of light.")
    kept = _response("Why is grass green?", "Chlorophyll.")
    _save(tmp_path, gone, _picture_record([{"page": 1, "image": PICTURE_TEXT}, {"page": 2, "image": other_text}]))
    _save(tmp_path, kept, _picture_record([{"page": 2, "image": other_text}]))

    assert saved_answers.delete_answer(tmp_path, str(gone.session.id)) is None

    assert _kept_pictures(tmp_path) == [OTHER_PICTURE_NAME]
    assert saved_answers.load_picture(tmp_path, OTHER_PICTURE_NAME) == OTHER_PICTURE


def test_deleting_an_answer_keeps_every_picture_when_another_answer_cannot_be_read(tmp_path):
    # Which pictures the other answer shows cannot be told, so none is removed rather than one it shows.
    gone = _response("Why is the sky blue?", "Scattering of light.")
    unreadable = _response("Why is grass green?", "Chlorophyll.")
    _save(tmp_path, gone, _picture_record([{"page": 1, "image": PICTURE_TEXT}]))
    _save(tmp_path, unreadable)
    real_read = SearchIndex.get_saved_object

    async def read(self, file_location, keep_filenames=False):
        if file_location == str(unreadable.session.id):
            raise OSError("cannot read")
        return await real_read(self, file_location, keep_filenames)

    with mock.patch.object(SearchIndex, "get_saved_object", read):
        assert saved_answers.delete_answer(tmp_path, str(gone.session.id)) is None

    assert _kept_pictures(tmp_path) == [PICTURE_NAME]
    assert asyncio.run(saved_answers.open_index(tmp_path).count) == 1


def test_deleting_an_answer_whose_stored_file_is_gone_still_takes_it_out_of_the_index(tmp_path):
    response = _response()
    _save(tmp_path, response, _picture_record([{"page": 1, "image": PICTURE_TEXT}]))
    for path in (saved_answers.answers_folder(tmp_path) / "docs").iterdir():
        path.unlink()

    assert saved_answers.delete_answer(tmp_path, str(response.session.id)) is None

    assert asyncio.run(saved_answers.open_index(tmp_path).count) == 0
    # With no record to name them, its pictures are left where they are.
    assert _kept_pictures(tmp_path) == [PICTURE_NAME]


def test_a_picture_that_cannot_be_removed_is_left_and_the_answer_still_deleted(tmp_path):
    response = _response()
    _save(tmp_path, response, _picture_record([{"page": 1, "image": PICTURE_TEXT}]))

    with mock.patch.object(saved_answers, "remove_picture", side_effect=PermissionError("in use")):
        assert saved_answers.delete_answer(tmp_path, str(response.session.id)) is None

    assert saved_answers.search(tmp_path, "sky")["searched"] == 0
    assert _kept_pictures(tmp_path) == [PICTURE_NAME]


def test_remove_picture_deletes_the_named_picture_and_ignores_one_already_gone(tmp_path):
    saved_answers.write_picture(saved_answers.pictures_folder(tmp_path), PICTURE)

    saved_answers.remove_picture(tmp_path, PICTURE_NAME)
    saved_answers.remove_picture(tmp_path, PICTURE_NAME)

    assert _kept_pictures(tmp_path) == []


def test_a_delete_that_fails_says_why_and_leaves_the_answer_in_place(tmp_path):
    response = _response()
    _save(tmp_path, response, _picture_record([{"page": 1, "image": PICTURE_TEXT}]))
    files_before = _stored_files(tmp_path)
    denied = ValueError("An IO error occurred: 'Access is denied. (os error 5)'")

    with mock.patch.object(saved_answers, "delete_from_search", side_effect=denied):
        error = saved_answers.delete_answer(tmp_path, str(response.session.id))

    assert error == "ValueError: An IO error occurred: 'Access is denied. (os error 5)'"
    assert [answer["id"] for answer in saved_answers.search(tmp_path, "sky")["answers"]] == [str(response.session.id)]
    assert _stored_files(tmp_path) == files_before
    assert _kept_pictures(tmp_path) == [PICTURE_NAME]


def test_a_delete_that_fails_while_saving_lets_go_of_the_writer_lock(tmp_path, failing_saves, monkeypatch):
    # A writer left holding the lock would make every later save or delete in the folder wait for it.
    monkeypatch.setattr(saved_answers, "LOCK_ATTEMPTS", 1)
    response = _response()
    _save(tmp_path, response)
    failing_saves["left"] = 1

    assert saved_answers.delete_answer(tmp_path, str(response.session.id)) is not None

    assert saved_answers.delete_answer(tmp_path, str(response.session.id)) is None
    assert saved_answers.search(tmp_path, "sky")["searched"] == 0


def test_a_delete_waits_only_a_little_for_a_writer_lock_another_save_holds(tmp_path, monkeypatch):
    # PaperQA's own delete tries a thousand times, each waiting up to a minute, so it could wait for hours.
    monkeypatch.setattr(saved_answers, "LOCK_ATTEMPTS", 1)
    response = _response()
    _save(tmp_path, response)
    holder = saved_answers.open_index(tmp_path)

    async def hold():
        async with holder.writer():
            pass

    asyncio.run(hold())
    try:
        error = saved_answers.delete_answer(tmp_path, str(response.session.id))
    finally:
        saved_answers.release_writer(holder)

    assert error is not None
    assert "lock" in error.lower()
    assert saved_answers.search(tmp_path, "sky")["searched"] == 1


@pytest.mark.parametrize(
    "stored",
    [
        None,
        {},
        {"app": "text"},
        {"app": {"contexts": "text"}},
        {"app": {"contexts": ["text", {"media": "text"}, {"media": ["text", {"picture": "../secret"}, {"image": "x"}]}]}},
    ],
)
def test_picture_names_finds_none_in_anything_without_kept_pictures(stored):
    assert saved_answers.picture_names(stored) == set()


def test_picture_names_lists_each_kept_picture_an_answer_shows():
    stored = {"app": {"contexts": [{"media": [{"picture": PICTURE_NAME}]}, {"media": [{"picture": OTHER_PICTURE_NAME}]}]}}

    assert saved_answers.picture_names(stored) == {PICTURE_NAME, OTHER_PICTURE_NAME}
