import json
import pickle
import sys
import zlib
from pathlib import Path
from unittest import mock

import pytest

from pdf_research_assistant import rebuild_client


@pytest.fixture(autouse=True)
def no_rebuild_left_over():
    """Each test starts and ends with no rebuild running and no result waiting."""
    rebuild_client._state.update(run=None, result=None)
    yield
    rebuild_client._state.update(run=None, result=None)


def _process(returncode=None):
    process = mock.Mock()
    process.poll.return_value = returncode
    return process


def _start(tmp_path, process=None, pdf_count=3):
    with mock.patch("subprocess.Popen", return_value=process or _process()) as popen:
        run = rebuild_client.start(tmp_path / "index" / "pqa_index_abc", pdf_count)
    return run, popen


def test_count_pdfs_counts_pdfs_in_the_folder_and_its_subfolders(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.pdf", "b.PDF", "notes.txt", "sub/c.pdf"):
        (tmp_path / name).write_bytes(b"")

    assert rebuild_client.count_pdfs(tmp_path) == 3


def test_count_pdfs_leaves_out_subfolders_when_they_are_not_searched(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.pdf", "b.PDF", "notes.txt", "sub/c.pdf"):
        (tmp_path / name).write_bytes(b"")

    assert rebuild_client.count_pdfs(tmp_path, recurse=False) == 2


def test_count_pdfs_is_zero_for_a_folder_that_does_not_exist(tmp_path):
    assert rebuild_client.count_pdfs(tmp_path / "missing") == 0


def test_log_path_sits_beside_the_index_folders(tmp_path):
    assert rebuild_client.log_path(tmp_path / "index" / "pqa_index_abc") == tmp_path / "index" / "rebuild.log"


def test_start_runs_the_rebuild_script_in_its_own_process_and_logs_its_output(tmp_path):
    process = _process()

    run, popen = _start(tmp_path, process)

    args, kwargs = popen.call_args
    assert args[0] == [sys.executable, "-u", str(rebuild_client.HELPER)]
    assert rebuild_client.HELPER.name == "rebuild.py"
    assert rebuild_client.HELPER.exists()
    assert Path(kwargs["stdout"].name) == tmp_path / "index" / "rebuild.log"
    assert kwargs["stdout"].closed
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert run["process"] is process
    assert run["index_dir"] == tmp_path / "index" / "pqa_index_abc"
    assert run["log_path"] == tmp_path / "index" / "rebuild.log"
    assert run["pdf_count"] == 3
    assert run["done"] == 0
    assert rebuild_client.get_run() is run


def test_start_creates_the_index_folder_for_the_log_when_it_is_missing(tmp_path):
    _start(tmp_path)

    assert (tmp_path / "index").is_dir()


def test_start_does_not_start_a_second_rebuild_while_one_is_running(tmp_path):
    first, _ = _start(tmp_path)

    second, popen = _start(tmp_path)

    popen.assert_not_called()
    assert second is first


def test_start_forgets_the_result_of_an_earlier_rebuild(tmp_path):
    rebuild_client._state["result"] = {"ok": True}

    _start(tmp_path)

    assert rebuild_client.take_result() is None


def test_get_run_is_none_when_no_rebuild_is_running():
    assert rebuild_client.get_run() is None


def test_note_progress_keeps_the_highest_count_seen():
    run = {"done": 0, "pdf_count": 5}

    assert rebuild_client.note_progress(run, 2) == 2
    # The index file can be unreadable for a moment while the rebuild is writing it.
    assert rebuild_client.note_progress(run, 0) == 2
    assert run["done"] == 2


def test_note_progress_never_goes_past_the_number_of_pdfs():
    assert rebuild_client.note_progress({"done": 0, "pdf_count": 5}, 9) == 5


def test_describe_progress_gives_the_count_and_the_start_time(tmp_path):
    run, _ = _start(tmp_path)
    run["started"] = run["started"].replace(hour=9, minute=5)

    assert rebuild_client.describe_progress(run, 1) == "1 of 3 PDFs done · started at 09:05"


def test_progress_fraction_is_the_share_of_pdfs_done():
    assert rebuild_client.progress_fraction({"pdf_count": 4}, 1) == 0.25


def test_progress_fraction_is_zero_when_there_are_no_pdfs():
    assert rebuild_client.progress_fraction({"pdf_count": 0}, 0) == 0.0


def test_collect_result_is_none_when_no_rebuild_is_running():
    assert rebuild_client.collect_result() is None


def test_collect_result_is_none_while_the_rebuild_is_still_running(tmp_path):
    run, _ = _start(tmp_path)

    assert rebuild_client.collect_result() is None
    assert rebuild_client.get_run() is run


def test_collect_result_ends_a_finished_rebuild_and_keeps_its_last_log_lines(tmp_path):
    run, _ = _start(tmp_path, _process(returncode=0))
    run["log_path"].write_text("Indexed after run: 3\nFailed PDFs: 0\n", encoding="utf-8")

    result = rebuild_client.collect_result()

    assert result == {
        "ok": True,
        "stopped": False,
        "log_path": str(run["log_path"]),
        "lines": ["Indexed after run: 3", "Failed PDFs: 0"],
        "index_dir": run["index_dir"],
        "index": {"usable": True, "indexed": 0},
        "cost": None,
        "pdf_count": 3,
    }
    assert rebuild_client.get_run() is None


def test_collect_result_reports_a_rebuild_that_stopped_with_an_error(tmp_path):
    _start(tmp_path, _process(returncode=1))

    assert rebuild_client.collect_result()["ok"] is False


def test_take_result_hands_the_result_over_once(tmp_path):
    _start(tmp_path, _process(returncode=0))
    result = rebuild_client.collect_result()

    assert rebuild_client.take_result() == result
    assert rebuild_client.take_result() is None


def test_read_log_tail_returns_only_the_last_lines(tmp_path):
    log = tmp_path / "rebuild.log"
    log.write_text("\n".join(f"line {number}" for number in range(100)), encoding="utf-8")

    lines = rebuild_client.read_log_tail(log)

    assert len(lines) == rebuild_client.LOG_TAIL_LINES
    assert lines[-1] == "line 99"


def test_read_log_tail_is_empty_when_there_is_no_log(tmp_path):
    assert rebuild_client.read_log_tail(tmp_path / "missing.log") == []


# Stopping a rebuild


def _write_files_zip(index_dir, payload):
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "files.zip").write_bytes(zlib.compress(pickle.dumps(payload)))


def test_stop_path_sits_beside_the_log(tmp_path):
    assert rebuild_client.stop_path(tmp_path / "index" / "pqa_index_abc") == tmp_path / "index" / "rebuild.stop"


def test_start_tells_the_rebuild_where_to_look_for_a_stop_request(tmp_path):
    _, popen = _start(tmp_path)

    env = popen.call_args.kwargs["env"]
    assert env[rebuild_client.STOP_ENV] == str(tmp_path / "index" / "rebuild.stop")
    assert rebuild_client.STOP_ENV == "PDF_RESEARCH_STOP_FILE"


def test_start_clears_a_stop_request_left_by_an_earlier_rebuild(tmp_path):
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "rebuild.stop").write_text("stop", encoding="utf-8")

    _start(tmp_path)

    assert not (tmp_path / "index" / "rebuild.stop").exists()


def test_request_stop_writes_the_stop_file_and_marks_the_run_as_stopping(tmp_path):
    run, _ = _start(tmp_path)

    rebuild_client.request_stop(run)

    assert (tmp_path / "index" / "rebuild.stop").exists()
    assert rebuild_client.is_stopping(run) is True


def test_request_stop_twice_changes_nothing(tmp_path):
    run, _ = _start(tmp_path)

    rebuild_client.request_stop(run)
    rebuild_client.request_stop(run)

    assert rebuild_client.is_stopping(run) is True


def test_a_run_is_not_stopping_until_a_stop_is_asked_for(tmp_path):
    run, _ = _start(tmp_path)

    assert rebuild_client.is_stopping(run) is False


def test_a_stop_asked_for_in_another_browser_session_shows_in_this_one(tmp_path):
    run, _ = _start(tmp_path)
    rebuild_client.request_stop(run)

    # The run is one dictionary in the server process, so every session sees the flag.
    assert rebuild_client.get_run()["stopping"] is True


def test_collect_result_reports_a_stopped_rebuild_and_clears_the_stop_file(tmp_path):
    run, _ = _start(tmp_path, _process(returncode=rebuild_client.STOPPED_EXIT_CODE))
    rebuild_client.request_stop(run)

    result = rebuild_client.collect_result()

    assert result["stopped"] is True
    assert result["ok"] is False
    assert not (tmp_path / "index" / "rebuild.stop").exists()


def test_collect_result_does_not_call_a_rebuild_that_finished_first_stopped(tmp_path):
    run, _ = _start(tmp_path, _process(returncode=0))
    rebuild_client.request_stop(run)

    result = rebuild_client.collect_result()

    assert result["stopped"] is False
    assert result["ok"] is True


def test_collect_result_says_whether_the_index_can_be_used_and_how_many_pdfs_it_holds(tmp_path):
    run, _ = _start(tmp_path, _process(returncode=rebuild_client.STOPPED_EXIT_CODE))
    _write_files_zip(run["index_dir"], {"a.pdf": "h1", "b.pdf": "h2", "c.pdf": "ERROR"})

    assert rebuild_client.collect_result()["index"] == {"usable": True, "indexed": 2}


def test_collect_result_says_how_many_pdfs_the_rebuild_set_out_to_index(tmp_path):
    _start(tmp_path, _process(returncode=rebuild_client.FAILED_EXIT_CODE), pdf_count=120)

    result = rebuild_client.collect_result()

    assert result["pdf_count"] == 120
    assert result["ok"] is False
    assert result["stopped"] is False


def test_read_index_state_is_an_empty_but_undamaged_index_when_nothing_was_saved_yet(tmp_path):
    assert rebuild_client.read_index_state(tmp_path / "missing") == {"usable": True, "indexed": 0}


def test_read_index_state_is_unusable_when_the_file_list_is_damaged(tmp_path):
    (tmp_path / "files.zip").write_bytes(b"half a write")

    assert rebuild_client.read_index_state(tmp_path) == {"usable": False, "indexed": 0}


def test_read_index_state_is_unusable_when_the_file_list_is_not_a_dictionary(tmp_path):
    (tmp_path / "files.zip").write_bytes(zlib.compress(pickle.dumps(["a.pdf"])))

    assert rebuild_client.read_index_state(tmp_path) == {"usable": False, "indexed": 0}


def test_read_index_state_counts_only_pdfs_that_did_not_fail(tmp_path):
    _write_files_zip(tmp_path, {"a.pdf": "h1", "b.pdf": rebuild_client.FAILED_DOCUMENT_ADD_ID})

    assert rebuild_client.read_index_state(tmp_path) == {"usable": True, "indexed": 1}


def test_the_failed_marker_matches_the_one_paperqa_writes():
    from paperqa.agents.search import FAILED_DOCUMENT_ADD_ID

    assert rebuild_client.FAILED_DOCUMENT_ADD_ID == FAILED_DOCUMENT_ADD_ID


# The real cost of a rebuild (#153).


def test_start_tells_the_rebuild_where_to_write_its_cost_and_clears_an_old_one(tmp_path):
    old = rebuild_client.cost_path(tmp_path / "index" / "pqa_index_abc")
    old.parent.mkdir(parents=True)
    old.write_text("{}", encoding="utf-8")

    _, popen = _start(tmp_path)

    assert popen.call_args.kwargs["env"][rebuild_client.COST_ENV] == str(tmp_path / "index" / "rebuild.cost.json")
    assert not old.exists()


def test_collect_result_reads_what_the_rebuild_cost(tmp_path):
    run, _ = _start(tmp_path, _process(returncode=0))
    costs = {"embeddings": 0.01, "enrichment": 0.2, "citations": 0.003}
    rebuild_client.cost_path(run["index_dir"]).write_text(json.dumps(costs), encoding="utf-8")

    assert rebuild_client.collect_result()["cost"] == costs


def test_collect_result_has_no_cost_when_the_rebuild_wrote_none(tmp_path):
    _start(tmp_path, _process(returncode=1))

    assert rebuild_client.collect_result()["cost"] is None


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"embeddings": "a lot"}', '{"embeddings": 0.1}'])
def test_read_cost_is_none_for_a_cost_file_it_cannot_use(tmp_path, text):
    path = tmp_path / "rebuild.cost.json"
    path.write_text(text, encoding="utf-8")

    assert rebuild_client.read_cost(path) is None


def test_read_cost_is_none_without_a_cost_file(tmp_path):
    assert rebuild_client.read_cost(tmp_path / "missing.json") is None


def test_empty_costs_start_every_kind_at_zero():
    assert rebuild_client.empty_costs() == {"embeddings": 0.0, "enrichment": 0.0, "citations": 0.0}


def test_describe_cost_gives_the_total_then_embeddings_and_enrichment():
    assert rebuild_client.describe_cost({"embeddings": 0.0123, "enrichment": 0.5, "citations": 0.0}) == (
        "Real cost: $0.5123 in all: embeddings $0.0123, describing images and tables $0.5000."
    )


def test_describe_cost_adds_reading_citations_when_it_cost_something():
    assert rebuild_client.describe_cost({"embeddings": 0.001, "enrichment": 0.0, "citations": 0.002}) == (
        "Real cost: $0.0030 in all: embeddings $0.0010, describing images and tables $0.0000, "
        "reading citations $0.0020."
    )
