import sys
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
        "log_path": str(run["log_path"]),
        "lines": ["Indexed after run: 3", "Failed PDFs: 0"],
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
