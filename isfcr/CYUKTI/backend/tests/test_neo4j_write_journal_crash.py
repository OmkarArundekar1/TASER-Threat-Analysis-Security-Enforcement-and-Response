"""Crash tests: a child process really dies by SIGKILL, and a fresh process reports what it finds.
No mocks of death and no in-process reopening: nothing from the dead process can help.

The effects file stands in for Neo4j. It shows what "reached the database", and that recovery
adds nothing to it. Power loss isn't tested here: SIGKILL keeps the page cache, so these prove
behaviour across process death, and fsync durability rests on SQLite's FULL setting."""
from __future__ import annotations

import glob
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from neo4j_write_journal import NeoWriteJournal, InvalidTransitionError, JournalState as S
from tests import _neo4j_child_tempdirs as tempdirs
from tests._neo4j_child_tempdirs import ChildTempCleanupError, child_processes, child_temp_area

CHILD = str(Path(__file__).resolve().parent / "_neo4j_journal_child.py")


@pytest.fixture
def env(tmp_path):
    with child_processes(str(tmp_path)) as children:
        yield {"journal": str(tmp_path / "journal.db"), "effects": str(tmp_path / "effects.log"), "cwd": str(tmp_path),
               "children": children}


def test_the_child_installs_its_guards_before_importing_any_backend_code():
    """Every process these tests start runs that file, so this covers all of them."""
    source = Path(CHILD).read_text(encoding="utf-8")
    assert source.index("install_child_guards()") < source.index("from neo4j_write_journal import")


def _command(env, action, *args):
    return [sys.executable, CHILD, action, env["journal"], env["effects"], *args]


def _run(env, action, *args, expect_killed=True):
    result = env["children"].run(_command(env, action, *args), capture_output=True, text=True, timeout=60, cwd=env["cwd"])
    if expect_killed:
        assert result.returncode == -signal.SIGKILL, (action, result.returncode, result.stderr[-800:])
    else:
        assert result.returncode == 0, (action, result.returncode, result.stderr[-800:])
    lines = result.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def _inspect(env):
    """A fresh process opens the journal, runs recovery and reports."""
    return _run(env, "inspect", expect_killed=False)


def _effects(env):
    return Path(env["effects"]).read_text(encoding="utf-8").splitlines() if os.path.exists(env["effects"]) else []


def _states(history):
    return [step[1] for step in history]


# ================================================================ the nine scenarios

def test_scenario_1_crash_before_execution(env):
    record_id = _run(env, "after_create")["id"]
    seen = _inspect(env)
    record = seen["records"][record_id]
    assert record["state"] == "PENDING" and record["intent_intact"] and record["attempts"] == 0
    assert seen["pending"] == [record_id] and seen["recovered"] == []
    assert _effects(env) == [], "nothing is executed"
    assert seen["integrity_ok"]


def test_scenario_2_crash_after_in_flight_before_the_query_was_sent(env):
    record_id = _run(env, "after_in_flight")["id"]
    seen = _inspect(env)
    record = seen["records"][record_id]
    assert record["state"] == "AMBIGUOUS" and not record["retry_eligible"]
    assert seen["recovered"] == [record_id]
    assert record["intent_intact"]
    assert _effects(env) == []


@pytest.mark.parametrize("action", ["during_run", "during_pull"])
def test_scenario_3_crash_during_run_or_pull(env, action):
    record_id = _run(env, action)["id"]
    seen = _inspect(env)
    record = seen["records"][record_id]
    assert record["state"] == "AMBIGUOUS" and record["intent_intact"] and not record["retry_eligible"]
    assert _effects(env) == [action], "and nothing was sent again"


def test_scenario_4_crash_during_commit(env):
    record_id = _run(env, "during_commit")["id"]
    seen = _inspect(env)
    record = seen["records"][record_id]
    assert record["state"] == "AMBIGUOUS" and record["last_outcome"] == "AMBIGUOUS"
    assert record["state"] != "NOT_COMMITTED", "the journal must not assume the transaction failed"
    assert _effects(env) == ["during_commit"]


def test_scenario_5_database_commits_then_crash_before_the_journal_is_told(env):
    """The case no local journal can close: the write did commit, and all it can say is unknown."""
    record_id = _run(env, "after_database_commit")["id"]
    seen = _inspect(env)
    record = seen["records"][record_id]
    assert seen["recovered"] == [record_id]
    assert record["state"] == "AMBIGUOUS" and not record["retry_eligible"]
    assert _states(record["history"]) == ["PENDING", "IN_FLIGHT", "AMBIGUOUS"]
    assert record["history"][-1][2] == "RECOVERY"
    assert record["recovery"]["attempt"] == 1
    assert _effects(env) == ["after_database_commit"], "exactly one write, however many restarts"
    again = _inspect(env)
    assert again["records"][record_id]["state"] == "AMBIGUOUS" and again["recovered"] == []
    assert _effects(env) == ["after_database_commit"]


def test_scenario_6_confirmed_committed_then_crash(env):
    record_id = _run(env, "after_committed")["id"]
    seen = _inspect(env)
    record = seen["records"][record_id]
    assert record["state"] == "COMMITTED" and not record["retry_eligible"] and seen["recovered"] == []
    with NeoWriteJournal(env["journal"]) as journal:
        with pytest.raises(InvalidTransitionError):
            journal.authorize_retry(record_id, authorized_by="op", reason="r")
        with pytest.raises(InvalidTransitionError):
            journal.mark_in_flight(record_id)


def test_scenario_7_confirmed_not_committed_then_crash_then_explicit_retry(env):
    record_id = _run(env, "not_committed")["id"]
    seen = _inspect(env)
    record = seen["records"][record_id]
    assert record["state"] == "NOT_COMMITTED" and record["retry_eligible"] and record["intent_intact"]
    assert seen["not_committed"] == [record_id] and seen["recovered"] == []
    assert record["state"] == "NOT_COMMITTED", "recovery doesn't retry it"
    retried = _run(env, "retry", record_id, expect_killed=False)
    assert retried == {"state": "IN_FLIGHT", "attempts": 2}
    after = _inspect(env)["records"][record_id]
    assert after["intent_intact"]
    assert _states(after["history"]) == ["PENDING", "IN_FLIGHT", "NOT_COMMITTED", "IN_FLIGHT", "AMBIGUOUS"], (
        "the retrying process ended with no outcome, so that attempt is unknown too")


def test_scenario_8a_caller_stops_waiting_then_the_process_dies(env):
    report = _run(env, "worker_outlives_caller")
    assert report["worker_alive"] and report["state"] == "IN_FLIGHT", "the caller's timeout concluded nothing"
    record = _inspect(env)["records"][report["id"]]
    assert record["state"] == "AMBIGUOUS"
    assert _effects(env) == ["query sent"], "no duplicate execution because the caller stopped waiting"


def test_scenario_8b_a_live_owner_is_not_recovered_until_it_really_dies(env):
    process = env["children"].spawn(_command(env, "holds_in_flight"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, cwd=env["cwd"])
    try:
        record_id = json.loads(process.stdout.readline())["id"]
        with NeoWriteJournal(env["journal"]) as journal:
            report = journal.recover_unfinished()
            assert report.still_running == (record_id,) and report.recovered == ()
            assert journal.get_record(record_id).state is S.IN_FLIGHT
            process.stdin.close()
            assert process.wait(30) == -signal.SIGKILL
            report = journal.recover_unfinished()
            assert report.recovered == (record_id,)
            assert journal.get_record(record_id).state is S.AMBIGUOUS
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(10)
        process.stdout.close()
        process.stderr.close()


def test_scenario_9_is_in_the_fault_tests():
    """Journal write failures need injected storage faults: test_neo4j_write_journal_faults.py."""
    assert (Path(__file__).parent / "test_neo4j_write_journal_faults.py").exists()


# ================================================================ dying inside a journal transaction

def test_dies_while_creating_leaves_no_record(env):
    _run(env, "dies_in_create")
    seen = _inspect(env)
    assert seen["records"] == {} and seen["integrity_ok"]


def test_dies_while_marking_in_flight_leaves_it_pending(env):
    record_id = _run(env, "dies_in_mark_in_flight")["id"]
    seen = _inspect(env)
    assert seen["records"][record_id]["state"] == "PENDING"
    assert _states(seen["records"][record_id]["history"]) == ["PENDING"], "row and history change together"
    assert seen["integrity_ok"]


def test_dies_while_marking_committed_leaves_no_false_committed(env):
    record_id = _run(env, "dies_in_mark_committed")["id"]
    seen = _inspect(env)
    assert seen["records"][record_id]["state"] == "AMBIGUOUS"
    assert "COMMITTED" not in _states(seen["records"][record_id]["history"])
    assert seen["integrity_ok"]


def test_dies_while_authorizing_a_retry_leaves_it_not_committed(env):
    record_id = _run(env, "dies_in_retry")["id"]
    seen = _inspect(env)
    assert seen["records"][record_id]["state"] == "NOT_COMMITTED" and seen["records"][record_id]["attempts"] == 1
    assert seen["integrity_ok"]


def test_dies_while_resolving_leaves_it_ambiguous(env):
    record_id = _run(env, "dies_in_resolve")["id"]
    seen = _inspect(env)
    assert seen["records"][record_id]["state"] == "AMBIGUOUS"
    assert seen["integrity_ok"]


def test_dies_during_recovery_then_the_next_recovery_finishes_the_rest(env):
    ids = _run(env, "many_in_flight")["ids"]
    _run(env, "dies_in_recovery")
    partial = _run(env, "peek", expect_killed=False)
    assert sorted(partial["states"].values()) == ["AMBIGUOUS", "IN_FLIGHT", "IN_FLIGHT"], "the first converted, the second died in its commit"
    seen = _inspect(env)
    assert all(seen["records"][i]["state"] == "AMBIGUOUS" for i in ids)
    for i in ids:
        assert [step[2] for step in seen["records"][i]["history"]].count("RECOVERY") == 1
    assert seen["integrity_ok"]


def test_dies_during_initialization_leaves_something_that_opens_cleanly(env):
    _run(env, "dies_during_initialization")
    seen = _inspect(env)
    assert seen["records"] == {} and seen["integrity_ok"]


def test_a_read_only_look_after_a_crash_sees_the_truth_and_converts_nothing(env):
    record_id = _run(env, "after_in_flight")["id"]
    seen = _run(env, "peek", expect_killed=False)
    assert seen["states"] == {record_id: "IN_FLIGHT"} and seen["integrity_ok"]
    assert _inspect(env)["records"][record_id]["state"] == "AMBIGUOUS"


def test_killed_at_arbitrary_moments_the_journal_is_always_sound(env):
    """Not aimed at any statement, so it lands wherever it lands: including mid-commit."""
    generator = random.Random(20261105)
    completed_states = {"PENDING", "IN_FLIGHT", "COMMITTED", "AMBIGUOUS"}
    for _ in range(8):
        process = env["children"].spawn(_command(env, "transition_loop"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                   cwd=env["cwd"])
        assert json.loads(process.stdout.readline())["ready"]
        time.sleep(generator.uniform(0.01, 0.25))
        process.kill()
        assert process.wait(30) == -signal.SIGKILL
        process.stdout.close()
        seen = _inspect(env)
        assert seen["integrity_ok"], seen["problems"]
        assert seen["corrupt"] == []
        for record in seen["records"].values():
            assert record["state"] in completed_states
            assert record["state"] == record["history"][-1][1]
            assert record["intent_intact"]
            if "COMMITTED" not in _states(record["history"]):
                assert record["state"] != "COMMITTED"
    assert seen["records"], "the child has to have written something for this to mean anything"


# ================================================================ several processes

def test_processes_racing_for_one_transition_produce_one_winner(env, tmp_path):
    with NeoWriteJournal(env["journal"]) as journal:
        record_id = journal.create_intent("op", [("Q", None)]).record_id
    go = tmp_path / "go"
    processes = [env["children"].spawn([sys.executable, CHILD, "race_mark_in_flight", env["journal"], str(go), record_id], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, cwd=env["cwd"]) for _ in range(4)]
    time.sleep(1.5)
    go.touch()
    results = []
    for process in processes:
        out, err = process.communicate(timeout=60)
        assert process.returncode == 0, err[-500:]
        results.append(json.loads(out.strip().splitlines()[-1]))
    assert sum(r["won"] for r in results) == 1
    assert all(r["error"] == "InvalidTransitionError" for r in results if not r["won"])
    with NeoWriteJournal(env["journal"]) as journal:
        assert journal.get_record(record_id).attempt_count == 1
        assert len(journal.get_history(record_id)) == 2


def _concurrent_creators(env):
    processes = [env["children"].spawn(_command(env, "create_many"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  cwd=env["cwd"]) for _ in range(3)]
    finished = []
    for process in processes:
        out, err = process.communicate(timeout=60)
        finished.append((process.returncode, out, err))
    return finished


def test_processes_writing_at_once_all_persist(env):
    NeoWriteJournal(env["journal"]).close()
    expected = set()
    for code, out, err in _concurrent_creators(env):
        assert code == 0, err[-500:]
        expected.update(json.loads(out.strip().splitlines()[-1])["ids"])
    assert len(expected) == 30
    seen = _inspect(env)
    assert expected == set(seen["records"]) and seen["integrity_ok"]


def test_processes_creating_the_file_at_once_never_damage_it(env):
    """Opening can report that the database is busy while another process finishes creating it,
    so a caller may have to retry. What it must not do is lose or damage anything."""
    expected = set()
    for code, out, err in _concurrent_creators(env):
        if code == 0:
            expected.update(json.loads(out.strip().splitlines()[-1])["ids"])
        else:
            assert "JournalBusyError" in err, err[-500:]
    assert expected
    seen = _inspect(env)
    assert expected == set(seen["records"]) and seen["integrity_ok"]


def test_a_forked_child_cannot_use_the_journal_or_close_the_parents(env):
    report = _run(env, "fork_check", expect_killed=False)
    assert report == {"child_status": 0, "state_after": "PENDING", "lock_kept": True}


# ================================================================ a killed child's temporary directory

def _state_directories_in_the_real_temp():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "cyukti-test-state-*")))


def _killed_child(tmp_path, children):
    result = children.run([sys.executable, CHILD, "after_create", str(tmp_path / "journal.db"), str(tmp_path / "effects.log")],
                          capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert result.returncode == -signal.SIGKILL, "it has to be real termination, or this proves nothing"
    return json.loads(result.stdout.strip().splitlines()[-1])["id"]


def test_cleanup_removes_what_a_killed_child_left_and_nothing_reaches_the_real_temp(tmp_path):
    before = _state_directories_in_the_real_temp()
    with child_processes(str(tmp_path)) as children:
        record_id = _killed_child(tmp_path, children)
        [area] = children.directories
        left = os.listdir(area)
        assert len(left) == 1 and left[0].startswith("cyukti-test-state-"), "SIGKILL skipped the child's own cleanup"
        held = os.path.join(area, left[0])
        assert os.path.isdir(held)
    assert not os.path.exists(held) and not os.path.exists(area)
    assert _state_directories_in_the_real_temp() == before
    with NeoWriteJournal(tmp_path / "journal.db") as journal:
        assert journal.get_record(record_id).state is S.PENDING, "the crash's result is untouched"


def test_cleanup_runs_when_the_test_itself_fails(tmp_path):
    with pytest.raises(ValueError):
        with child_processes(str(tmp_path)) as children:
            _killed_child(tmp_path, children)
            [area] = children.directories
            raise ValueError("the test failed")
    assert not os.path.exists(area)


def test_cleanup_removes_a_state_directory_that_has_contents(tmp_path):
    with child_temp_area(str(tmp_path)) as area:
        held = os.path.join(area, "cyukti-test-state-abcd_123")
        os.mkdir(held)
        (Path(held) / "playbook_memory.db").write_bytes(b"x")
    assert not os.path.exists(area)


@pytest.mark.parametrize("name", ["unrelated", "cyukti-test-state-", "cyukti-test-state-short", "cyukti-test-state-toolongname", "cyukti-test-state-ABCD1234", "xcyukti-test-state-abcd1234"])
def test_cleanup_refuses_an_unexpected_name_and_leaves_it(tmp_path, name):
    with pytest.raises(ChildTempCleanupError, match="unexpected name"):
        with child_temp_area(str(tmp_path)) as area:
            os.mkdir(os.path.join(area, name))
    assert os.path.isdir(os.path.join(area, name))


def test_cleanup_refuses_a_symlink_and_never_follows_it(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    with pytest.raises(ChildTempCleanupError, match="not a real directory"):
        with child_temp_area(str(tmp_path)) as area:
            os.symlink(outside, os.path.join(area, "cyukti-test-state-abcd1234"))
    assert (outside / "keep").read_text() == "keep"


def test_cleanup_refuses_a_file_with_the_expected_name(tmp_path):
    with pytest.raises(ChildTempCleanupError, match="not a real directory"):
        with child_temp_area(str(tmp_path)) as area:
            Path(area, "cyukti-test-state-abcd1234").write_text("x")
    assert os.path.isfile(os.path.join(area, "cyukti-test-state-abcd1234"))


def test_cleanup_refuses_a_directory_owned_by_someone_else(tmp_path, monkeypatch):
    with pytest.raises(ChildTempCleanupError, match="owned by uid"):
        with child_temp_area(str(tmp_path)) as area:
            os.mkdir(os.path.join(area, "cyukti-test-state-abcd1234"))
            monkeypatch.setattr(tempdirs.os, "getuid", lambda: os.stat(area).st_uid + 1)
    assert os.path.isdir(os.path.join(area, "cyukti-test-state-abcd1234"))


def test_cleanup_reports_a_failed_removal_rather_than_ignoring_it(tmp_path, monkeypatch):
    def fails(path):
        raise PermissionError("injected")

    with pytest.raises(ChildTempCleanupError, match="PermissionError"):
        with child_temp_area(str(tmp_path)) as area:
            os.mkdir(os.path.join(area, "cyukti-test-state-abcd1234"))
            monkeypatch.setattr(tempdirs.shutil, "rmtree", fails)


def test_cleanup_covers_every_child_these_tests_start():
    """A child started with the plain environment would leak into the real temp directory again."""
    plain = "env=dict(" + "os.environ)"
    for name in ("test_neo4j_write_journal_crash.py", "test_neo4j_write_journal_faults.py"):
        assert plain not in (Path(__file__).parent / name).read_text(encoding="utf-8"), name
