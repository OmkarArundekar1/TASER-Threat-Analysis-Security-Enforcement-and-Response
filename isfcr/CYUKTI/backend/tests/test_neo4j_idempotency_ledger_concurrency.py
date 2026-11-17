"""Races: threads, and real separate processes. Whoever arrives first, one logical request results.

Timing can't be forced across processes, so these line children up behind a start file. They show
what happened in these runs under real contention, and the guarantee itself rests on the UNIQUE
constraint and BEGIN IMMEDIATE, which the unit and fault tests pin separately."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from neo4j_idempotency_ledger import (
    CheckStatus,
    IdempotencyConflictError,
    IdempotencyLedger,
    InvalidTransitionError,
    LedgerState as S,
    fingerprint_payload,
)

# Calls install_child_guards before importing backend code, as the crash tests check.
CHILD = str(Path(__file__).resolve().parent / "_neo4j_ledger_child.py")
KEY = "campaign-42/flush/7"
OP = "flush_duplicates"
ONE = [("CREATE (n:X {v:$v})", {"v": 1})]
TWO = [("CREATE (n:X {v:$v})", {"v": 2})]
FP = fingerprint_payload(OP, ONE).value


@pytest.fixture
def path(tmp_path):
    return tmp_path / "ledger.db"


def _race(count, target):
    barrier = threading.Barrier(count)
    results = []

    def run(index):
        barrier.wait()
        results.append(target(index))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(results) == count, "a thread failed or hung"
    return results


# ================================================================ 8-9 threads

def test_threads_registering_one_key_make_one_request(path):
    with IdempotencyLedger(path) as ledger:
        results = _race(16, lambda i: ledger.register(KEY, OP, ONE).status)
        assert sorted(r.value for r in results) == ["DUPLICATE"] * 15 + ["NEW"]
        assert len(ledger.list_requests()) == 1 and len(ledger.get_history(KEY)) == 1


def test_several_instances_in_one_process_registering_one_key_make_one_request(path):
    instances = [IdempotencyLedger(path) for _ in range(6)]
    try:
        results = _race(6, lambda i: instances[i].register(KEY, OP, ONE).status)
        assert sorted(r.value for r in results) == ["DUPLICATE"] * 5 + ["NEW"]
    finally:
        for instance in instances:
            instance.close()


def test_threads_registering_conflicting_payloads_cannot_both_win(path):
    with IdempotencyLedger(path) as ledger:
        def attempt(index):
            try:
                return ("ok", index % 2, ledger.register(KEY, OP, ONE if index % 2 else TWO).status.value)
            except IdempotencyConflictError:
                return ("conflict", index % 2, None)

        results = _race(16, attempt)
        winners = {variant for kind, variant, _ in results if kind == "ok"}
        assert len(winners) == 1
        losers = [r for r in results if r[0] == "conflict"]
        assert len(losers) == 8 and {r[1] for r in losers} != winners
        assert [r[2] for r in results if r[0] == "ok"].count("NEW") == 1
        request = ledger.get_request(KEY)
        assert request.fingerprint == fingerprint_payload(OP, ONE if winners == {1} else TWO).value
        assert request.conflict_count == 8 and len(ledger.get_conflicts(KEY)) == 8
        assert ledger.check_integrity().ok


def test_threads_claiming_one_request_produce_one_winner(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, OP, ONE)

        def attempt(_):
            try:
                ledger.begin_execution(KEY, FP)
                return "won"
            except InvalidTransitionError:
                return "refused"

        assert sorted(_race(16, attempt)) == ["refused"] * 15 + ["won"]
        assert ledger.get_request(KEY).attempt_count == 1 and len(ledger.get_history(KEY)) == 2


def test_threads_using_one_authorization_produce_one_attempt(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, OP, ONE)
        ledger.begin_execution(KEY, FP)
        ledger.record_not_completed(KEY, attempt=1)
        ledger.authorize_retry(KEY, authorized_by="op", reason="r")

        def attempt(_):
            try:
                ledger.begin_execution(KEY, FP)
                return "won"
            except InvalidTransitionError:
                return "refused"

        assert sorted(_race(8, attempt)) == ["refused"] * 7 + ["won"]
        assert ledger.get_request(KEY).attempt_count == 2


def test_threads_reporting_outcomes_for_one_attempt_produce_one_result(path):
    with IdempotencyLedger(path) as ledger:
        ledger.register(KEY, OP, ONE)
        ledger.begin_execution(KEY, FP)
        moves = [ledger.record_completed, ledger.record_not_completed, ledger.record_uncertain] * 4

        def attempt(index):
            try:
                moves[index](KEY, attempt=1)
                return "won"
            except InvalidTransitionError:
                return "refused"

        assert sorted(_race(12, attempt)) == ["refused"] * 11 + ["won"]
        assert len(ledger.get_history(KEY)) == 3 and ledger.check_integrity().ok


def test_a_read_during_writes_sees_only_complete_states(path):
    with IdempotencyLedger(path) as writer, IdempotencyLedger(path, read_only=True) as reader:
        stop = threading.Event()
        seen, failures = [], []

        def read():
            while not stop.is_set():
                try:
                    for request in reader.list_requests():
                        seen.append(request.state)
                except Exception as error:  # the assertion below reports it
                    failures.append(repr(error))
                    return

        thread = threading.Thread(target=read)
        thread.start()
        for number in range(30):
            key = f"k/{number}"
            writer.register(key, OP, ONE)
            writer.begin_execution(key, FP)
            writer.record_completed(key, attempt=1)
        stop.set()
        thread.join(30)
        assert failures == [] and seen


def test_many_threads_writing_different_keys_keep_the_ledger_sound(path):
    with IdempotencyLedger(path) as ledger:
        def work(index):
            for number in range(10):
                key = f"t{index}/{number}"
                ledger.register(key, OP, ONE)
                ledger.begin_execution(key, FP)
                ledger.record_completed(key, attempt=1)
            return index

        _race(6, work)
        report = ledger.check_integrity()
        assert report.ok and report.requests_checked == 60
        assert len(ledger.list_requests(S.COMPLETED)) == 60


# ================================================================ 10 separate processes

class _Racers:
    def __init__(self, tmp_path, path, specs):
        self.go = tmp_path / "go"
        self.processes = [
            subprocess.Popen([sys.executable, CHILD, action, str(path), str(self.go), *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, cwd=str(tmp_path), env=dict(os.environ))
            for action, *arguments in specs
        ]

    def run(self):
        time.sleep(2.0)  # long enough for each to import and reach the start line
        self.go.touch()
        results = []
        for process in self.processes:
            out, err = process.communicate(timeout=90)
            assert process.returncode == 0, err[-800:]
            results.append(json.loads(out.strip().splitlines()[-1]))
        return results


def test_processes_registering_one_key_make_one_request(tmp_path, path):
    IdempotencyLedger(path).close()
    results = _Racers(tmp_path, path, [("race_register", "same")] * 6).run()
    assert sorted(r["status"] for r in results) == ["DUPLICATE"] * 5 + ["NEW"]
    with IdempotencyLedger(path) as ledger:
        assert len(ledger.list_requests()) == 1 and len(ledger.get_history(KEY)) == 1 and ledger.check_integrity().ok


def test_processes_registering_conflicting_payloads_cannot_both_win(tmp_path, path):
    IdempotencyLedger(path).close()
    results = _Racers(tmp_path, path, [("race_register", "a")] * 3 + [("race_register", "b")] * 3).run()
    winners = {r["variant"] for r in results if r["status"] != "CONFLICT"}
    assert len(winners) == 1, results
    assert [r["status"] for r in results].count("NEW") == 1
    conflicts = [r for r in results if r["status"] == "CONFLICT"]
    assert len(conflicts) == 3 and all(r["variant"] not in winners for r in conflicts)
    assert sorted(r["conflict_no"] for r in conflicts) == [1, 2, 3], "each audited, none lost"
    with IdempotencyLedger(path) as ledger:
        assert len(ledger.list_requests()) == 1 and ledger.get_request(KEY).conflict_count == 3 and ledger.check_integrity().ok


def _registered_by_child(tmp_path, path):
    result = subprocess.run([sys.executable, CHILD, "register_and_exit", str(path), str(tmp_path / "unused")], capture_output=True, text=True,
                            timeout=60, cwd=str(tmp_path), env=dict(os.environ))
    assert result.returncode == 0, result.stderr[-500:]
    return json.loads(result.stdout.strip().splitlines()[-1])["fingerprint"]


def test_processes_claiming_one_request_produce_one_winner(tmp_path, path):
    _registered_by_child(tmp_path, path)
    results = _Racers(tmp_path, path, [("race_begin",)] * 6).run()
    assert sum(r["won"] for r in results) == 1
    with IdempotencyLedger(path) as ledger:
        assert ledger.get_request(KEY).attempt_count == 1 and len(ledger.get_history(KEY)) == 2


def test_processes_authorizing_one_retry_produce_one_authorization(tmp_path, path):
    fingerprint = _registered_by_child(tmp_path, path)
    with IdempotencyLedger(path) as ledger:
        ledger.begin_execution(KEY, fingerprint)
        ledger.record_not_completed(KEY, attempt=1)
    results = _Racers(tmp_path, path, [("race_authorize",)] * 4).run()
    assert sum(r["won"] for r in results) == 1
    with IdempotencyLedger(path) as ledger:
        assert ledger.get_request(KEY).state is S.RETRY_AUTHORIZED
        assert [e.cause for e in ledger.get_history(KEY)].count("RETRY") == 1


def test_processes_writing_different_keys_all_persist(tmp_path, path):
    IdempotencyLedger(path).close()
    results = _Racers(tmp_path, path, [("many_keys", f"p{i}") for i in range(3)]).run()
    expected = {key for result in results for key in result["keys"]}
    assert len(expected) == 30
    with IdempotencyLedger(path) as ledger:
        assert {r.idempotency_key for r in ledger.list_requests()} == expected and ledger.check_integrity().ok


def test_processes_creating_the_file_at_once_never_damage_it(tmp_path, path):
    """Opening can report busy while another process finishes creating it, so a caller may have to
    retry. What it must not do is lose or damage anything."""
    go = tmp_path / "go"
    processes = [subprocess.Popen([sys.executable, CHILD, "many_keys", str(path), str(go), f"p{i}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, cwd=str(tmp_path), env=dict(os.environ)) for i in range(3)]
    go.touch()
    expected = set()
    for process in processes:
        out, err = process.communicate(timeout=90)
        if process.returncode == 0:
            expected.update(json.loads(out.strip().splitlines()[-1])["keys"])
        else:
            assert "LedgerBusyError" in err, err[-500:]
    assert expected
    with IdempotencyLedger(path) as ledger:
        assert {r.idempotency_key for r in ledger.list_requests()} == expected and ledger.check_integrity().ok


def test_a_conflict_after_a_race_still_changes_nothing(tmp_path, path):
    IdempotencyLedger(path).close()
    _Racers(tmp_path, path, [("race_register", "same")] * 3).run()
    with IdempotencyLedger(path) as ledger:
        original = ledger.get_request(KEY)
        assert ledger.check(KEY, "other-operation", ONE).status is CheckStatus.CONFLICT
        assert ledger.get_request(KEY) == original


def test_recoveries_at_once_convert_each_request_exactly_once(path):
    with IdempotencyLedger(path) as first:
        keys = [f"k/{number}" for number in range(20)]
        for key in keys:
            first.register(key, OP, ONE)
            first.begin_execution(key, FP)

    def recover(_):
        with IdempotencyLedger(path) as instance:
            return instance.recover_unfinished()

    _race(4, recover)
    with IdempotencyLedger(path) as final:
        final.recover_unfinished()
        for key in keys:
            assert final.get_request(key).state is S.UNCERTAIN
            assert [e.cause for e in final.get_history(key)].count("RECOVERY") == 1
        assert final.check_integrity().ok


def test_a_recovery_that_loses_the_race_reports_the_real_state_and_does_not_raise(path, monkeypatch):
    """Forces the loser's path: its scan saw IN_PROGRESS, then another recovery converted it."""
    with IdempotencyLedger(path) as first:
        first.register(KEY, OP, ONE)
        first.begin_execution(KEY, FP)
    with IdempotencyLedger(path) as loser, IdempotencyLedger(path) as winner:
        real = loser._session_is_alive

        def winner_goes_first(session_id):
            winner.recover_unfinished()
            return real(session_id)

        monkeypatch.setattr(loser, "_session_is_alive", winner_goes_first)
        report = loser.recover_unfinished()
        assert report.recovered == () and report.uncertain == (KEY,)
        assert [e.cause for e in loser.get_history(KEY)].count("RECOVERY") == 1
