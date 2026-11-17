"""Crash tests: a child process really dies by SIGKILL, and a fresh process reports what it finds.
The effects file stands in for Neo4j and shows recovery adds nothing to it. SIGKILL keeps the page
cache, so these prove behaviour across process death, not power loss."""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from neo4j_idempotency_ledger import (
    IdempotencyConflictError,
    IdempotencyLedger,
    InvalidTransitionError,
    LedgerState as S,
    fingerprint_payload,
)

CHILD = str(Path(__file__).resolve().parent / "_neo4j_ledger_child.py")
KEY = "campaign-42/flush/7"


@pytest.fixture
def env(tmp_path):
    return {"ledger": str(tmp_path / "ledger.db"), "effects": str(tmp_path / "effects.log"), "cwd": str(tmp_path)}


def _child_environment(env):
    """A killed child can't remove its temporary directory, so it goes in pytest's area."""
    return {**os.environ, "TMPDIR": env["cwd"]}


def _command(env, action, *args):
    return [sys.executable, CHILD, action, env["ledger"], env["effects"], *args]


def _run(env, action, *args, expect_killed=True, extra_env=None):
    result = subprocess.run(_command(env, action, *args), capture_output=True, text=True, timeout=60, cwd=env["cwd"],
                            env={**_child_environment(env), **(extra_env or {})})
    if expect_killed:
        assert result.returncode == -signal.SIGKILL, (action, result.returncode, result.stderr[-800:])
    else:
        assert result.returncode == 0, (action, result.returncode, result.stderr[-800:])
    lines = result.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def _inspect(env):
    return _run(env, "inspect", expect_killed=False)


def _effects(env):
    return Path(env["effects"]).read_text(encoding="utf-8").splitlines() if os.path.exists(env["effects"]) else []


def _states(request):
    return [step[1] for step in request["history"]]


def test_the_child_installs_its_guards_before_importing_any_backend_code():
    """Every process these tests start runs that file, so this covers all of them."""
    source = Path(CHILD).read_text(encoding="utf-8")
    assert source.index("install_child_guards()") < source.index("from neo4j_idempotency_ledger import")


def test_fingerprints_are_the_same_in_every_process_whatever_the_hash_seed(env):
    results = [_run(env, "print_fingerprint", expect_killed=False, extra_env={"PYTHONHASHSEED": seed}) for seed in ("1", "2", "31337")]
    assert results[0] == results[1] == results[2]


# ================================================================ 11 registration

def test_crash_after_registration_keeps_the_request_and_starts_nothing(env):
    _run(env, "after_register")
    seen = _inspect(env)
    request = seen["requests"][KEY]
    assert request["state"] == "REGISTERED" and request["attempts"] == 0 and request["matches_original"]
    assert seen["registered"] == [KEY] and seen["recovered"] == []
    assert _effects(env) == [] and seen["integrity_ok"]


def test_crash_during_registration_leaves_no_request_and_the_key_stays_free(env):
    _run(env, "dies_in_register")
    seen = _inspect(env)
    assert seen["requests"] == {} and seen["integrity_ok"]
    assert _run(env, "register_and_exit", expect_killed=False)["status"] == "NEW"


def test_crash_while_auditing_a_conflict_leaves_the_request_and_records_nothing(env):
    _run(env, "dies_in_conflict_audit")
    seen = _inspect(env)
    assert seen["requests"][KEY]["state"] == "REGISTERED" and seen["requests"][KEY]["conflicts"] == 0
    assert seen["requests"][KEY]["matches_original"] and seen["integrity_ok"]


def test_crash_during_initialization_leaves_something_that_opens_cleanly(env):
    _run(env, "dies_during_initialization")
    seen = _inspect(env)
    assert seen["requests"] == {} and seen["integrity_ok"]


# ================================================================ 12 state transitions

def test_crash_while_claiming_leaves_it_registered_and_claimable(env):
    _run(env, "dies_in_begin")
    request = _inspect(env)["requests"][KEY]
    assert request["state"] == "REGISTERED" and request["attempts"] == 0 and request["claimable"]
    assert _states(request) == ["REGISTERED"], "row and history change together"


def test_crash_after_claiming_before_sending_is_uncertain(env):
    _run(env, "after_begin")
    seen = _inspect(env)
    assert seen["requests"][KEY]["state"] == "UNCERTAIN" and not seen["requests"][KEY]["claimable"]
    assert seen["recovered"] == [KEY] and _effects(env) == []


@pytest.mark.parametrize("action", ["during_run", "during_commit"])
def test_crash_during_execution_is_uncertain_and_nothing_is_sent_again(env, action):
    _run(env, action)
    request = _inspect(env)["requests"][KEY]
    assert request["state"] == "UNCERTAIN" and request["last_outcome"] == "UNCERTAIN" and request["matches_original"]
    assert _effects(env) == [action]


def test_database_commits_then_crash_before_the_ledger_is_told(env):
    """The gap no local ledger can close: it did commit, and all that can be said is unknown."""
    _run(env, "after_database_commit")
    seen = _inspect(env)
    request = seen["requests"][KEY]
    assert seen["recovered"] == [KEY] and request["state"] == "UNCERTAIN"
    assert _states(request) == ["REGISTERED", "IN_PROGRESS", "UNCERTAIN"] and request["history"][-1][2] == "RECOVERY"
    assert request["recovery"]["attempt"] == 1
    again = _inspect(env)
    assert again["requests"][KEY]["state"] == "UNCERTAIN" and again["recovered"] == []
    assert _effects(env) == ["after_database_commit"], "one write, however many restarts"


def test_crash_while_recording_completion_leaves_no_false_completed(env):
    _run(env, "dies_in_record_completed")
    request = _inspect(env)["requests"][KEY]
    assert request["state"] == "UNCERTAIN" and "COMPLETED" not in _states(request)
    assert _effects(env) == ["committed"]


def test_confirmed_completion_survives_a_crash_and_cannot_run_again(env):
    _run(env, "after_completed")
    seen = _inspect(env)
    assert seen["requests"][KEY]["state"] == "COMPLETED" and seen["recovered"] == []
    fingerprint = seen["requests"][KEY]["fingerprint"]
    with IdempotencyLedger(env["ledger"]) as ledger:
        with pytest.raises(InvalidTransitionError):
            ledger.begin_execution(KEY, fingerprint)


def test_crash_while_resolving_leaves_it_uncertain(env):
    _run(env, "dies_in_resolve")
    seen = _inspect(env)
    assert seen["requests"][KEY]["state"] == "UNCERTAIN" and seen["integrity_ok"]


# ================================================================ 13 retry authorization

def test_confirmed_not_completed_survives_a_crash_and_stays_unclaimable(env):
    _run(env, "not_completed")
    seen = _inspect(env)
    request = seen["requests"][KEY]
    assert request["state"] == "NOT_COMPLETED" and not request["claimable"] and seen["not_completed"] == [KEY]
    assert seen["recovered"] == [], "recovery doesn't retry it"


def test_crash_during_retry_authorization_grants_nothing(env):
    _run(env, "dies_in_authorize")
    request = _inspect(env)["requests"][KEY]
    assert request["state"] == "NOT_COMPLETED" and not request["claimable"] and request["attempts"] == 1
    assert "RETRY_AUTHORIZED" not in _states(request)


def test_an_authorization_survives_a_crash_unused(env):
    _run(env, "after_authorize")
    seen = _inspect(env)
    request = seen["requests"][KEY]
    assert request["state"] == "RETRY_AUTHORIZED" and request["attempts"] == 1 and request["claimable"]
    assert seen["retry_authorized"] == [KEY]
    assert _effects(env) == [], "and recovery doesn't use it"


def test_authorizing_from_a_fresh_process_keeps_identity_and_history(env):
    _run(env, "not_completed")
    _run(env, "authorize_existing", expect_killed=False)
    request = _inspect(env)["requests"][KEY]
    assert _states(request) == ["REGISTERED", "IN_PROGRESS", "NOT_COMPLETED", "RETRY_AUTHORIZED"] and request["matches_original"]


# ================================================================ 14 recovery after process death

def test_caller_stops_waiting_then_the_process_dies(env):
    report = _run(env, "worker_outlives_caller")
    assert report == {"worker_alive": True, "state": "IN_PROGRESS"}
    assert _inspect(env)["requests"][KEY]["state"] == "UNCERTAIN"
    assert _effects(env) == ["query sent"]


def test_a_live_owner_is_not_recovered_until_it_really_dies(env):
    process = subprocess.Popen(_command(env, "holds_in_progress"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, cwd=env["cwd"], env=_child_environment(env))
    try:
        assert json.loads(process.stdout.readline())["ready"]
        with IdempotencyLedger(env["ledger"]) as ledger:
            report = ledger.recover_unfinished()
            assert report.still_running == (KEY,) and report.recovered == ()
            assert ledger.get_request(KEY).state is S.IN_PROGRESS
            process.stdin.close()
            assert process.wait(30) == -signal.SIGKILL
            assert ledger.recover_unfinished().recovered == (KEY,)
            assert ledger.get_request(KEY).state is S.UNCERTAIN
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(10)
        process.stdout.close()
        process.stderr.close()


def test_crash_during_recovery_then_the_next_recovery_finishes_the_rest(env):
    keys = _run(env, "many_in_progress")["keys"]
    _run(env, "dies_in_recovery")
    assert sorted(_run(env, "peek", expect_killed=False)["states"].values()) == ["IN_PROGRESS", "IN_PROGRESS", "UNCERTAIN"]
    seen = _inspect(env)
    for key in keys:
        assert seen["requests"][key]["state"] == "UNCERTAIN"
        assert [step[2] for step in seen["requests"][key]["history"]].count("RECOVERY") == 1
    assert seen["integrity_ok"]


def test_a_read_only_look_after_a_crash_sees_the_truth_and_converts_nothing(env):
    _run(env, "after_begin")
    seen = _run(env, "peek", expect_killed=False)
    assert list(seen["states"].values()) == ["IN_PROGRESS"] and seen["integrity_ok"]
    assert _inspect(env)["requests"][KEY]["state"] == "UNCERTAIN"


# ================================================================ 23 persistence across restarts

def test_identity_persists_across_process_restarts(env):
    first = _run(env, "register_and_exit", expect_killed=False)
    assert first["status"] == "NEW"
    for _ in range(2):
        again = _run(env, "register_and_exit", expect_killed=False)
        assert again == {"status": "DUPLICATE", "fingerprint": first["fingerprint"]}
    with IdempotencyLedger(env["ledger"]) as ledger:
        with pytest.raises(IdempotencyConflictError):
            ledger.register(KEY, "flush_duplicates", [("Q", None)])
        assert ledger.get_request(KEY).fingerprint == first["fingerprint"]
        assert first["fingerprint"] != fingerprint_payload("flush_duplicates", [("Q", None)]).value


def test_a_conflict_recorded_before_a_crash_is_still_there_afterwards(env):
    _run(env, "register_and_exit", expect_killed=False)
    with IdempotencyLedger(env["ledger"]) as ledger:
        with pytest.raises(IdempotencyConflictError):
            ledger.register(KEY, "flush_duplicates", [("Q", None)])
    assert _inspect(env)["requests"][KEY]["conflicts"] == 1


def test_killed_at_arbitrary_moments_the_ledger_is_always_sound(env):
    """Not aimed at any statement, so it lands wherever it lands, including mid-commit."""
    generator = random.Random(20261105)
    for run in range(8):
        process = subprocess.Popen(_command(env, "transition_loop", f"run{run}"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                   cwd=env["cwd"], env=_child_environment(env))
        assert json.loads(process.stdout.readline())["ready"]
        time.sleep(generator.uniform(0.01, 0.25))
        process.kill()
        assert process.wait(30) == -signal.SIGKILL
        process.stdout.close()
        seen = _inspect(env)
        assert seen["integrity_ok"], seen["problems"]
        assert seen["corrupt"] == []
        for request in seen["requests"].values():
            assert request["state"] in {"REGISTERED", "COMPLETED", "UNCERTAIN"}
            assert request["state"] == request["history"][-1][1]
    assert seen["requests"], "the child has to have written something for this to mean anything"


def test_a_forked_child_cannot_use_the_ledger_or_close_the_parents(env):
    report = _run(env, "fork_check", expect_killed=False)
    assert report == {"child_status": 0, "state_after": "REGISTERED", "lock_kept": True}
