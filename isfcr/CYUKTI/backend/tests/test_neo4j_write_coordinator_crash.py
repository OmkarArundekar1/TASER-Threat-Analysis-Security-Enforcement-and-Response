"""WP-C1: a real process, using the real stores, really dies at each coordinator transition, and a fresh
process recovers. Also races across real processes.

The effects file only stands in for "a write took place": a fake writer says nothing about Neo4j. SIGKILL
keeps the page cache, so these show behaviour across process death, not power loss.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from tests._neo4j_child_tempdirs import child_processes
from tests._neo4j_fake_writer import Script

CHILD = str(Path(__file__).resolve().parent / "_neo4j_coordinator_child.py")


class Harness:
    def __init__(self, directory, children):
        self.journal = str(directory / "journal.db")
        self.ledger = str(directory / "ledger.db")
        self.effects_path = str(directory / "effects.log")
        self.children = children

    def command(self, action, *options):
        return [sys.executable, CHILD, action, self.journal, self.ledger, self.effects_path, *options]

    def run(self, action, *options, killed=False):
        result = self.children.run(self.command(action, *options), timeout=60, capture_output=True, text=True)
        if killed:
            assert result.returncode == -signal.SIGKILL, (action, options, result.returncode, result.stderr[-800:])
            return None
        assert result.returncode == 0, (action, options, result.returncode, result.stderr[-800:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    def die_at(self, action, point, *options):
        return self.run(action, "--kill", point, *options, killed=True)

    @property
    def effects(self):
        return Path(self.effects_path).read_text(encoding="utf-8").splitlines() if os.path.exists(self.effects_path) else []


@pytest.fixture
def harness(tmp_path):
    with child_processes(str(tmp_path)) as children:
        yield Harness(tmp_path, children)


def assert_after_recovery(found, *, ledger, journal, combined, attempt, category=None):
    assert found["ledger"] == ledger and found["journal"] == journal and found["combined"] == combined
    assert found["attempt"] == attempt and found["consistent"] and found["integrity"] and found["integrity_ok"]
    if category is not None:
        assert category in found["categories"], found["categories"]


def test_the_child_installs_its_guards_before_importing_any_backend_code():
    source = Path(CHILD).read_text(encoding="utf-8")
    assert source.index("install_child_guards()") < source.index("from neo4j_write_coordinator import")


# ================================================================ a first attempt, dying at each transition
# point, step, what the stores hold after recovery, how many effects, the category, and what a re-delivery says

FIRST_ATTEMPT = [
    # Before the claim, nothing can have run, and a re-delivery continues it.
    ("ledger.register.before", Script.COMMIT, None, None, None, 0, 0, None, "EXECUTED_COMMITTED"),
    ("ledger.register.after", Script.COMMIT, "REGISTERED", [], "REGISTERED_NO_INTENT", 0, 0, "REGISTERED_NO_INTENT", "EXECUTED_COMMITTED"),
    ("journal.create_intent.before", Script.COMMIT, "REGISTERED", [], "REGISTERED_NO_INTENT", 0, 0, "REGISTERED_NO_INTENT", "EXECUTED_COMMITTED"),
    ("journal.create_intent.after", Script.COMMIT, "REGISTERED", ["PENDING"], "REGISTERED", 0, 0, "PENDING", "EXECUTED_COMMITTED"),
    ("ledger.begin_execution.before", Script.COMMIT, "REGISTERED", ["PENDING"], "REGISTERED", 0, 0, "PENDING", "EXECUTED_COMMITTED"),
    # After the claim, nothing runs again without a person.
    ("ledger.begin_execution.after", Script.COMMIT, "UNCERTAIN", ["PENDING"], "CLAIMED_NOT_DISPATCHED", 1, 0, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("journal.mark_in_flight.before", Script.COMMIT, "UNCERTAIN", ["PENDING"], "CLAIMED_NOT_DISPATCHED", 1, 0, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("journal.mark_in_flight.after", Script.COMMIT, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 0, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("writer.before", Script.COMMIT, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 0, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("writer.after", Script.COMMIT, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 1, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("journal.mark_committed.before", Script.COMMIT, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 1, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    # Once the journal holds the outcome, recovery finishes the ledger with no person.
    ("journal.mark_committed.after", Script.COMMIT, "COMPLETED", ["COMMITTED"], "COMPLETED", 1, 1, "RECONCILED", "ALREADY_COMPLETED"),
    ("ledger.record_completed.before", Script.COMMIT, "COMPLETED", ["COMMITTED"], "COMPLETED", 1, 1, "RECONCILED", "ALREADY_COMPLETED"),
    ("ledger.record_completed.after", Script.COMMIT, "COMPLETED", ["COMMITTED"], "COMPLETED", 1, 1, None, "ALREADY_COMPLETED"),
    ("journal.mark_not_committed.before", Script.FAILED_BEFORE_COMMIT, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 0, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("journal.mark_not_committed.after", Script.FAILED_BEFORE_COMMIT, "NOT_COMPLETED", ["NOT_COMMITTED"], "AWAITING_AUTHORIZATION", 1, 0, "RECONCILED", "AWAITING_AUTHORIZATION"),
    ("ledger.record_not_completed.after", Script.FAILED_BEFORE_COMMIT, "NOT_COMPLETED", ["NOT_COMMITTED"], "AWAITING_AUTHORIZATION", 1, 0, "AWAITING_AUTHORIZATION", "AWAITING_AUTHORIZATION"),
    ("journal.mark_ambiguous.before", Script.AMBIGUOUS_APPLIED, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 1, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("journal.mark_ambiguous.after", Script.AMBIGUOUS_APPLIED, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 1, "NEWLY_UNCERTAIN", "BLOCKED_UNCERTAIN"),
    ("ledger.record_uncertain.after", Script.AMBIGUOUS_APPLIED, "UNCERTAIN", ["AMBIGUOUS"], "UNCERTAIN", 1, 1, "UNCERTAIN", "BLOCKED_UNCERTAIN"),
]


@pytest.mark.parametrize("point, step, ledger, journal, combined, attempt, effects, category, redelivery", FIRST_ATTEMPT,
                         ids=[f"{row[0]}-{row[1]}" for row in FIRST_ATTEMPT])
def test_dying_at_each_transition_of_a_first_attempt(harness, point, step, ledger, journal, combined, attempt, effects, category, redelivery):
    harness.die_at("submit", point, "--step", step)
    assert len(harness.effects) == effects

    found = harness.run("recover")
    if ledger is None:
        assert found["combined"] == "NOT_FOUND" and found["integrity"] and found["categories"] == {}
    else:
        assert_after_recovery(found, ledger=ledger, journal=journal, combined=combined, attempt=attempt, category=category)
    assert len(harness.effects) == effects, "recovery ran nothing"

    assert harness.run("recover")["categories"].get("NEWLY_UNCERTAIN") is None, "repeating recovery finds nothing new"
    assert len(harness.effects) == effects

    after = harness.run("submit")
    assert after["disposition"] == redelivery
    would_run = redelivery == "EXECUTED_COMMITTED"
    assert len(harness.effects) == effects + (1 if would_run else 0), "a re-delivery after a restart adds nothing unless nothing had run"
    assert after["integrity"] and after["consistent"]


def test_a_redelivery_without_recovery_never_runs_a_claimed_request(harness):
    harness.die_at("submit", "writer.after")
    assert harness.run("submit")["disposition"] == "IN_PROGRESS_ELSEWHERE"
    assert len(harness.effects) == 1


def test_a_crash_after_commit_needs_a_person_and_found_committed_ends_it(harness):
    harness.die_at("submit", "writer.after")
    harness.run("recover")
    found = harness.run("resolve", "--decision", "FOUND_COMMITTED")
    assert found["ledger"] == "COMPLETED" and found["journal"] == ["RESOLVED"] and found["consistent"]
    assert harness.run("submit")["disposition"] == "ALREADY_COMPLETED"
    assert len(harness.effects) == 1


def test_a_crash_before_dispatch_can_be_found_not_committed_and_retried_on_the_same_record(harness):
    harness.die_at("submit", "journal.mark_in_flight.before")
    harness.run("recover")
    found = harness.run("resolve")
    assert found["combined"] == "AWAITING_AUTHORIZATION" and found["journal"] == ["PENDING"]
    harness.run("authorize")
    found = harness.run("retry")
    assert found["disposition"] == "EXECUTED_COMMITTED" and found["attempt"] == 2 and found["journal"] == ["COMMITTED"] and found["tags"] == [2]
    assert len(harness.effects) == 1


# ================================================================ resolving and authorizing

def _uncertain(harness):
    harness.run("submit", "--step", Script.AMBIGUOUS_NOT_APPLIED)


def test_dying_between_the_journals_and_ledgers_resolution_is_finished_by_repeating_it(harness):
    _uncertain(harness)
    harness.die_at("resolve", "journal.resolve_ambiguous.after")
    found = harness.run("recover")
    assert found["combined"] == "RESOLUTION_INCOMPLETE" and "RESOLUTION_INCOMPLETE" in found["categories"]
    assert harness.run("submit")["disposition"] == "BLOCKED_UNCERTAIN"
    finished = harness.run("resolve")
    assert finished["combined"] == "AWAITING_AUTHORIZATION" and finished["ledger"] == "NOT_COMPLETED"
    assert harness.effects == []


def test_dying_after_the_ledgers_resolution_leaves_it_done(harness):
    _uncertain(harness)
    harness.die_at("resolve", "ledger.resolve_uncertain.after")
    found = harness.run("recover")
    assert found["combined"] == "AWAITING_AUTHORIZATION" and found["consistent"]


def test_an_authorization_survives_a_crash_unused(harness):
    harness.run("submit", "--step", Script.FAILED_BEFORE_COMMIT)
    harness.die_at("authorize", "ledger.authorize_retry.after")
    found = harness.run("recover")
    assert found["ledger"] == "RETRY_AUTHORIZED" and "RETRY_AUTHORIZED" in found["categories"] and harness.effects == []
    assert harness.run("submit")["disposition"] == "AWAITING_AUTHORIZATION"
    assert harness.effects == []
    assert harness.run("retry")["disposition"] == "EXECUTED_COMMITTED"
    assert len(harness.effects) == 1


# ================================================================ a retry on the same record (Path A)

def _authorized_non_commit(harness):
    harness.run("submit", "--step", Script.FAILED_BEFORE_COMMIT)
    harness.run("authorize")


def test_path_a_dying_after_the_claim_is_claimed_not_dispatched(harness):
    _authorized_non_commit(harness)
    harness.die_at("retry", "ledger.begin_execution.after")
    found = harness.run("recover")
    assert_after_recovery(found, ledger="UNCERTAIN", journal=["NOT_COMMITTED"], combined="CLAIMED_NOT_DISPATCHED", attempt=2, category="NEWLY_UNCERTAIN")
    assert found["tags"] == [1] and harness.effects == []
    assert harness.run("submit")["disposition"] == "BLOCKED_UNCERTAIN"


def test_path_a_a_claim_that_never_dispatched_can_be_found_not_committed_and_retried(harness):
    _authorized_non_commit(harness)
    harness.die_at("retry", "journal.authorize_retry.before")
    harness.run("recover")
    assert harness.run("resolve")["combined"] == "AWAITING_AUTHORIZATION"
    harness.run("authorize")
    found = harness.run("retry")
    assert found["disposition"] == "EXECUTED_COMMITTED" and found["attempt"] == 3 and found["tags"] == [3] and found["epochs"] == [1]
    assert len(harness.effects) == 1


def test_path_a_dying_after_in_flight_is_uncertain(harness):
    _authorized_non_commit(harness)
    harness.die_at("retry", "journal.authorize_retry.after")
    found = harness.run("recover")
    assert_after_recovery(found, ledger="UNCERTAIN", journal=["AMBIGUOUS"], combined="UNCERTAIN", attempt=2, category="NEWLY_UNCERTAIN")
    assert found["tags"] == [2] and harness.effects == []


def test_path_a_dying_after_the_write_is_uncertain_and_not_repeated(harness):
    _authorized_non_commit(harness)
    harness.die_at("retry", "writer.after")
    found = harness.run("recover")
    assert_after_recovery(found, ledger="UNCERTAIN", journal=["AMBIGUOUS"], combined="UNCERTAIN", attempt=2)
    assert harness.run("submit")["disposition"] == "BLOCKED_UNCERTAIN"
    assert len(harness.effects) == 1


# ================================================================ a retry on a new epoch (Path B)

def _authorized_after_resolution(harness):
    _uncertain(harness)
    harness.run("resolve")
    harness.run("authorize")


def test_path_b_dying_before_the_new_epoch_leaves_the_authorization(harness):
    _authorized_after_resolution(harness)
    harness.die_at("retry", "journal.create_intent.before")
    found = harness.run("recover")
    assert found["ledger"] == "RETRY_AUTHORIZED" and found["journal"] == ["RESOLVED"] and found["consistent"]
    assert harness.run("retry")["journal"] == ["RESOLVED", "COMMITTED"]
    assert len(harness.effects) == 1


def test_path_b_dying_after_the_new_epoch_reuses_it(harness):
    _authorized_after_resolution(harness)
    harness.die_at("retry", "journal.create_intent.after")
    found = harness.run("recover")
    assert found["ledger"] == "RETRY_AUTHORIZED" and found["journal"] == ["RESOLVED", "PENDING"] and found["consistent"]
    retried = harness.run("retry")
    assert retried["journal"] == ["RESOLVED", "COMMITTED"] and retried["epochs"] == [1, 2] and retried["tags"] == [1, 2]
    assert len(harness.effects) == 1


def test_path_b_dying_after_the_claim_is_claimed_not_dispatched(harness):
    _authorized_after_resolution(harness)
    harness.die_at("retry", "ledger.begin_execution.after")
    found = harness.run("recover")
    assert_after_recovery(found, ledger="UNCERTAIN", journal=["RESOLVED", "PENDING"], combined="CLAIMED_NOT_DISPATCHED", attempt=2)
    assert harness.effects == []


def test_path_b_dying_after_the_write_is_uncertain_on_the_new_epoch(harness):
    _authorized_after_resolution(harness)
    harness.die_at("retry", "writer.after")
    found = harness.run("recover")
    assert_after_recovery(found, ledger="UNCERTAIN", journal=["RESOLVED", "AMBIGUOUS"], combined="UNCERTAIN", attempt=2)
    assert found["tags"] == [1, 2] and len(harness.effects) == 1


# ================================================================ recovery itself

@pytest.mark.parametrize("point", ["journal.recover_unfinished.after", "ledger.recover_unfinished.before", "ledger.recover_unfinished.after"])
def test_dying_during_recovery_is_finished_by_running_it_again(harness, point):
    harness.die_at("submit", "writer.after")
    harness.die_at("recover", point)
    found = harness.run("recover")
    assert found["ledger"] == "UNCERTAIN" and found["journal"] == ["AMBIGUOUS"] and found["consistent"] and found["integrity"]
    assert len(harness.effects) == 1


def test_a_live_owner_in_another_process_is_left_alone(harness, tmp_path):
    """The owner is really in the middle of its write while another process recovers and re-delivers."""
    started, release = str(tmp_path / "started"), str(tmp_path / "release")
    owner = harness.children.spawn(harness.command("submit", "--hold", started, release), stdout=-1, stderr=-1, text=True)
    deadline = time.monotonic() + 30
    while not os.path.exists(started):
        assert owner.poll() is None and time.monotonic() < deadline
        time.sleep(0.01)

    found = harness.run("recover")
    assert found["categories"] == {"STILL_RUNNING": 1} and found["ledger"] == "IN_PROGRESS" and found["journal"] == ["IN_FLIGHT"]
    assert harness.run("submit")["disposition"] == "IN_PROGRESS_ELSEWHERE"
    assert harness.effects == []

    Path(release).write_text("go", encoding="utf-8")
    output, errors = owner.communicate(timeout=60)
    assert owner.returncode == 0, errors[-800:]
    finished = json.loads(output.strip().splitlines()[-1])
    assert finished["disposition"] == "EXECUTED_COMMITTED" and finished["ledger"] == "COMPLETED" and finished["journal"] == ["COMMITTED"]
    assert len(harness.effects) == 1


# ================================================================ real processes racing

def _start_racers(harness, count, go, *options):
    harness.run("inspect")  # creates both files: these tests race for the claim, not for creating the stores
    racers = [harness.children.spawn(harness.command("race", "--go", go, *options), stdout=-1, stderr=-1, text=True) for _ in range(count)]
    for racer in racers:
        line = racer.stdout.readline()
        assert line, racer.stderr.read()[-800:]
        assert json.loads(line)["ready"]
    Path(go).write_text("go", encoding="utf-8")
    return racers


@pytest.mark.parametrize("round_number", range(3))
def test_four_processes_racing_one_key_run_it_once(harness, tmp_path, round_number):
    racers = _start_racers(harness, 4, str(tmp_path / "go"))
    answers = []
    for racer in racers:
        output, errors = racer.communicate(timeout=60)
        assert racer.returncode == 0, errors[-800:]
        answers.append(json.loads(output.strip().splitlines()[-1]))
    dispositions = [answer["disposition"] for answer in answers]
    assert dispositions.count("EXECUTED_COMMITTED") == 1
    assert set(dispositions) <= {"EXECUTED_COMMITTED", "ALREADY_COMPLETED", "IN_PROGRESS_ELSEWHERE"}
    assert len(harness.effects) == 1
    assert harness.run("submit")["disposition"] == "ALREADY_COMPLETED"
    assert len(harness.effects) == 1


def test_a_process_killed_mid_race_never_lets_another_run_the_key_again(harness, tmp_path):
    """One racer dies straight after its write. Whoever else was racing can't repeat it."""
    racers = _start_racers(harness, 4, str(tmp_path / "go"), "--kill", "writer.after")
    for racer in racers:
        racer.communicate(timeout=60)
    assert sum(racer.returncode == -signal.SIGKILL for racer in racers) == 1
    assert len(harness.effects) == 1
    found = harness.run("recover")
    assert found["ledger"] == "UNCERTAIN" and found["journal"] == ["AMBIGUOUS"] and found["attempt"] == 1
    assert len(harness.effects) == 1
