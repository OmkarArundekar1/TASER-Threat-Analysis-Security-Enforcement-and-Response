"""Runs coordinator steps in a real process, which can really die. Started by the crash tests.

usage: _neo4j_coordinator_child.py ACTION JOURNAL LEDGER EFFECTS [--kill POINT] [--step STEP] [--go PATH]
       [--hold STARTED RELEASE] [--source KEY] [--decision DECISION]

A POINT such as ledger.begin_execution.after makes that call on the real store, then dies by SIGKILL,
which runs no cleanup. writer.before and writer.after die inside the fake writer, either side of the
line it appends to EFFECTS. That file stands in for "a write took place" and validates nothing about
Neo4j. One JSON line goes to stdout at the end, unless the process died.
"""
import argparse
import json
import os
import signal
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0] = BACKEND

from tests._production_safety import install_child_guards  # noqa: E402

install_child_guards()

from listener.utils import request_key  # noqa: E402
from neo4j_idempotency_ledger import IdempotencyLedger  # noqa: E402
from neo4j_write_coordinator import (  # noqa: E402
    Decision,
    OperationRegistry,
    RegisteredOperation,
    WriteCoordinator,
    WriteRequest,
)
from neo4j_write_journal import NeoWriteJournal  # noqa: E402
from tests._neo4j_fake_writer import FakeWriter, Script  # noqa: E402

SOURCE = "wazuh:manager-01/1700000000.1234"
OPERATION = RegisteredOperation("test_event.update", "MATCH (n:CoordinatorTestOnly {id:$id}) SET n.v = $v", frozenset({"id", "v"}))
PARAMETERS = {"id": "E1", "v": 1}
REGISTRY = OperationRegistry([OPERATION])

JOURNAL_METHODS = ("create_intent", "mark_in_flight", "authorize_retry", "mark_committed", "mark_not_committed",
                   "mark_ambiguous", "resolve_ambiguous", "recover_unfinished")
LEDGER_METHODS = ("register", "begin_execution", "record_completed", "record_not_completed", "record_uncertain",
                  "authorize_retry", "resolve_uncertain", "recover_unfinished")


def die():
    os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(60)


def killing(base, store, methods, point):
    """A real store, dying either side of one named call."""
    def wrap(name):
        original = getattr(base, name)

        def method(self, *args, **kwargs):
            if point == f"{store}.{name}.before":
                die()
            result = original(self, *args, **kwargs)
            if point == f"{store}.{name}.after":
                die()
            return result
        return method
    return type(f"Killing{base.__name__}", (base,), {name: wrap(name) for name in methods})


def build(arguments):
    journal = killing(NeoWriteJournal, "journal", JOURNAL_METHODS, arguments.kill)(arguments.journal)
    ledger = killing(IdempotencyLedger, "ledger", LEDGER_METHODS, arguments.kill)(arguments.ledger)

    def effect(operation, statements):
        with open(arguments.effects, "a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()

    writer = FakeWriter(default=arguments.step, effect=effect)
    if arguments.hold:
        started, release = arguments.hold

        def hold():
            with open(started, "w", encoding="utf-8"):
                pass
            while not os.path.exists(release):
                time.sleep(0.005)
        writer.during = hold
    if arguments.kill == "writer.before":
        writer.during = die
    elif arguments.kill == "writer.after":
        original = writer.effect

        def effect_then_die(operation, statements):
            original(operation, statements)
            die()
        writer.effect = effect_then_die
    return journal, ledger, WriteCoordinator(journal, ledger, writer=writer, registry=REGISTRY, store_retry_delay=0.0)


def describe(journal, ledger, coordinator, key):
    view = coordinator.inspect(key)
    return {
        "combined": view.combined.value, "ledger": view.ledger_state.value if view.ledger_state else None,
        "attempt": view.ledger_attempt, "consistent": view.consistent,
        "journal": [record.state.value for record in view.journal],
        "tags": [record.ledger_attempt for record in view.journal],
        "epochs": [record.epoch for record in view.journal],
        "integrity": journal.check_integrity().ok and ledger.check_integrity().ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("journal")
    parser.add_argument("ledger")
    parser.add_argument("effects")
    parser.add_argument("--kill")
    parser.add_argument("--step", default=Script.COMMIT)
    parser.add_argument("--go")
    parser.add_argument("--hold", nargs=2, metavar=("STARTED", "RELEASE"))
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--decision", default=Decision.FOUND_NOT_COMMITTED.value)
    arguments = parser.parse_args()
    journal, ledger, coordinator = build(arguments)
    key = request_key(arguments.source, OPERATION.name)
    output = {}
    if arguments.action == "submit":
        result = coordinator.submit(WriteRequest(arguments.source, OPERATION.name, dict(PARAMETERS)))
        output["disposition"] = result.disposition.value
    elif arguments.action == "race":
        print(json.dumps({"ready": True}), flush=True)
        while not os.path.exists(arguments.go):
            time.sleep(0.001)
        result = coordinator.submit(WriteRequest(arguments.source, OPERATION.name, dict(PARAMETERS)))
        output["disposition"] = result.disposition.value
    elif arguments.action == "authorize":
        coordinator.authorize_retry(key, authorized_by="analyst", reason="crash test")
    elif arguments.action == "resolve":
        coordinator.resolve(key, decision=Decision(arguments.decision), resolved_by="analyst", reason="crash test")
    elif arguments.action == "retry":
        result = coordinator.retry(key)
        output["disposition"] = result.disposition.value
    elif arguments.action == "recover":
        found = coordinator.recover()
        output["categories"] = found.counts()
        output["integrity_ok"] = found.integrity_ok
    elif arguments.action != "inspect":
        raise SystemExit(f"unknown action {arguments.action}")
    output.update(describe(journal, ledger, coordinator, key))
    print(json.dumps(output), flush=True)
    journal.close()
    ledger.close()


if __name__ == "__main__":
    main()
