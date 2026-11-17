"""Measures the Phase 2.6B journal on synthetic data, in a fresh temporary directory that it removes afterwards. Opens no
service and no real journal. The payload is W07U-shaped: the query text of update_duplicate_event and a 384-byte
investigation_payload, the sample's median (phase_2_6a_decisions/evidence/d3_d13_payload_analysis_output.txt).
usage (from isfcr/CYUKTI/backend): python -B measure_performance.py"""
import datetime as dt
import os
import platform
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())

import neo4j_write_journal as v2  # noqa: E402
from neo4j_write_journal import NeoWriteJournal, Resolution  # noqa: E402
from tests import _neo4j_journal_v1_frozen as v1  # noqa: E402

QUERY = "WITH coalesce($event_time, datetime()) AS dt MATCH (e:AttackEvent {event_id:$event_id}) SET " + "e.x = coalesce(e.x,0) + 1, " * 12 + "e.y = 1"
STATEMENTS = [(QUERY, {"event_id": "00000000-0000-4000-8000-000000000000", "tps": 0, "investigation_payload": "p" * 384,
                       "event_time": dt.datetime(2026, 3, 11, 3, 29, 24, 123000, tzinfo=dt.timezone.utc)})]
OP = "duplicate_event.update"
ROOT = tempfile.mkdtemp(prefix="cyukti-p26b-perf-")
SMALL, LARGE = 1000, 10000


def key(number):
    return f"cyukti.v1|wazuh:manager-01/1773199644.{number}|duplicate_event.update"


def timed(call, repeat):
    samples = []
    for number in range(repeat):
        started = time.perf_counter()
        call(number)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return samples


def line(label, samples):
    print(f"{label:<58} n={len(samples):<6} median {statistics.median(samples):8.3f} ms  p99 {samples[int(len(samples) * .99)]:8.3f} ms  max {samples[-1]:8.3f} ms")


def path(name):
    return os.path.join(ROOT, name)


def file_bytes(name):
    return sum(os.path.getsize(path(name) + suffix) for suffix in ("", "-wal") if os.path.exists(path(name) + suffix))


def resolve(journal, record, attempt):
    journal.mark_in_flight(record.record_id, ledger_attempt=attempt)
    journal.mark_ambiguous(record.record_id, expected_ledger_attempt=attempt)
    return journal.resolve_ambiguous(record.record_id, resolution=Resolution.FOUND_NOT_COMMITTED, resolved_by="a", reason="r")


print(f"python {platform.python_version()}, sqlite {sqlite3.sqlite_version}, {platform.platform()}")
print(f"temporary directory filesystem device {os.stat(ROOT).st_dev}; payload statement bytes {len(QUERY) + 384}")
try:
    print("\n--- one intent")
    with v1.NeoWriteJournal(path("v1.db")) as journal:
        line("create_intent, previous implementation (v1)", timed(lambda n: journal.create_intent(OP, STATEMENTS), SMALL))
    with NeoWriteJournal(path("plain.db")) as journal:
        line("create_intent, uncorrelated", timed(lambda n: journal.create_intent(OP, STATEMENTS), SMALL))
    with NeoWriteJournal(path("corr.db")) as journal:
        line("create_intent, correlated, epoch 1", timed(lambda n: journal.create_intent(OP, STATEMENTS, correlation_key=key(n), epoch=1), SMALL))

    print("\n--- a whole normal life: create, IN_FLIGHT, COMMITTED")
    with v1.NeoWriteJournal(path("v1life.db")) as journal:
        def life(n):
            record = journal.create_intent(OP, STATEMENTS)
            journal.mark_in_flight(record.record_id)
            journal.mark_committed(record.record_id)
        line("previous implementation (v1)", timed(life, SMALL))
    with NeoWriteJournal(path("plainlife.db")) as journal:
        def life(n):
            record = journal.create_intent(OP, STATEMENTS)
            journal.mark_in_flight(record.record_id)
            journal.mark_committed(record.record_id)
        line("uncorrelated", timed(life, SMALL))
    with NeoWriteJournal(path("corrlife.db")) as journal:
        def life(n):
            record = journal.create_intent(OP, STATEMENTS, correlation_key=key(n), epoch=1)
            journal.mark_in_flight(record.record_id, ledger_attempt=1)
            journal.mark_committed(record.record_id, expected_ledger_attempt=1)
        line("correlated", timed(life, SMALL))

    print("\n--- does cost grow with the journal? correlated create and lookup at two sizes")
    for size, name in ((SMALL, "s.db"), (LARGE, "l.db")):
        with NeoWriteJournal(path(name)) as journal:
            for n in range(size):
                journal.create_intent(OP, STATEMENTS, correlation_key=key(n), epoch=1)
            line(f"create_intent, correlated, {size} records present", timed(lambda n: journal.create_intent(OP, STATEMENTS, correlation_key=key(size + n), epoch=1), 300))
            line(f"preflight_intent, new key, {size} records present", timed(lambda n: journal.preflight_intent(OP, STATEMENTS, correlation_key=key(size * 2 + n), epoch=1), 1000))
            line(f"preflight_intent, duplicate, {size} records present", timed(lambda n: journal.preflight_intent(OP, STATEMENTS, correlation_key=key(n), epoch=1), 1000))
            line(f"preflight_intent, uncorrelated, {size} records present", timed(lambda n: journal.preflight_intent(OP, STATEMENTS), 1000))
            line(f"find_by_correlation_key, one epoch, {size} records present", timed(lambda n: journal.find_by_correlation_key(key(n)), 1000))
            line(f"find_by_correlation_key, unknown, {size} records present", timed(lambda n: journal.find_by_correlation_key(key(size * 3 + n)), 1000))

    print("\n--- cost by number of epochs, one request")
    with NeoWriteJournal(path("epochs.db")) as journal:
        previous = resolve(journal, journal.create_intent(OP, STATEMENTS, correlation_key=key(0), epoch=1), 1)
        epoch = 1
        for wanted in (1, 2, 5, 20, 100):
            while epoch < wanted:
                epoch += 1
                previous = resolve(journal, journal.create_intent(OP, previous.statements, correlation_key=key(0), epoch=epoch, authorized_by="a", reason="r"), epoch)
            line(f"find_by_correlation_key, {wanted} epochs", timed(lambda n: journal.find_by_correlation_key(key(0)), 300))
            line(f"preflight_intent, next epoch, {wanted} epochs before it", timed(
                lambda n: journal.preflight_intent(OP, previous.statements, correlation_key=key(0), epoch=epoch + 1, authorized_by="a", reason="r"), 300))
        started = time.perf_counter()
        record = journal.create_intent(OP, previous.statements, correlation_key=key(0), epoch=epoch + 1, authorized_by="a", reason="r")
        print(f"create_intent, epoch {epoch + 1}, once: {(time.perf_counter() - started) * 1000:.3f} ms")
        started = time.perf_counter()
        journal.mark_in_flight(record.record_id, ledger_attempt=epoch + 1)
        print(f"mark_in_flight with {epoch + 1} epochs to compare, once: {(time.perf_counter() - started) * 1000:.3f} ms")

    print("\n--- integrity check over 3 transitions a record")
    for label, correlated in (("uncorrelated", False), ("correlated", True)):
        name = f"integrity-{label}.db"
        with NeoWriteJournal(path(name)) as journal:
            for n in range(SMALL * 3):
                if correlated:
                    record = journal.create_intent(OP, STATEMENTS, correlation_key=key(n), epoch=1)
                    journal.mark_in_flight(record.record_id, ledger_attempt=1)
                    journal.mark_committed(record.record_id, expected_ledger_attempt=1)
                else:
                    record = journal.create_intent(OP, STATEMENTS)
                    journal.mark_in_flight(record.record_id)
                    journal.mark_committed(record.record_id)
            started = time.perf_counter()
            assert journal.check_integrity().ok
            print(f"check_integrity, {label}, {SMALL * 3} records: {(time.perf_counter() - started) * 1000:.1f} ms")
        print(f"file size after closing, {label}: {file_bytes(name)} bytes, {file_bytes(name) / (SMALL * 3):.0f} bytes a record with 3 history entries")

    print("\n--- growth of the empty-column cost: the same records, version 1 and migrated")
    with v1.NeoWriteJournal(path("grow.db")) as journal:
        for n in range(SMALL * 3):
            record = journal.create_intent(OP, STATEMENTS)
            journal.mark_in_flight(record.record_id)
            journal.mark_committed(record.record_id)
    before = file_bytes("grow.db")
    started = time.perf_counter()
    NeoWriteJournal(path("grow.db")).close()
    print(f"migration of {SMALL * 3} records and {SMALL * 9} history entries: {(time.perf_counter() - started) * 1000:.1f} ms, including this open's checks")
    print(f"file size: {before} bytes before, {file_bytes('grow.db')} bytes after")
    with NeoWriteJournal(path("grow.db")) as journal:
        started = time.perf_counter()
        assert journal.check_integrity().ok
        print(f"check_integrity after migrating: {(time.perf_counter() - started) * 1000:.1f} ms")

    print(f"\ndid the code under test come from the working tree? {v2.__file__}; schema {v2.SCHEMA_VERSION}")
finally:
    shutil.rmtree(ROOT)
print("temporary directory removed:", not os.path.exists(ROOT))
