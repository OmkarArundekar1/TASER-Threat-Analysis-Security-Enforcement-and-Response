"""
Regression tests for DuplicateBuffer.restore() and DuplicateFlushWorker.

Defects:
- restore() merged a failed older record into a newer one for the same
  fingerprint: older last_seen and payload overwrote the newer ones, and the
  older counters moved onto the newer record's event and campaign.
- After a multi-batch failure, restore() put back batches already written, so
  their additive counters would apply twice.
- stop() used an undefined logger, after the writes and again in its handler.
- A logging or print failure after a successful write triggered restore().
- The worker slept 7 s uninterruptibly while stop() waited 2 s.

No test here touches Neo4j. batch_update_duplicates is replaced at the module
boundary by an autouse fixture that fails if a test has not chosen a writer.
These tests show the buffer's handling only. They do not show how Neo4j applies
the records, and no behaviour here amounts to crash recovery.
"""

from __future__ import annotations

import threading
import time

import pytest

import dedup_engine as dedup_module
from dedup_engine import DuplicateBuffer, DuplicateFlushWorker

HANG_TIMEOUT = 5.0


class _Writer:
    """Fake batch writer. Fails on the listed 1-based call numbers."""

    def __init__(self, fail_on=(), error=None):
        self.fail_on = set(fail_on)
        self.error = error or RuntimeError("write failed")
        self.calls = 0
        self.batches = []

    def __call__(self, batch):
        self.calls += 1
        if self.calls in self.fail_on:
            raise self.error
        self.batches.append([dict(record) for record in batch])

    @property
    def written(self):
        return [record for batch in self.batches for record in batch]


class _BrokenLogger:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise OSError("log handler failed")
        return fail


@pytest.fixture
def buffer(monkeypatch):
    fresh = DuplicateBuffer()
    monkeypatch.setattr(dedup_module, "duplicate_buffer", fresh)
    return fresh


@pytest.fixture(autouse=True)
def _no_real_writer(monkeypatch, buffer):
    def forbidden(batch):
        raise AssertionError("test did not choose a writer")
    monkeypatch.setattr(dedup_module, "batch_update_duplicates", forbidden)


def _use(monkeypatch, writer):
    monkeypatch.setattr(dedup_module, "batch_update_duplicates", writer)
    return writer


def _add(buffer, fingerprint="F", event="E1", campaign="C1", tps=1, payload="p1", timestamp="t1"):
    return buffer.add_duplicate(fingerprint, event, campaign, tps, payload, timestamp=timestamp)


def _record(fingerprint="F", event="E1", campaign="C1", occurrences=1, tps=1, payload="p1", last_seen="t1"):
    return {"fingerprint": fingerprint, "event_id": event, "campaign_id": campaign,
            "occurrences": occurrences, "total_tps": tps, "latest_payload": payload,
            "last_seen": last_seen}


# ---------------------------------------------------------------- merge semantics pinned (D4, D5, D12 open)

def test_merge_keeps_the_first_association_and_the_last_arrival(buffer):
    """Today's behaviour, deliberately unchanged, including that an earlier
    event time arriving later replaces last_seen."""
    _add(buffer, event="E1", campaign="C1", tps=2, payload="p1", timestamp="2026-01-01T00:00:10+00:00")
    _add(buffer, event="E2", campaign="C2", tps=3, payload="p2", timestamp="2026-01-01T00:00:05+00:00")

    assert buffer.snapshot() == [_record(
        occurrences=2, tps=5, payload="p2", last_seen="2026-01-01T00:00:05+00:00")]


def test_snapshot_without_any_failure_is_unchanged(buffer):
    _add(buffer, fingerprint="A")
    _add(buffer, fingerprint="B")

    assert [r["fingerprint"] for r in buffer.snapshot()] == ["A", "B"]
    assert buffer.snapshot() == []
    assert buffer.size() == 0


# ---------------------------------------------------------------- restore, no collision

def test_restore_without_collision_returns_the_record_to_the_buffer(buffer):
    record = _record()

    buffer.restore([record])

    assert buffer.buffer == {"F": record}
    assert buffer.restored == []
    assert buffer.size() == 1


def test_duplicate_after_a_collision_free_restore_merges_as_it_always_did(buffer):
    buffer.restore([_record()])

    _add(buffer, event="E2", campaign="C2", tps=4, payload="p2", timestamp="t2")

    assert buffer.snapshot() == [_record(occurrences=2, tps=5, payload="p2", last_seen="t2")]


# ---------------------------------------------------------------- restore, collision

@pytest.fixture
def collided(buffer):
    """Old record failed its write while a newer one arrived for F."""
    _add(buffer, event="E2", campaign="C2", tps=7, payload="newer", timestamp="t2")
    buffer.restore([_record(event="E1", campaign="C1", occurrences=3, tps=9, payload="older", last_seen="t1")])
    return buffer


NEWER = _record(event="E2", campaign="C2", occurrences=1, tps=7, payload="newer", last_seen="t2")
OLDER = _record(event="E1", campaign="C1", occurrences=3, tps=9, payload="older", last_seen="t1")


def test_collision_preserves_the_newer_timestamp_and_payload(collided):
    assert collided.buffer["F"]["last_seen"] == "t2"
    assert collided.buffer["F"]["latest_payload"] == "newer"


def test_collision_does_not_combine_or_move_counters(collided):
    assert collided.buffer["F"] == NEWER
    assert collided.restored == [OLDER]


def test_collision_keeps_each_records_own_association(collided):
    records = collided.snapshot()
    assert [(r["event_id"], r["campaign_id"]) for r in records] == [("E1", "C1"), ("E2", "C2")]


def test_collision_flushes_older_record_first(collided):
    """Same order as if the first flush had succeeded."""
    assert collided.snapshot() == [OLDER, NEWER]
    assert collided.size() == 0


def test_collision_counts_both_records_in_size(collided):
    assert collided.size() == 2


def test_later_duplicate_merges_only_into_the_newer_record(collided):
    _add(collided, event="E3", campaign="C3", tps=1, payload="latest", timestamp="t3")

    assert collided.restored == [OLDER]
    assert collided.buffer["F"]["occurrences"] == 2
    assert collided.buffer["F"]["event_id"] == "E2"
    assert collided.buffer["F"]["last_seen"] == "t3"


def test_several_restored_records_for_one_fingerprint_keep_their_order(buffer):
    first = _record(payload="first", last_seen="t1")
    second = _record(payload="second", last_seen="t2")

    buffer.restore([first, second])

    assert buffer.snapshot() == [first, second]


# ---------------------------------------------------------------- repeated restoration

def test_restoring_the_same_records_twice_has_no_further_effect(buffer):
    record = _record(occurrences=3, tps=9)

    buffer.restore([record])
    buffer.restore([record])

    assert buffer.snapshot() == [_record(occurrences=3, tps=9)]


def test_restoring_the_same_colliding_record_twice_has_no_further_effect(buffer):
    _add(buffer, event="E2", campaign="C2", tps=7, payload="newer", timestamp="t2")
    record = dict(OLDER)

    buffer.restore([record])
    buffer.restore([record])

    assert buffer.snapshot() == [OLDER, NEWER]


def test_repeated_failed_cycles_lose_nothing_and_do_not_double(buffer, monkeypatch):
    _add(buffer, tps=2, payload="only")
    writer = _use(monkeypatch, _Writer(fail_on=(1, 2, 3)))
    worker = DuplicateFlushWorker()

    for _ in range(4):
        worker._flush_once()

    assert writer.written == [_record(tps=2, payload="only")]
    assert buffer.size() == 0


def test_collisions_across_failed_cycles_keep_every_record_once_in_order(buffer, monkeypatch):
    _use(monkeypatch, _Writer(fail_on=(1, 2)))
    worker = DuplicateFlushWorker()
    _add(buffer, payload="a", timestamp="t1")
    worker._flush_once()  # back in the buffer
    snapshot_then_new = buffer.snapshot()
    _add(buffer, payload="b", timestamp="t2")
    buffer.restore(snapshot_then_new)  # a held, b in the buffer
    worker._flush_once()  # fails; both back, and still ordered

    writer = _use(monkeypatch, _Writer())
    worker._flush_once()

    assert [r["latest_payload"] for r in writer.written] == ["a", "b"]
    assert buffer.size() == 0


# ---------------------------------------------------------------- capacity

def test_restored_records_count_towards_capacity_and_are_never_discarded(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "MAX_PENDING_DUPLICATES", 2)
    _add(buffer, fingerprint="F", event="E2")
    buffer.restore([_record(fingerprint="F", event="E1"), _record(fingerprint="G", event="E9")])
    assert buffer.size() == 3

    assert _add(buffer, fingerprint="H") is False  # existing overflow path takes over

    assert {r["event_id"] for r in buffer.snapshot()} == {"E1", "E2", "E9"}


def test_capacity_rule_is_unchanged_when_nothing_was_restored(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "MAX_PENDING_DUPLICATES", 1)
    assert _add(buffer, fingerprint="F") is True
    assert _add(buffer, fingerprint="F") is True
    assert _add(buffer, fingerprint="G") is False


# ---------------------------------------------------------------- partial writes

def _fill(buffer, count):
    for i in range(count):
        _add(buffer, fingerprint=f"F{i}", event=f"E{i}")


def test_batch_failure_restores_only_unwritten_batches(buffer, monkeypatch):
    """450 records, batch size 200, third batch fails: 400 persisted, 50 back."""
    monkeypatch.setattr(dedup_module, "MAX_BATCH_SIZE", 200)
    _fill(buffer, 450)
    writer = _use(monkeypatch, _Writer(fail_on=(3,)))

    DuplicateFlushWorker()._flush_once()

    assert len(writer.written) == 400
    assert set(buffer.buffer) == {f"F{i}" for i in range(400, 450)}


def test_retry_after_partial_failure_writes_each_record_exactly_once(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "MAX_BATCH_SIZE", 200)
    _fill(buffer, 450)
    writer = _use(monkeypatch, _Writer(fail_on=(2,)))
    worker = DuplicateFlushWorker()

    worker._flush_once()
    worker._flush_once()

    assert sorted(r["fingerprint"] for r in writer.written) == sorted(f"F{i}" for i in range(450))


def test_first_batch_failure_restores_everything(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "MAX_BATCH_SIZE", 200)
    _fill(buffer, 250)
    _use(monkeypatch, _Writer(fail_on=(1,)))

    DuplicateFlushWorker()._flush_once()

    assert buffer.size() == 250


# ---------------------------------------------------------------- worker: logging failure after success

def test_worker_logging_failure_after_success_does_not_restore(buffer, monkeypatch):
    _add(buffer)
    writer = _use(monkeypatch, _Writer())
    monkeypatch.setattr(dedup_module, "logger", _BrokenLogger())

    DuplicateFlushWorker()._flush_once()

    assert len(writer.written) == 1
    assert buffer.size() == 0


def test_worker_logs_a_failed_write_with_the_original_exception(buffer, monkeypatch, caplog):
    _add(buffer)
    error = RuntimeError("original")
    _use(monkeypatch, _Writer(fail_on=(1,), error=error))

    with caplog.at_level("ERROR", logger="dedup_engine"):
        DuplicateFlushWorker()._flush_once()

    assert caplog.records[-1].exc_info[1] is error


# ---------------------------------------------------------------- stop(): normal paths

def test_stop_with_empty_buffer_and_no_worker(monkeypatch):
    writer = _use(monkeypatch, _Writer())

    result = DuplicateFlushWorker().stop()

    assert result.complete
    assert result.written == 0
    assert writer.calls == 0


def test_stop_with_pending_records_writes_once_and_empties_the_buffer(buffer, monkeypatch, caplog):
    _add(buffer)
    writer = _use(monkeypatch, _Writer())

    with caplog.at_level("INFO", logger="dedup_engine"):
        result = DuplicateFlushWorker().stop()

    assert result.complete
    assert result.written == 1
    assert len(writer.written) == 1
    assert buffer.size() == 0
    assert "Final flush completed (1 records)." in caplog.text


def test_stop_module_defines_a_real_logger():
    """The original defect: `logger` did not exist in this module."""
    assert dedup_module.logger.name == "dedup_engine"


def test_stop_logging_failure_after_success_neither_raises_nor_restores(buffer, monkeypatch):
    _add(buffer)
    writer = _use(monkeypatch, _Writer())
    monkeypatch.setattr(dedup_module, "logger", _BrokenLogger())

    result = DuplicateFlushWorker().stop()

    assert result.complete
    assert len(writer.written) == 1
    assert buffer.size() == 0


# ---------------------------------------------------------------- stop(): persistence failure

def test_stop_persistence_failure_keeps_records_and_the_original_exception(buffer, monkeypatch, caplog):
    _add(buffer)
    error = RuntimeError("original")
    _use(monkeypatch, _Writer(fail_on=(1,), error=error))

    with caplog.at_level("ERROR", logger="dedup_engine"):
        result = DuplicateFlushWorker().stop()

    assert not result.complete
    assert result.error is error
    assert result.unflushed == 1
    assert buffer.size() == 1
    assert caplog.records[-1].exc_info[1] is error
    assert "lost if the process exits" in caplog.text


def test_stop_persistence_failure_with_broken_logging_still_reports_the_original(buffer, monkeypatch):
    _add(buffer)
    error = RuntimeError("original")
    _use(monkeypatch, _Writer(fail_on=(1,), error=error))
    monkeypatch.setattr(dedup_module, "logger", _BrokenLogger())

    result = DuplicateFlushWorker().stop()

    assert result.error is error
    assert buffer.size() == 1


def test_stop_partial_failure_restores_only_the_unwritten_part(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "MAX_BATCH_SIZE", 200)
    _fill(buffer, 250)
    _use(monkeypatch, _Writer(fail_on=(2,)))

    result = DuplicateFlushWorker().stop()

    assert result.written == 200
    assert result.unflushed == 50
    assert buffer.size() == 50


# ---------------------------------------------------------------- stop(): repeated, and worker states

def test_repeated_stop_does_not_write_again(buffer, monkeypatch):
    _add(buffer)
    writer = _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()

    first, second = worker.stop(), worker.stop()

    assert first.written == 1
    assert second.complete and second.written == 0
    assert writer.calls == 1


def test_second_stop_retries_records_the_first_could_not_write(buffer, monkeypatch):
    _add(buffer)
    writer = _use(monkeypatch, _Writer(fail_on=(1,)))
    worker = DuplicateFlushWorker()

    assert worker.stop().unflushed == 1
    assert worker.stop().written == 1
    assert len(writer.written) == 1


def test_stop_when_worker_already_stopped(buffer, monkeypatch):
    _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()
    worker.start()
    assert worker.stop().complete
    assert not worker.thread.is_alive()

    assert worker.stop().complete


def test_stop_while_worker_sleeping_returns_promptly(buffer, monkeypatch):
    """Uses the real 7 s interval. stop() used to wait out its full 2 s join."""
    monkeypatch.setattr(dedup_module, "DUPLICATE_FLUSH_INTERVAL", 7)
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()
    worker.start()
    time.sleep(0.05)

    started = time.monotonic()
    result = worker.stop()
    elapsed = time.monotonic() - started

    assert result.complete
    assert elapsed < 1.0
    assert not worker.thread.is_alive()


def test_stop_flushes_records_still_buffered_when_the_worker_was_sleeping(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    writer = _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()
    worker.start()
    _add(buffer)

    assert worker.stop().written == 1
    assert len(writer.written) == 1


def test_periodic_flush_still_runs_on_the_interval(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "DUPLICATE_FLUSH_INTERVAL", 0.02)
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    writer = _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()
    worker.start()
    _add(buffer)

    deadline = time.monotonic() + HANG_TIMEOUT
    while not writer.written and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()

    assert len(writer.written) == 1


def test_periodic_flush_does_not_fire_before_the_interval(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "DUPLICATE_FLUSH_INTERVAL", 7)
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    writer = _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()
    worker.start()
    _add(buffer)
    time.sleep(0.1)

    assert writer.calls == 0
    worker.stop()


def test_worker_can_restart_after_stop(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "DUPLICATE_FLUSH_INTERVAL", 0.02)
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    writer = _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()
    worker.start()
    worker.stop()

    worker.start()
    _add(buffer)
    deadline = time.monotonic() + HANG_TIMEOUT
    while not writer.written and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()

    assert len(writer.written) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_stop_after_the_worker_thread_died_still_flushes(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "DUPLICATE_FLUSH_INTERVAL", 0.01)
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    writer = _use(monkeypatch, _Writer())
    worker = DuplicateFlushWorker()

    def die():
        raise RuntimeError("worker failed")

    monkeypatch.setattr(worker, "_flush_once", die)
    worker.start()
    worker.thread.join(HANG_TIMEOUT)
    assert not worker.thread.is_alive()
    _add(buffer)

    result = worker.stop()

    assert result.complete
    assert len(writer.written) == 1


# ---------------------------------------------------------------- stop(): bounds

class _BlockingWriter:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.batches = []

    def __call__(self, batch):
        self.entered.set()
        self.release.wait(HANG_TIMEOUT)
        self.batches.append(batch)


def test_worker_stuck_in_a_write_bounds_stop_and_reports_the_records(buffer, monkeypatch, caplog):
    monkeypatch.setattr(dedup_module, "DUPLICATE_FLUSH_INTERVAL", 0.01)
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    monkeypatch.setattr(dedup_module, "WORKER_STOP_TIMEOUT", 0.1)
    blocking = _use(monkeypatch, _BlockingWriter())
    worker = DuplicateFlushWorker()
    _add(buffer)
    worker.start()
    assert blocking.entered.wait(HANG_TIMEOUT)

    try:
        started = time.monotonic()
        with caplog.at_level("ERROR", logger="dedup_engine"):
            result = worker.stop()
        elapsed = time.monotonic() - started

        assert elapsed < 2.0
        assert result.worker_stopped is False
        assert result.unresolved == 1
        assert not result.complete
        assert "did not stop" in caplog.text
    finally:
        blocking.release.set()
        worker.thread.join(HANG_TIMEOUT)


def test_records_buffered_while_worker_is_stuck_are_still_flushed(buffer, monkeypatch):
    monkeypatch.setattr(dedup_module, "DUPLICATE_FLUSH_INTERVAL", 0.01)
    monkeypatch.setattr(dedup_module, "ENABLE_DUPLICATE_BUFFER", True)
    monkeypatch.setattr(dedup_module, "WORKER_STOP_TIMEOUT", 0.1)
    blocking = _BlockingWriter()
    written = []

    def writer(batch):
        if batch[0]["fingerprint"] == "stuck":
            blocking(batch)
        else:
            written.extend(batch)

    _use(monkeypatch, writer)
    worker = DuplicateFlushWorker()
    _add(buffer, fingerprint="stuck")
    worker.start()
    assert blocking.entered.wait(HANG_TIMEOUT)
    _add(buffer, fingerprint="later")

    try:
        result = worker.stop()

        assert [r["fingerprint"] for r in written] == ["later"]
        assert result.written == 1
        assert result.unresolved == 1
    finally:
        blocking.release.set()
        worker.thread.join(HANG_TIMEOUT)


def test_final_flush_that_hangs_bounds_stop_and_reports_the_records(buffer, monkeypatch, caplog):
    monkeypatch.setattr(dedup_module, "FINAL_FLUSH_TIMEOUT", 0.1)
    blocking = _use(monkeypatch, _BlockingWriter())
    _add(buffer)

    try:
        started = time.monotonic()
        with caplog.at_level("ERROR", logger="dedup_engine"):
            result = DuplicateFlushWorker().stop()
        elapsed = time.monotonic() - started

        assert elapsed < 2.0
        assert result.unresolved == 1
        assert not result.complete
        assert "did not finish" in caplog.text
    finally:
        blocking.release.set()
