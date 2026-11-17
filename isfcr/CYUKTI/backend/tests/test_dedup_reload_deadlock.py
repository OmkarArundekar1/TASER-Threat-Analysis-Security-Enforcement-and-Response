"""
Regression tests for the dedup restart/cache-reload self-deadlock.

Defect: DeduplicationEngine.is_duplicate() holds self.lock (a non-reentrant
threading.Lock) and, when the in-memory cache misses but Neo4j still holds a
recent event with the same fingerprint, calls reload_cache(), which acquired
self.lock again. The thread blocked on itself forever, with no exception and
no log output. The in-memory cache is empty after a listener restart, so the
first repeat alert inside the dedup window reached this path.

Every call that could hang runs in a daemon thread with a bounded join, so a
regression fails the test instead of hanging the suite. No test here touches
Neo4j: find_recent_duplicate is replaced at the module boundary.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import dedup_engine as dedup_module
from config import DEDUP_WINDOW

T0 = datetime(2026, 3, 11, 6, 19, 10, tzinfo=timezone.utc)
ARGS = ("192.168.56.106", "pes1ug23cs411-VirtualBox", "T1595.002", "31151", "001")
HANG_TIMEOUT = 5.0


def _bounded(fn, *args, **kwargs):
    """Runs fn in a daemon thread; fails the test if it does not return."""
    box = {}

    def target():
        box["value"] = fn(*args, **kwargs)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(HANG_TIMEOUT)
    assert not thread.is_alive(), f"{getattr(fn, '__name__', fn)} deadlocked (no return in {HANG_TIMEOUT}s)"
    return box["value"]


def _join_all(threads):
    """Joins against one shared deadline, so a total deadlock fails in
    HANG_TIMEOUT seconds rather than HANG_TIMEOUT per thread."""
    deadline = time.monotonic() + HANG_TIMEOUT
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


@pytest.fixture
def db(monkeypatch):
    """Stateful stand-in for neo4j_client.find_recent_duplicate."""
    state = {"event": None, "calls": []}

    def fake(fingerprint, reference_time=None):
        state["calls"].append((fingerprint, reference_time))
        return state["event"]

    monkeypatch.setattr(dedup_module, "find_recent_duplicate", fake)
    return state


@pytest.fixture
def engine():
    return dedup_module.DeduplicationEngine()


RECENT = {"event_id": "evt-recent", "campaign_id": "CAMP_RECENT", "last_seen": None}


# ---------------------------------------------------------------- the deadlock

def test_restart_path_completes_and_reports_a_duplicate(engine, db):
    """THE regression: empty cache (fresh process) + recent event in Neo4j."""
    db["event"] = RECENT

    duplicate, fingerprint = _bounded(engine.is_duplicate, *ARGS, event_time=T0)

    assert duplicate is True
    assert fingerprint == engine.generate_fingerprint(*ARGS)


def test_restart_path_repopulates_the_cache_from_the_database_event(engine, db):
    db["event"] = RECENT

    _, fingerprint = _bounded(engine.is_duplicate, *ARGS, event_time=T0)

    assert engine.get_metadata(fingerprint) == {
        "fingerprint": fingerprint,
        "first_seen": T0.timestamp(),
        "last_seen": T0.timestamp(),
        "occurrences": 1,
        "campaign_id": "CAMP_RECENT",
        "event_id": "evt-recent",
    }


def test_lock_is_released_after_the_reload_path(engine, db):
    db["event"] = RECENT
    _bounded(engine.is_duplicate, *ARGS, event_time=T0)

    assert engine.lock.acquire(blocking=False) is True
    engine.lock.release()


def test_next_repeat_after_reload_is_served_from_the_cache(engine, db):
    db["event"] = RECENT
    _bounded(engine.is_duplicate, *ARGS, event_time=T0)
    db["calls"].clear()

    duplicate, fingerprint = _bounded(engine.is_duplicate, *ARGS, event_time=T0 + timedelta(seconds=1))

    assert duplicate is True
    assert db["calls"] == []
    assert engine.get_metadata(fingerprint)["occurrences"] == 2


# ---------------------------------------------------------------- lock semantics preserved

def test_public_reload_cache_still_takes_the_lock_itself(engine):
    """reload_cache() stays safe for a caller that does not hold the lock: it
    waits while another thread holds it, then completes."""
    fingerprint = engine.generate_fingerprint(*ARGS)
    done = threading.Event()

    def call():
        engine.reload_cache(fingerprint, "evt-1", "CAMP_1", event_time=T0)
        done.set()

    with engine.lock:
        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        assert done.wait(0.2) is False
        assert engine.get_metadata(fingerprint) is None
    assert done.wait(HANG_TIMEOUT) is True
    assert engine.get_metadata(fingerprint)["event_id"] == "evt-1"


def test_engine_lock_is_still_a_plain_non_reentrant_lock(engine):
    """Guards the design choice: nested acquisition must stay a visible bug
    rather than being masked by an RLock."""
    assert isinstance(engine.lock, type(threading.Lock()))


def test_database_lookup_still_runs_under_the_lock(engine, monkeypatch):
    """Check-then-populate stays atomic (and the fix did not change when the
    lookup happens relative to the lock)."""
    observed = []

    def fake(fingerprint, reference_time=None):
        observed.append(engine.lock.locked())
        return RECENT

    monkeypatch.setattr(dedup_module, "find_recent_duplicate", fake)
    _bounded(engine.is_duplicate, *ARGS, event_time=T0)

    assert observed == [True]


# ---------------------------------------------------------------- concurrency

def test_concurrent_repeats_after_restart_all_complete_with_one_database_lookup(engine, db):
    """N workers hit the same cold fingerprint: one reloads, the rest are cache
    hits, every occurrence is counted, nobody hangs."""
    db["event"] = RECENT
    workers = 16
    barrier = threading.Barrier(workers)
    results = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait(HANG_TIMEOUT)
        outcome = engine.is_duplicate(*ARGS, event_time=T0)
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    _join_all(threads)
    assert not any(thread.is_alive() for thread in threads), "deadlock under concurrency"

    fingerprint = engine.generate_fingerprint(*ARGS)
    assert results == [(True, fingerprint)] * workers
    assert len(db["calls"]) == 1
    assert engine.get_metadata(fingerprint)["occurrences"] == workers


def test_reload_path_runs_safely_alongside_register_event_and_other_fingerprints(engine, db):
    db["event"] = RECENT
    errors = []

    def reloader(i):
        try:
            args = ("10.0.0.5", "10.0.0.9", "T1110", str(i), "001")
            assert engine.is_duplicate(*args, event_time=T0)[0] is True
        except Exception as exc:  # surfaced below; a thread cannot fail the test directly
            errors.append(exc)

    def registrar(i):
        try:
            fingerprint = engine.generate_fingerprint("10.0.0.6", "10.0.0.9", "T1110", str(i), "001")
            engine.register_event(fingerprint, campaign_id="C", event_id=f"e{i}", event_time=T0)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reloader, args=(i,), daemon=True) for i in range(20)]
    threads += [threading.Thread(target=registrar, args=(i,), daemon=True) for i in range(20)]
    for thread in threads:
        thread.start()
    _join_all(threads)

    assert not any(thread.is_alive() for thread in threads), "deadlock under concurrency"
    assert errors == []
    assert len(engine.cache) == 40


# ---------------------------------------------------------------- semantics pinned (unchanged)

def test_fingerprint_generation_is_unchanged(engine):
    assert engine.generate_fingerprint("a", "v", "T1", "r", "g") == \
        __import__("hashlib").sha256(b"a|v|T1|r|g").hexdigest()


def test_no_database_event_still_means_not_a_duplicate_and_no_cache_entry(engine, db):
    duplicate, fingerprint = _bounded(engine.is_duplicate, *ARGS, event_time=T0)

    assert duplicate is False
    assert engine.get_metadata(fingerprint) is None


def test_database_lookup_receives_the_alert_event_time(engine, db):
    _bounded(engine.is_duplicate, *ARGS, event_time=T0)
    assert db["calls"] == [(engine.generate_fingerprint(*ARGS), T0)]


def test_cached_entry_past_the_window_still_falls_through_to_the_database(engine, db):
    _, fingerprint = _bounded(engine.is_duplicate, *ARGS, event_time=T0)
    engine.register_event(fingerprint, campaign_id="C", event_id="E", event_time=T0)
    db["calls"].clear()

    later = T0 + timedelta(seconds=DEDUP_WINDOW + 1)
    assert _bounded(engine.is_duplicate, *ARGS, event_time=later)[0] is False
    assert len(db["calls"]) == 1
