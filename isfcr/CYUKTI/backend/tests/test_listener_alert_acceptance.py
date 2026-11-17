"""
Regression tests for alert acceptance in listener/wazuh_listener.py.

Defect: the alert id was recorded as seen before put_nowait(). On a full queue
the offset stayed put, the next poll re-read the line, found the id already
seen, ignored it as a raw duplicate and advanced past it, losing the alert.

Isolation: the real listener source is loaded from its file, but everything it
imports that could reach Neo4j, MISP or campaign logic is a stub. ALERT_FILE,
offset.dat and the log file all point into tmp_path. start_listener() is never
called, so the real /var/ossec alert file is never opened.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import signal
import sys
import time
import types
from queue import Empty, Full, Queue

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LISTENER_DIR = os.path.join(BACKEND_DIR, "listener")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def listener(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("listener test reached a stubbed dependency")

    stubs = {
        "realtime_socgraph": {"process_alert": forbidden},
        "dedup_engine": {"duplicate_flush_worker": object(), "duplicate_buffer": object()},
        "neo4j_client": {"batch_update_duplicates": forbidden},
        "campaign_manager": {"campaign_manager": object()},
        "operation_manager": {"operation_manager": object()},
        "event_time": {"event_clock": object()},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "utils", _load("listener_utils_under_test", os.path.join(LISTENER_DIR, "utils.py")))

    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level

    module = _load("wazuh_listener_under_test", os.path.join(LISTENER_DIR, "wazuh_listener.py"))

    module.__file__ = str(tmp_path / "wazuh_listener.py")  # offset.dat lands in tmp_path
    module.ALERT_FILE = str(tmp_path / "alerts.json")
    module.alert_queue = Queue(maxsize=1)
    (tmp_path / "alerts.json").write_text("")
    yield module

    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def _alert(n):
    return {"timestamp": f"2026-03-11T00:00:{n:02d}.000+0000", "rule": {"id": "100"},
            "agent": {"id": "001"}, "location": "test",
            "id": f"1773187200.{n}", "manager": {"name": "test-manager"}}


def _line(n):
    return json.dumps(_alert(n)) + "\n"


def _write(listener, *lines):
    with open(listener.ALERT_FILE, "a") as handle:
        handle.writelines(lines)


def _handler(listener):
    handler = listener.AlertFileHandler()
    handler.offset = 0
    return handler


def _saved_offset(tmp_path):
    return int((tmp_path / "offset.dat").read_text())


def _drain(queue):
    alerts = []
    while True:
        try:
            alerts.append(queue.get_nowait())
        except Empty:
            return alerts


# ---------------------------------------------------------------- successful insertion

def test_successful_insertion_queues_records_and_advances(listener, tmp_path):
    listener.alert_queue = Queue(maxsize=10)
    _write(listener, _line(1), _line(2))
    handler = _handler(listener)

    handler.read_new_lines()

    queued = _drain(listener.alert_queue)
    assert [a["timestamp"] for a in queued] == [_alert(1)["timestamp"], _alert(2)["timestamp"]]
    assert set(listener.seen_alerts) == {a["alert_id"] for a in queued}
    assert handler.offset == os.path.getsize(listener.ALERT_FILE)
    assert _saved_offset(tmp_path) == handler.offset


# ---------------------------------------------------------------- queue full

def test_queue_full_does_not_mark_the_alert_seen_or_advance_past_it(listener, tmp_path):
    _write(listener, _line(1), _line(2))
    handler = _handler(listener)

    handler.read_new_lines()

    assert listener.alert_queue.qsize() == 1
    assert len(listener.seen_alerts) == 1
    assert handler.offset == len(_line(1))
    assert _saved_offset(tmp_path) == len(_line(1))


def test_queue_full_retry_is_rejected_while_still_full_and_nothing_is_lost(listener):
    _write(listener, _line(1), _line(2))
    handler = _handler(listener)

    handler.read_new_lines()
    handler.read_new_lines()
    handler.read_new_lines()

    assert listener.alert_queue.qsize() == 1
    assert handler.offset == len(_line(1))


def test_retry_after_capacity_returns_accepts_the_alert(listener):
    """THE regression: before the fix the retried alert was ignored as a raw
    duplicate and the offset moved past it."""
    _write(listener, _line(1), _line(2), _line(3))
    handler = _handler(listener)
    received = []

    for _ in range(5):
        handler.read_new_lines()
        received += _drain(listener.alert_queue)

    assert [a["timestamp"] for a in received] == [_alert(n)["timestamp"] for n in (1, 2, 3)]
    assert handler.offset == os.path.getsize(listener.ALERT_FILE)


def test_failed_insertions_do_not_evict_accepted_ids(listener):
    """A rejected attempt must not consume seen-cache capacity, or repeated
    retries would push accepted ids out and weaken duplicate suppression."""
    listener.MAX_ALERT_CACHE = 1
    accepted = {**_alert(1), "alert_id": "accepted"}
    assert listener.accept_alert(accepted) is True

    for _ in range(3):
        with pytest.raises(Full):
            listener.accept_alert({**_alert(2), "alert_id": "rejected"})

    assert list(listener.seen_alerts) == ["accepted"]


# ---------------------------------------------------------------- repeated delivery

def test_repeated_delivery_of_an_accepted_alert_is_suppressed(listener, tmp_path):
    listener.alert_queue = Queue(maxsize=10)
    _write(listener, _line(1), _line(1))
    handler = _handler(listener)

    handler.read_new_lines()

    assert listener.alert_queue.qsize() == 1
    assert handler.offset == os.path.getsize(listener.ALERT_FILE)
    assert _saved_offset(tmp_path) == handler.offset


def test_repeat_is_suppressed_even_when_the_queue_is_full(listener):
    """A repeat needs no queue slot, so it must not stall the reader."""
    _write(listener, _line(1), _line(1))
    handler = _handler(listener)

    handler.read_new_lines()

    assert listener.alert_queue.qsize() == 1
    assert handler.offset == os.path.getsize(listener.ALERT_FILE)


def test_accepted_alert_stays_suppressed_after_it_leaves_the_queue(listener):
    alert = {**_alert(1), "alert_id": "a"}
    assert listener.accept_alert(alert) is True
    _drain(listener.alert_queue)

    assert listener.accept_alert(alert) is False
    assert listener.alert_queue.empty()


def test_suppression_lifetime_is_unchanged(listener):
    alert = {**_alert(1), "alert_id": "a"}
    assert listener.accept_alert(alert) is True
    _drain(listener.alert_queue)
    listener.seen_alerts["a"] = time.time() - listener.ALERT_CACHE_TTL - 1

    assert listener.accept_alert(alert) is True


# ---------------------------------------------------------------- insertion exceptions

class _FailingQueue:
    def __init__(self, error):
        self.error = error

    def put_nowait(self, alert):
        raise self.error


def test_insertion_exception_propagates_without_marking_seen(listener):
    listener.alert_queue = _FailingQueue(RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        listener.accept_alert({**_alert(1), "alert_id": "a"})

    assert "a" not in listener.seen_alerts


def test_insertion_exception_keeps_the_offset_and_the_alert_is_retried(listener, tmp_path):
    _write(listener, _line(1))
    handler = _handler(listener)
    listener.alert_queue = _FailingQueue(RuntimeError("boom"))

    handler.read_new_lines()

    assert handler.offset == 0
    assert _saved_offset(tmp_path) == 0
    assert listener.seen_alerts == {}

    listener.alert_queue = Queue(maxsize=1)
    handler.read_new_lines()

    assert listener.alert_queue.qsize() == 1
    assert handler.offset == os.path.getsize(listener.ALERT_FILE)


def test_lock_is_released_after_a_failed_insertion(listener):
    listener.alert_queue = _FailingQueue(Full())
    with pytest.raises(Full):
        listener.accept_alert({**_alert(1), "alert_id": "a"})

    assert listener.seen_alerts_lock.acquire(blocking=False) is True
    listener.seen_alerts_lock.release()


# ---------------------------------------------------------------- behaviour left alone

def test_malformed_line_is_still_skipped(listener):
    listener.alert_queue = Queue(maxsize=10)
    _write(listener, "{not json}\n", _line(1))
    handler = _handler(listener)

    handler.read_new_lines()

    assert listener.alert_queue.qsize() == 1
    assert handler.offset == os.path.getsize(listener.ALERT_FILE)
