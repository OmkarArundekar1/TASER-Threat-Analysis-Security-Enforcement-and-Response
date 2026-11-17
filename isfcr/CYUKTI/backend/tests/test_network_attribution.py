"""
Tests for how the network guard attributes blocked attempts to tests.

Record-building tests call NetworkGuard._record() directly on an instance that was
never installed, so they use no socket at all. The two tests that need a real blocked
attempt use port 9 on loopback, never a port a production service uses.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import _production_safety as safety
from tests._production_safety import (
    DELIBERATE_NETWORK,
    NET_LOG,
    ROOT_PID,
    UNKNOWN_TEST,
    NetworkGuard,
)

BACKEND_DIR = Path(safety.BACKEND_DIR)


def _record(target="127.0.0.1:1", kind="connect", log_path=None):
    guard = NetworkGuard(log_path=log_path)
    guard._record(kind, target)
    return guard.records[0], guard


# ---------------------------------------------------------------- test identity

def test_record_names_the_running_test_and_phase():
    record, _ = _record()
    assert record["test"].endswith("test_network_attribution.py::test_record_names_the_running_test_and_phase")
    assert record["phase"] == "call"


def test_no_current_test_is_recorded_as_unknown(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    record, _ = _record()
    assert record["test"] == UNKNOWN_TEST
    assert record["phase"] is None


@pytest.mark.parametrize("parameter", [
    "NEO4J_URI-bolt://user:hunter2@localhost:7687",
    "user:hunter2@localhost:7687",
    "hunter2",
])
def test_parameter_values_are_dropped_from_the_test_id(monkeypatch, parameter):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"tests/a.py::test_b[{parameter}] (call)")
    record, _ = _record()
    assert record["test"] == "tests/a.py::test_b[<param>]"
    assert record["phase"] == "call"
    assert "hunter2" not in json.dumps(record)


# ---------------------------------------------------------------- deliberate

def test_an_attempt_is_not_deliberate_just_because_of_this_module():
    record, _ = _record()
    assert record["deliberate"] is False


def test_deliberate_requires_the_fixture(deliberate_network):
    record, _ = _record()
    assert record["deliberate"] is True


def test_the_deliberate_label_ends_with_the_test_that_asked_for_it():
    assert DELIBERATE_NETWORK not in os.environ


# ---------------------------------------------------------------- operation and destination

@pytest.mark.parametrize("kind", ["connect", "connect_ex", "sendto", "resolve"])
def test_operation_type_is_recorded(kind):
    record, _ = _record(kind=kind)
    assert record["operation"] == kind


def test_connect_ex_is_recorded_as_its_own_operation(deliberate_network):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect_ex(("127.0.0.1", 9))
    assert safety.ACTIVE.network.records[-1]["operation"] == "connect_ex"


@pytest.mark.parametrize("target", [
    "127.0.0.1:7687", "localhost:8443", "::1:7687", "misp.invalid.example",
])
def test_plain_destinations_are_kept(target):
    assert _record(target)[0]["destination"] == target


@pytest.mark.parametrize("target", [
    "user:hunter2@localhost:7687",
    "https://shuffler.io/hooks/webhook-id",
    "bolt://neo4j:hunter2@host:7687",
    "host name with spaces",
    "",
])
def test_anything_else_as_a_destination_is_not_recorded(target):
    record, guard = _record(target)
    assert record["destination"] == "<unrecordable>"
    assert guard.attempts == [("connect", "<unrecordable>")]
    assert "hunter2" not in json.dumps(record)
    assert "webhook-id" not in json.dumps(record)


# ---------------------------------------------------------------- log file

def test_log_gets_one_json_line_per_attempt(tmp_path):
    log = tmp_path / "net.log"
    guard = NetworkGuard(log_path=str(log))
    guard._record("connect", "127.0.0.1:1")
    guard._record("resolve", "misp.invalid.example")

    lines = [json.loads(line) for line in log.read_text().splitlines()]

    assert [line["operation"] for line in lines] == ["connect", "resolve"]
    assert set(lines[0]) == {"test", "phase", "operation", "destination", "deliberate", "child", "thread"}


# ---------------------------------------------------------------- thread

def test_a_background_thread_is_visible_in_the_record():
    import threading

    box = {}

    def worker():
        box["record"] = _record()[0]

    thread = threading.Thread(target=worker, name="leftover-worker")
    thread.start()
    thread.join()
    assert box["record"]["thread"] == "leftover-worker"
    assert _record()[0]["thread"] == "MainThread"


# ---------------------------------------------------------------- child processes

def _child(tmp_path, env):
    code = textwrap.dedent(f"""
        import socket, sys
        sys.path.insert(0, {str(BACKEND_DIR)!r})
        from tests._production_safety import install_child_guards, BlockedConnection
        install_child_guards()
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=0.2)
        except BlockedConnection:
            print("BLOCKED")
    """)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(tmp_path), timeout=60, env=env,
    )


def test_child_attempt_is_attributed_to_the_parents_test(tmp_path, deliberate_network):
    log = tmp_path / "net.log"
    result = _child(tmp_path, {**os.environ, NET_LOG: str(log)})

    assert "BLOCKED" in result.stdout, result.stderr
    record = json.loads(log.read_text())
    assert record["test"].endswith("test_child_attempt_is_attributed_to_the_parents_test")
    assert record["child"] is True
    assert record["deliberate"] is True


def test_child_started_without_test_context_is_unknown_not_invented(tmp_path, deliberate_network):
    log = tmp_path / "net.log"
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_CURRENT_TEST"}
    result = _child(tmp_path, {**env, NET_LOG: str(log)})

    assert "BLOCKED" in result.stdout, result.stderr
    assert json.loads(log.read_text())["test"] == UNKNOWN_TEST


def test_a_process_without_an_inherited_root_is_not_called_a_child(tmp_path, deliberate_network):
    log = tmp_path / "net.log"
    env = {key: value for key, value in os.environ.items() if key != ROOT_PID}
    result = _child(tmp_path, {**env, NET_LOG: str(log)})

    assert "BLOCKED" in result.stdout, result.stderr
    assert json.loads(log.read_text())["child"] is False


def test_the_pytest_process_itself_is_not_a_child():
    assert _record()[0]["child"] is (os.environ[ROOT_PID] != str(os.getpid()))


# ---------------------------------------------------------------- guard not weakened

def test_blocked_attempts_are_still_blocked_after_being_recorded(deliberate_network):
    before = len(safety.ACTIVE.network.records)
    with pytest.raises(safety.BlockedConnection):
        socket.create_connection(("127.0.0.1", 9), timeout=0.2)
    assert len(safety.ACTIVE.network.records) == before + 1
