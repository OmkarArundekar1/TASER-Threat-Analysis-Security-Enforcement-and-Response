"""
Regression tests for listener/wazuh_listener.py's audit-file logging.

Root cause (fixed by this session): wazuh_listener.py imports
realtime_socgraph (line 15, `from realtime_socgraph import process_alert`)
BEFORE its own logging setup ran. realtime_socgraph.py calls
logging.basicConfig(level=INFO, format=...) at ITS OWN module level with
no `handlers=` argument, which attaches a default StreamHandler to the
root logger as a side effect of that import. logging.basicConfig() is
documented to do nothing once the root logger already has any handler --
so wazuh_listener.py's original
`logging.basicConfig(handlers=[FileHandler(LOG_FILE), StreamHandler()])`
call silently did nothing: the FileHandler it constructed was discarded,
never attached to any logger, anywhere. Console/stdout logging kept
working throughout (every logger in the process propagates to root by
default, and root already had realtime_socgraph's StreamHandler) -- which
is exactly why this looked like "logging works, but the file stays
empty" rather than crashing outright. Not buffering, not permissions,
not a wrong path: a handler that was never wired up.

The fix (`wazuh_listener._configure_logging()`) attaches the missing
FileHandler directly to the root logger, idempotently (by inspecting
existing handlers rather than a one-shot flag), instead of relying on
logging.basicConfig()'s existence check.

These tests run the real, unmodified listener module in fresh
subprocesses. Root-logger handler state is a process-global side effect
of import order, so a fresh interpreter is the only way to get a clean,
deterministic answer to "did the file handler actually get attached and
receive real bytes" -- an in-process assertion after other test modules
have already imported logging-touching code (many do) would be testing
whatever handler soup happens to already exist, not this fix. Every test
points the log file at a pytest tmp_path, never the developer's cwd or
the real backend/logs/prerana_listener.log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LISTENER_DIR = os.path.join(BACKEND_DIR, "listener")

_PATH_PRELUDE = f"""
import sys
sys.path.insert(0, {LISTENER_DIR!r})
sys.path.insert(0, {BACKEND_DIR!r})
"""


def _run(code: str, cwd: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """Runs `code` in a fresh Python process with cwd set to an isolated
    temp directory -- LOG_DIR/LOG_FILE in wazuh_listener.py are relative
    paths resolved against cwd at import time, so this is what actually
    isolates each test's log file from the real one and from every other
    test, without needing to monkeypatch module globals mid-run.

    `code` is dedented on its own before the (already flush-left)
    sys.path prelude is prepended -- dedenting the two concatenated
    pieces together would find no common leading whitespace (the
    prelude has none) and leave the caller's indented block untouched,
    producing an IndentationError in the subprocess.
    """
    full_code = _PATH_PRELUDE + "\n" + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )


def _log_path(cwd: str) -> str:
    return os.path.join(cwd, "logs", "prerana_listener.log")


# ---------------------------------------------------------------- basic persistence

def test_audit_event_is_actually_written_to_the_log_file(tmp_path):
    result = _run(
        """
        import wazuh_listener as wl
        wl.logger.info("AUDIT_EVENT_BASIC_PERSISTENCE_CHECK")
        """,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    log_path = _log_path(str(tmp_path))
    assert os.path.exists(log_path), "logging setup must create the log file"
    content = open(log_path, encoding="utf-8").read()
    assert "AUDIT_EVENT_BASIC_PERSISTENCE_CHECK" in content
    assert " | INFO | " in content  # existing format string preserved


# ---------------------------------------------------------------- stdout preservation

def test_stdout_logging_still_works_alongside_file_logging(tmp_path):
    result = _run(
        """
        import wazuh_listener as wl
        wl.logger.info("AUDIT_EVENT_STDOUT_CHECK")
        """,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    # basicConfig()'s default StreamHandler writes to stderr, not stdout --
    # the fix must not have removed console visibility either way.
    console_output = result.stdout + result.stderr
    assert "AUDIT_EVENT_STDOUT_CHECK" in console_output

    log_path = _log_path(str(tmp_path))
    assert "AUDIT_EVENT_STDOUT_CHECK" in open(log_path, encoding="utf-8").read()


# ---------------------------------------------------------------- repeated initialization

def test_repeated_configure_logging_does_not_duplicate_handlers_or_lines(tmp_path):
    result = _run(
        """
        import logging
        import wazuh_listener as wl

        handler_counts = [len(logging.getLogger().handlers)]
        for _ in range(4):
            wl._configure_logging()
            handler_counts.append(len(logging.getLogger().handlers))

        assert len(set(handler_counts)) == 1, (
            f"handler count changed across repeated _configure_logging() calls: {handler_counts}"
        )

        file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1, f"expected exactly one FileHandler, found {len(file_handlers)}"

        wl.logger.info("AUDIT_EVENT_DUPLICATE_INIT_CHECK")
        """,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    content = open(_log_path(str(tmp_path)), encoding="utf-8").read()
    assert content.count("AUDIT_EVENT_DUPLICATE_INIT_CHECK") == 1, (
        f"expected exactly one occurrence, got {content.count('AUDIT_EVENT_DUPLICATE_INIT_CHECK')}:\n{content}"
    )


def test_reimporting_via_fresh_interpreter_does_not_duplicate_prior_content(tmp_path):
    """A second, independent process logging into the SAME file (the
    realistic "listener restarted" scenario) must append, not duplicate
    or clobber, what a prior process already wrote."""
    first = _run(
        """
        import wazuh_listener as wl
        wl.logger.info("AUDIT_EVENT_FIRST_PROCESS")
        """,
        cwd=str(tmp_path),
    )
    assert first.returncode == 0, first.stderr

    second = _run(
        """
        import wazuh_listener as wl
        wl.logger.info("AUDIT_EVENT_SECOND_PROCESS")
        """,
        cwd=str(tmp_path),
    )
    assert second.returncode == 0, second.stderr

    content = open(_log_path(str(tmp_path)), encoding="utf-8").read()
    assert content.count("AUDIT_EVENT_FIRST_PROCESS") == 1
    assert content.count("AUDIT_EVENT_SECOND_PROCESS") == 1


# ---------------------------------------------------------------- lifecycle / shutdown

def test_events_persist_across_the_real_start_listener_shutdown_lifecycle(tmp_path):
    """Drives the REAL, unmodified start_listener() through its actual
    stop_event-based graceful-shutdown path (the same mechanism
    shutdown_handler/SIGTERM uses) and confirms both pre-shutdown and
    the shutdown-triggered log lines themselves ("Listener stopped.")
    reach the file -- i.e. the finally block's cleanup does not leave
    anything silently unwritten. Driven via stop_event directly, in a
    background thread, rather than an OS signal, since real interpreter
    shutdown/signal-delivery timing varies across platforms and isn't
    what's under test here -- the audit-logging behavior is.
    """
    result = _run(
        """
        import threading, time
        import wazuh_listener as wl

        def trigger_stop():
            time.sleep(1.5)
            wl.stop_event.set()

        threading.Thread(target=trigger_stop, daemon=True).start()
        wl.start_listener()
        print("START_LISTENER_RETURNED", flush=True)
        """,
        cwd=str(tmp_path),
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "START_LISTENER_RETURNED" in result.stdout, (
        "start_listener() must return once stop_event is set, not hang"
    )

    content = open(_log_path(str(tmp_path)), encoding="utf-8").read()
    assert "Prerana Listener initialized." in content
    assert "Queue worker started." in content
    assert "Listener stopped." in content, (
        "the shutdown path's own log line must reach the file, not just startup lines"
    )


# ---------------------------------------------------------------- realistic listener path -> audit event -> file

def test_real_listener_startup_path_produces_audit_events_that_reach_the_file(tmp_path):
    """The smallest realistic listener path that produces genuine audit
    events without live Wazuh/Neo4j: start_listener()'s own real
    lifecycle logging (offset detection, worker startup, polling
    announcement) -- these are real calls through the real logger this
    module configures, not simulated ones."""
    result = _run(
        """
        import threading, time
        import wazuh_listener as wl

        def trigger_stop():
            time.sleep(1.0)
            wl.stop_event.set()

        threading.Thread(target=trigger_stop, daemon=True).start()
        wl.start_listener()
        """,
        cwd=str(tmp_path),
        timeout=20,
    )
    assert result.returncode == 0, result.stderr

    content = open(_log_path(str(tmp_path)), encoding="utf-8").read()
    expected_lines = [
        "Prerana Listener initialized.",
        "Watching :",
        "Initial offset =",
        "Queue worker started.",
        "Maintenance worker started.",
        "Polling started.",
        "Waiting for new Wazuh alerts...",
    ]
    missing = [line for line in expected_lines if line not in content]
    assert not missing, f"expected real lifecycle audit lines missing from file: {missing}\ngot:\n{content}"
