"""
Runs the real listener source in a subprocess, with nothing real behind it.

The subprocess has its own signal handlers and root logger, so the pytest process
keeps its own. Before wazuh_listener is imported, everything that could reach
Neo4j, MISP or campaign logic is stubbed. Before any of the snippet runs,
ALERT_FILE and offset.dat are pointed into the working directory, which callers
set to a temporary directory. The real AlertFileHandler, start_listener() and
DuplicateFlushWorker do run.

Snippets see `wl` (the listener module) and `processed` (alerts handed to the
stubbed process_alert).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LISTENER_DIR = os.path.join(BACKEND_DIR, "listener")

_PRELUDE = f"""
import logging, os, sys, types
sys.path.insert(0, {LISTENER_DIR!r})
sys.path.insert(0, {BACKEND_DIR!r})

from tests._production_safety import install_child_guards
install_child_guards()

# The real realtime_socgraph calls basicConfig at import. Keep that side effect,
# because the listener's logging setup is written around it.
logging.basicConfig(level=logging.INFO)

class _Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f"isolated listener reached stubbed dependency: {{name}}")

def _forbidden(*args, **kwargs):
    raise AssertionError("isolated listener reached stubbed Neo4j")

def _stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module

processed = []

def _process_alert(alert):
    processed.append(alert)
    return True

_stub("realtime_socgraph", process_alert=_process_alert)
_stub("neo4j_client", batch_update_duplicates=_forbidden, find_recent_duplicate=_forbidden)
_stub("campaign_manager", campaign_manager=_Forbidden())
_stub("operation_manager", operation_manager=_Forbidden())

import wazuh_listener as wl

wl.ALERT_FILE = os.path.join(os.getcwd(), "alerts.json")
wl.__file__ = os.path.join(os.getcwd(), "wazuh_listener.py")
"""


def run_isolated_listener(code: str, cwd: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """cwd must be a temporary directory: logs and offset.dat are written there."""
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + "\n" + textwrap.dedent(code)],
        capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )


def real_offset_state():
    """Size and modification time of the real offset file. Never opens it."""
    from tests._production_safety import REAL_OFFSET_FILE
    try:
        status = os.stat(REAL_OFFSET_FILE)
    except FileNotFoundError:
        return None
    return status.st_size, status.st_mtime_ns
