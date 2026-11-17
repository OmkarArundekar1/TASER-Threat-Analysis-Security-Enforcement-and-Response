import os
import signal
import sys

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Must stay above every backend import: backend config reads the environment and
# .env when imported. See tests/_production_safety.py for what this does and does
# not cover.
from tests._production_safety import (
    DELIBERATE_NETWORK,
    REAL_LISTENER_LOG,
    UnsafeConfiguration,
    activate,
    verify_disposable_marker,
    verify_resolved_settings,
)

_UNSAFE_CONFIGURATION = None
try:
    SAFETY = activate()
except UnsafeConfiguration as error:
    SAFETY = None
    _UNSAFE_CONFIGURATION = error


def pytest_configure(config):
    if _UNSAFE_CONFIGURATION is not None:
        raise pytest.UsageError(str(_UNSAFE_CONFIGURATION))


def pytest_sessionstart(session):
    import config as backend_config
    try:
        verify_resolved_settings(backend_config, SAFETY.config)
    except UnsafeConfiguration as error:
        raise pytest.UsageError(str(error))


def _redirect_dashboard_listener_log():
    """dashboard_api reads this path for its status and audit routes, including
    through os.stat, which the path guard cannot see. Tests that need a log already
    patch it to a temporary file; that patch is applied on top of this."""
    dashboard_api = sys.modules.get("dashboard_api")
    if dashboard_api is not None and dashboard_api.LISTENER_LOG_PATH == REAL_LISTENER_LOG:
        dashboard_api.LISTENER_LOG_PATH = os.path.join(SAFETY.scratch_dir, "no_listener_log.log")


def pytest_collection_finish(session):
    _redirect_dashboard_listener_log()


def pytest_terminal_summary(terminalreporter):
    if SAFETY is None:
        return
    records = SAFETY.network.records
    if records:
        terminalreporter.section("network attempts blocked by the test guard")
        lines = {
            f"{'deliberate' if r['deliberate'] else 'UNEXPECTED'} {r['operation']} "
            f"{r['destination']} <- {r['test']}"
            for r in records
        }
        for line in sorted(lines):
            terminalreporter.line(line)
    if SAFETY.paths.violations:
        terminalreporter.section("protected resources refused by the test guard")
        for violation in SAFETY.paths.violations:
            terminalreporter.line(violation)


@pytest.fixture(autouse=True)
def _production_safety_checks():
    """Fails a test that reached a protected resource, or that left this process
    with different signal handlers. The guard refuses the access itself, but code
    under test may swallow the exception."""
    _redirect_dashboard_listener_log()
    handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    violations_before = len(SAFETY.paths.violations)
    yield
    new_violations = SAFETY.paths.violations[violations_before:]
    assert not new_violations, f"test reached protected resources: {new_violations}"
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == handlers, (
        "test replaced this process's signal handlers"
    )


@pytest.fixture
def deliberate_network(monkeypatch):
    """For a test whose purpose is to attempt a connection and see it blocked. Labels
    its attempts as deliberate, including those of any child it starts. Requested per
    test, so a test isn't assumed deliberate because of the module it's in."""
    monkeypatch.setenv(DELIBERATE_NETWORK, "1")


@pytest.fixture
def live_neo4j():
    """For tests that need a real database. Skips unless live mode was requested.
    Unsafe or incomplete settings never reach this point: they stop the run in
    pytest_configure. Connects only here, at test run time, never at import or
    collection, and fails rather than skips if the database can't be verified.
    See tests/_production_safety.py for what that verification does not prove."""
    if SAFETY.config.live_uri is None:
        pytest.skip(SAFETY.config.live_skip_reason)
    from neo4j_client import driver
    try:
        verify_disposable_marker(driver, SAFETY.config.live_marker)
    except UnsafeConfiguration as error:
        pytest.fail(str(error), pytrace=False)


@pytest.fixture(autouse=True)
def _reset_gnn_inference_singleton():
    """ml.gnn.inference.gnn_inference_service is a process-wide singleton
    that, by design, loads the model artifact once and caches that
    decision for the process's lifetime (real production behavior: a
    feature flag read once at startup, not re-checked per request).

    That design makes tests order-dependent: whichever test runs FIRST
    with real config.GNN_ENABLED (now True by default in this
    environment's .env, per the user's request) permanently marks the
    shared singleton "loaded", so a LATER test's
    `monkeypatch.setattr(config, "GNN_ENABLED", False)` has no effect on
    it -- the singleton never re-reads the flag once `_loaded=True`.
    Resetting its private state before every test restores isolation
    without changing the singleton's real (correct) production
    behavior."""
    from ml.gnn.inference import gnn_inference_service

    gnn_inference_service._loaded = False
    gnn_inference_service._model = None
    gnn_inference_service._metadata = None
    gnn_inference_service._embedding_cache.clear()
    yield
    gnn_inference_service._loaded = False
    gnn_inference_service._model = None
    gnn_inference_service._metadata = None
    gnn_inference_service._embedding_cache.clear()
