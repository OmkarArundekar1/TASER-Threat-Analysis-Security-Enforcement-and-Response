"""
Tests for the test-suite safety layer (tests/_production_safety.py).

Nothing here reaches a real resource, even if a guard were broken:
- Network tests use port 9 on loopback and 192.0.2.1 (RFC 5737 TEST-NET),
  never a port a production service uses.
- Path tests check real paths through refusal(), which never opens anything.
  Actual refusal is exercised on a temporary file, in a subprocess, because audit
  hooks cannot be removed from a process.
- Refused commands would be harmless if run: `docker --version`.
- Unsafe settings are given to a child pytest that stops before collecting.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import _production_safety as safety
from tests._isolated_listener import real_offset_state, run_isolated_listener
from tests._production_safety import (
    CLOSED_HTTP_URL,
    CLOSED_NEO4J_URI,
    LIVE_OPT_IN,
    LIVE_URI,
    NONEXISTENT_ALERT_FILE,
    REAL_LISTENER_LOG,
    REAL_MISP_CACHES,
    REAL_OFFSET_FILE,
    BlockedConnection,
    NetworkGuard,
    PathGuard,
    SafetyConfig,
    UnsafeConfiguration,
    prepare_environment,
    verify_resolved_settings,
)

BACKEND_DIR = Path(safety.BACKEND_DIR)
TESTS_DIR = BACKEND_DIR / "tests"
GATED_MODULES = [
    "test_dashboard_api_routes.py",
    "test_determinism.py",
    "test_campaign_graphs.py",
    "test_temporal_graphs.py",
    "test_gnn_inference.py",
    "test_xgboost_ablation.py",
    "test_gnn_production_integration.py",
]
DISPOSABLE = "bolt://127.0.0.1:17687"
LIVE_CREDENTIALS = {
    safety.LIVE_USERNAME: "disposable-user",
    safety.LIVE_PASSWORD: "disposable-password",
    safety.LIVE_MARKER: "disposable-marker",
}


@pytest.fixture
def dotenv(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "NEO4J_URI=bolt://localhost:7690\n"
        "NEO4J_PASSWORD=not-a-real-secret\n"
        "MISP_URL=https://localhost:8443/\n"
        "SHUFFLE_WEBHOOK=https://shuffler.io/hooks/secret-webhook-id\n"
    )
    return path


# ---------------------------------------------------------------- configuration: defaults

def test_absent_settings_get_closed_defaults(dotenv):
    env = {}
    config = prepare_environment(env, dotenv)

    assert env["NEO4J_URI"] == CLOSED_NEO4J_URI
    assert env["MISP_URL"] == CLOSED_HTTP_URL
    assert env["SHUFFLE_WEBHOOK"] == ""
    assert env["SHUFFLE_BASE_URL"] == ""
    assert env["WAZUH_ALERT_FILE"] == NONEXISTENT_ALERT_FILE
    assert config.live_uri is None


def test_dotenv_secrets_are_not_loaded_into_the_environment(dotenv):
    env = {}
    prepare_environment(env, dotenv)
    assert env["NEO4J_PASSWORD"] == ""


def test_second_preparation_accepts_the_first_ones_values(dotenv):
    """A child pytest inherits the parent's prepared environment."""
    env = {}
    prepare_environment(env, dotenv)
    prepare_environment(env, dotenv)


# ---------------------------------------------------------------- configuration: refused

@pytest.mark.parametrize("name, value", [
    ("NEO4J_URI", "bolt://localhost:7687"),
    ("NEO4J_URI", "bolt://127.0.0.1:7687"),
    ("NEO4J_URI", "neo4j://localhost:7687"),
    ("NEO4J_URI", "bolt://db.example.com:7687"),
    ("NEO4J_URI", "bolt://127.0.0.1:17687"),
    ("MISP_URL", "https://localhost:8443"),
    ("MISP_URL", "https://misp.example.com"),
    ("SHUFFLE_WEBHOOK", "https://shuffler.io/hooks/abc"),
    ("SHUFFLE_BASE_URL", "http://localhost:3001"),
    ("REPLAY_MISP_URL", "https://localhost:8443"),
])
def test_explicit_unsafe_endpoint_is_refused_not_overridden(dotenv, name, value):
    env = {name: value}
    with pytest.raises(UnsafeConfiguration, match=name):
        prepare_environment(env, dotenv)
    assert env[name] == value


def test_real_alert_file_setting_is_refused(dotenv):
    with pytest.raises(UnsafeConfiguration, match="WAZUH_ALERT_FILE"):
        prepare_environment({"WAZUH_ALERT_FILE": "/var/ossec/logs/alerts/alerts.json"}, dotenv)


def test_all_problems_are_reported_together(dotenv):
    env = {"NEO4J_URI": "bolt://localhost:7687", "MISP_URL": "https://localhost:8443"}
    with pytest.raises(UnsafeConfiguration) as caught:
        prepare_environment(env, dotenv)
    assert "NEO4J_URI" in str(caught.value) and "MISP_URL" in str(caught.value)


def test_refusal_message_does_not_leak_webhook_ids_or_credentials(dotenv):
    env = {"SHUFFLE_WEBHOOK": "https://shuffler.io/hooks/secret-webhook-id",
           "NEO4J_URI": "bolt://user:hunter2@localhost:7687"}
    with pytest.raises(UnsafeConfiguration) as caught:
        prepare_environment(env, dotenv)
    assert "secret-webhook-id" not in str(caught.value)
    assert "hunter2" not in str(caught.value)


def test_explicit_closed_values_are_accepted(dotenv):
    env = {"NEO4J_URI": CLOSED_NEO4J_URI, "MISP_URL": "https://127.0.0.1:1"}
    prepare_environment(env, dotenv)


# ---------------------------------------------------------------- configuration: live opt-in

def test_no_opt_in_means_skip_even_with_a_uri(dotenv):
    env = {LIVE_URI: DISPOSABLE}
    config = prepare_environment(env, dotenv)

    assert config.live_uri is None
    assert LIVE_OPT_IN in config.live_skip_reason
    assert env["NEO4J_URI"] == CLOSED_NEO4J_URI


def test_opt_in_without_a_uri_means_skip(dotenv):
    config = prepare_environment({LIVE_OPT_IN: "1"}, dotenv)

    assert config.live_uri is None
    assert LIVE_URI in config.live_skip_reason


def test_opt_in_with_a_disposable_uri_is_used(dotenv):
    env = {LIVE_OPT_IN: "1", LIVE_URI: DISPOSABLE, **LIVE_CREDENTIALS}
    config = prepare_environment(env, dotenv)

    assert config.live_uri == DISPOSABLE
    assert config.live_skip_reason is None
    assert env["NEO4J_URI"] == DISPOSABLE


@pytest.mark.parametrize("uri", [
    "bolt://localhost:7687",           # known production port
    "bolt://127.0.0.1:7687",
    "bolt://localhost:7690",           # what this .env names as production
    "bolt://127.0.0.1:7690",
    "bolt://db.example.com:17687",     # not loopback
    "bolt://10.0.0.5:17687",
    "neo4j://127.0.0.1:17687",         # routing scheme can go elsewhere
    "bolt://127.0.0.1:1",
])
def test_opt_in_with_an_unapproved_uri_is_refused(dotenv, uri):
    with pytest.raises(UnsafeConfiguration, match=LIVE_URI):
        prepare_environment({LIVE_OPT_IN: "1", LIVE_URI: uri, **LIVE_CREDENTIALS}, dotenv)


def test_neo4j_uri_conflicting_with_the_live_uri_is_refused(dotenv):
    env = {LIVE_OPT_IN: "1", LIVE_URI: DISPOSABLE, "NEO4J_URI": "bolt://localhost:7687", **LIVE_CREDENTIALS}
    with pytest.raises(UnsafeConfiguration, match="differs"):
        prepare_environment(env, dotenv)


# ---------------------------------------------------------------- resolved backend config

class _Config:
    NEO4J_URI = CLOSED_NEO4J_URI
    MISP_URL = CLOSED_HTTP_URL
    SHUFFLE_WEBHOOK = ""
    SHUFFLE_BASE_URL = ""
    ALERT_FILE = NONEXISTENT_ALERT_FILE


def test_this_session_resolved_safe_backend_config():
    import config
    verify_resolved_settings(config, safety.ACTIVE.config)
    assert config.NEO4J_URI == CLOSED_NEO4J_URI
    assert config.ALERT_FILE == NONEXISTENT_ALERT_FILE
    assert config.NEO4J_PASSWORD == ""
    assert config.MISP_API_KEY == ""


@pytest.mark.parametrize("name, value", [
    ("NEO4J_URI", "bolt://localhost:7687"),
    ("MISP_URL", "https://localhost:8443"),
    ("SHUFFLE_WEBHOOK", "https://shuffler.io/hooks/abc"),
    ("ALERT_FILE", "/var/ossec/logs/alerts/alerts.json"),
])
def test_resolved_unsafe_backend_config_is_refused(name, value):
    module = type("Resolved", (_Config,), {name: value})
    with pytest.raises(UnsafeConfiguration):
        verify_resolved_settings(module, SafetyConfig())


# ---------------------------------------------------------------- session refusal

@pytest.mark.parametrize("overrides, expected", [
    ({"NEO4J_URI": "bolt://localhost:7687"}, "NEO4J_URI"),
    ({"MISP_URL": "https://localhost:8443"}, "MISP_URL"),
    ({"SHUFFLE_WEBHOOK": "https://shuffler.io/hooks/abc"}, "SHUFFLE_WEBHOOK"),
    ({"WAZUH_ALERT_FILE": "/var/ossec/logs/alerts/alerts.json"}, "WAZUH_ALERT_FILE"),
    ({LIVE_OPT_IN: "1", LIVE_URI: "bolt://localhost:7687"}, LIVE_URI),
])
def test_pytest_stops_with_a_usage_error_on_unsafe_configuration(overrides, expected):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "tests/test_production_safety.py"],
        capture_output=True, text=True, cwd=BACKEND_DIR, timeout=120,
        env={**os.environ, **overrides},
    )
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "Refusing to run tests" in output
    assert expected in output
    assert "tests collected" not in output


# ---------------------------------------------------------------- network guard

def test_guard_never_allows_production_ports_without_explicit_approval():
    guard = NetworkGuard()
    for port in (7687, 8443, 5002, 9200, 9000, 3001):
        assert not guard.is_allowed(socket.AF_INET, ("127.0.0.1", port))
        assert not guard.is_allowed(socket.AF_INET, ("localhost", port))
        assert not guard.is_allowed(socket.AF_INET6, ("::1", port, 0, 0))


def test_allowing_a_port_applies_to_loopback_only():
    guard = NetworkGuard()
    guard.allow_port(17687)
    assert guard.is_allowed(socket.AF_INET, ("127.0.0.1", 17687))
    assert not guard.is_allowed(socket.AF_INET, ("192.0.2.1", 17687))
    assert not guard.is_allowed(socket.AF_INET, ("127.0.0.1", 17688))


def test_session_guard_blocks_a_loopback_connection(deliberate_network):
    with pytest.raises(BlockedConnection):
        socket.create_connection(("127.0.0.1", 9), timeout=0.2)
    assert ("connect", "127.0.0.1:9") in safety.ACTIVE.network.attempts


def test_session_guard_blocks_a_remote_connection(deliberate_network):
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("192.0.2.1", 9), timeout=0.2)


def test_session_guard_blocks_connect_ex(deliberate_network):
    import errno
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        assert sock.connect_ex(("192.0.2.1", 9)) == errno.ECONNREFUSED


def test_session_guard_blocks_sendto(deliberate_network):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(BlockedConnection):
            sock.sendto(b"x", ("192.0.2.1", 9))


def test_session_guard_blocks_name_lookup(deliberate_network):
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("misp.invalid.example", 443)


def test_tls_socket_goes_through_the_guard(deliberate_network):
    import ssl
    context = ssl.create_default_context()
    context.check_hostname = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raw:
        wrapped = context.wrap_socket(raw, do_handshake_on_connect=False)
        with pytest.raises(BlockedConnection):
            wrapped.connect(("192.0.2.1", 443))


def test_requests_is_blocked(deliberate_network):
    import requests
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get("http://192.0.2.1:9/", timeout=0.2)
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get("http://misp.invalid.example/", timeout=0.2)


def test_neo4j_driver_to_the_closed_default_is_blocked(deliberate_network):
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    driver = GraphDatabase.driver(CLOSED_NEO4J_URI, auth=("x", "y"), connection_timeout=0.2)
    try:
        with pytest.raises(ServiceUnavailable):
            driver.verify_connectivity()
    finally:
        driver.close()
    assert ("connect", "127.0.0.1:1") in safety.ACTIVE.network.attempts


def test_a_server_this_process_bound_is_still_reachable():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass


def test_unix_sockets_and_socketpair_are_unaffected(tmp_path):
    left, right = socket.socketpair()
    left.close(), right.close()

    path = str(tmp_path / "s")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(path)


# ---------------------------------------------------------------- path guard

def test_real_paths_are_refused_without_being_opened():
    guard = PathGuard()
    assert guard.refusal(REAL_OFFSET_FILE, write=False)
    assert guard.refusal(REAL_OFFSET_FILE, write=True)
    assert guard.refusal("/var/ossec/logs/alerts/alerts.json", write=False)
    assert guard.refusal("/var/ossec/logs/alerts", write=False)
    assert guard.refusal("/var/ossec/logs/alerts/../alerts/alerts.json", write=False)
    assert guard.refusal(os.fsencode(REAL_OFFSET_FILE), write=False)


def test_listener_log_and_misp_caches_are_refused_for_reads_and_writes():
    guard = PathGuard()
    for path in (REAL_LISTENER_LOG, *REAL_MISP_CACHES):
        assert guard.refusal(path, write=True)
        assert guard.refusal(path, write=False)


def test_other_paths_are_unaffected(tmp_path):
    guard = PathGuard()
    assert not guard.refusal(tmp_path / "offset.dat", write=True)
    assert not guard.refusal("/var/ossecure/file", write=False)


def test_relative_path_is_resolved_against_the_working_directory(monkeypatch):
    monkeypatch.chdir(REAL_OFFSET_FILE.rsplit(os.sep, 1)[0])
    assert PathGuard().refusal("offset.dat", write=False)


@pytest.mark.parametrize("command, refused", [
    (["docker", "ps"], True),
    (["/usr/bin/systemctl", "stop", "wazuh-manager"], True),
    (["sudo", "service", "x", "stop"], True),
    ("systemctl restart neo4j", True),
    ([sys.executable, "-c", "pass"], False),
])
def test_commands_that_control_services_are_refused(command, refused):
    assert bool(PathGuard().command_refusal(command)) is refused


def _child(code, cwd, env=None):
    prelude = (
        f"import sys\nsys.path.insert(0, {str(BACKEND_DIR)!r})\n"
        "from tests._production_safety import install_child_guards, ProtectedResourceAccess, BlockedConnection\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True, text=True, cwd=cwd, timeout=60, env=env,
    )


def test_audit_hook_blocks_every_way_of_touching_a_protected_file(tmp_path):
    target = tmp_path / "protected.dat"
    target.write_text("original")
    other = tmp_path / "other.dat"
    other.write_text("other")
    code = f"""
    import os, pathlib
    target = {str(target)!r}
    other = {str(other)!r}
    guards = install_child_guards(extra_protected=[target])
    attempts = [
        lambda: open(target, "w"),
        lambda: open(target),
        lambda: pathlib.Path(target).write_text("changed"),
        lambda: os.open(target, os.O_WRONLY | os.O_TRUNC),
        lambda: os.truncate(target, 0),
        lambda: os.remove(target),
        lambda: os.rename(target, target + ".moved"),
        lambda: os.rename(other, target),
    ]
    for attempt in attempts:
        try:
            attempt()
        except ProtectedResourceAccess:
            continue
        print("NOT BLOCKED")
    print("violations", len(guards.paths.violations))
    """
    result = _child(code, str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "NOT BLOCKED" not in result.stdout
    assert "violations 8" in result.stdout
    assert target.read_text() == "original"


def test_audit_hook_blocks_service_commands(tmp_path):
    code = """
    import os, subprocess
    install_child_guards()
    for attempt in (lambda: subprocess.run(["docker", "--version"], capture_output=True),
                    lambda: os.system("docker --version")):
        try:
            attempt()
        except ProtectedResourceAccess:
            print("BLOCKED")
    """
    result = _child(code, str(tmp_path))
    assert result.stdout.count("BLOCKED") == 2, result.stdout + result.stderr


def test_child_guards_block_the_network_and_write_a_log(tmp_path, deliberate_network):
    log = tmp_path / "net.log"
    code = """
    import socket
    install_child_guards()
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=0.2)
    except BlockedConnection:
        print("BLOCKED")
    """
    result = _child(code, str(tmp_path), env={**os.environ, safety.NET_LOG: str(log)})

    assert "BLOCKED" in result.stdout, result.stderr
    record = json.loads(log.read_text())
    assert (record["operation"], record["destination"]) == ("connect", "127.0.0.1:9")
    assert record["child"] is True
    assert record["deliberate"] is True
    assert record["test"].endswith("test_child_guards_block_the_network_and_write_a_log")


# ---------------------------------------------------------------- isolated listener

ALERT = json.dumps({"timestamp": "2026-03-11T00:00:00.000+0000", "rule": {"id": "1"},
                    "agent": {"id": "001"}, "location": "test"})


def _lifecycle(extra=""):
    return f"""
    import threading, time
    {extra}
    with open(wl.ALERT_FILE, "w") as f:
        f.write({ALERT!r} + "\\n")
    with open(os.path.join(os.getcwd(), "offset.dat"), "w") as f:
        f.write("0")

    def trigger_stop():
        time.sleep(1.2)
        wl.stop_event.set()

    threading.Thread(target=trigger_stop, daemon=True).start()
    wl.start_listener()
    print("PROCESSED", len(processed))
    print("REAL_LOG_HANDLER", any(
        getattr(h, "baseFilename", "") == {REAL_LISTENER_LOG!r} for h in logging.getLogger().handlers))
    """


def test_isolated_listener_uses_only_the_temporary_alert_and_offset_files(tmp_path):
    before = real_offset_state()

    result = run_isolated_listener(_lifecycle(), cwd=str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "PROCESSED 1" in result.stdout
    assert "REAL_LOG_HANDLER False" in result.stdout
    assert int((tmp_path / "offset.dat").read_text()) == len(ALERT) + 1
    assert real_offset_state() == before


def test_real_offset_stays_untouched_when_processing_fails_as_with_neo4j_down(tmp_path):
    """The audit's scenario: a closed port makes processing fail, yet the offset
    still advances. That must happen only in the temporary directory."""
    before = real_offset_state()
    failing = (
        "def _down(alert):\n"
        "        raise ConnectionRefusedError('neo4j unavailable')\n"
        "    wl.process_alert = _down"
    )

    result = run_isolated_listener(_lifecycle(failing), cwd=str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert int((tmp_path / "offset.dat").read_text()) == len(ALERT) + 1
    assert real_offset_state() == before


def test_isolated_listener_fails_loudly_if_it_reaches_a_stubbed_dependency(tmp_path):
    result = run_isolated_listener("wl.batch_update_duplicates([])", cwd=str(tmp_path))
    assert result.returncode != 0
    assert "stubbed Neo4j" in result.stderr


def test_listener_points_at_the_temporary_paths_before_any_snippet_runs(tmp_path):
    result = run_isolated_listener(
        """
        print("ALERT", wl.ALERT_FILE)
        print("DIR", os.path.dirname(wl.__file__))
        """,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert f"ALERT {os.path.realpath(tmp_path)}" in result.stdout
    assert f"DIR {os.path.realpath(tmp_path)}" in result.stdout


def test_listener_import_failure_test_leaves_pytest_signal_handlers_alone(tmp_path):
    handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    result = run_isolated_listener("wl.AlertFileHandler()", cwd=str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == handlers
    assert "wazuh_listener" not in sys.modules


# ---------------------------------------------------------------- static checks on the suite

def _test_files():
    return sorted(TESTS_DIR.rglob("*.py"))


def _module_level_nodes(tree):
    """Everything that runs at import, so not function or lambda bodies."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pending.extend(node.decorator_list)  # decorators do run at import
            continue
        if isinstance(node, ast.Lambda):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def test_no_test_module_connects_at_import():
    forbidden = {"verify_connectivity", "_neo4j_reachable", "health_check", "session"}
    offenders = []
    for path in _test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _module_level_nodes(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in forbidden:
                    offenders.append(f"{path.name}:{node.lineno} {name}")
    assert offenders == []


def test_no_test_imports_the_real_listener_into_the_pytest_process():
    offenders = []
    for path in _test_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[-1] == "wazuh_listener" for name in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_gated_modules_use_the_runtime_fixture():
    for name in GATED_MODULES:
        source = (TESTS_DIR / name).read_text(encoding="utf-8")
        assert 'usefixtures("live_neo4j"' in source, name
        assert "_neo4j_reachable" not in source, name


# ---------------------------------------------------------------- collection and skips

def _pytest(args, tmp_path, timeout=420):
    log = tmp_path / "net.log"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        capture_output=True, text=True, cwd=BACKEND_DIR, timeout=timeout,
        env={**os.environ, safety.NET_LOG: str(log)},
    )
    return result, (log.read_text().splitlines() if log.exists() else [])


def test_collecting_the_gated_modules_makes_no_connection_attempt(tmp_path):
    result, attempts = _pytest(
        ["--collect-only", "-q", *[f"tests/{name}" for name in GATED_MODULES]], tmp_path
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert attempts == []


def test_live_tests_skip_without_opt_in_and_without_connecting(tmp_path):
    result, attempts = _pytest(
        ["-q", "-rs", "tests/test_temporal_graphs.py", "-k", "against_live_neo4j"], tmp_path
    )
    assert result.returncode == 0, result.stdout[-2000:]
    assert LIVE_OPT_IN in result.stdout
    assert " passed" not in result.stdout
    assert attempts == []
