"""
Regression tests for the Phase 1.2 safety hardening: the listener log, MISP cache and
playbook database, child processes, and live-mode credentials.

Nothing here opens a real file or endpoint, even if a guard were broken. Real paths
are only ever passed to refusal(), which never opens them, and actual refusal is
exercised on temporary files. Network attempts use port 9 on loopback or TEST-NET
addresses. Live mode is tested with fake drivers and values, in dicts that never
become the real environment, against a temporary .env.
"""

from __future__ import annotations

import ast
import os
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import _production_safety as safety
from tests._production_safety import (
    EXTRA_PROTECTED,
    LIVE_MARKER,
    LIVE_OPT_IN,
    LIVE_PASSWORD,
    LIVE_URI,
    LIVE_USERNAME,
    NET_LOG,
    REAL_LISTENER_LOG,
    REAL_LISTENER_LOGS,
    REAL_MISP_CACHES,
    REAL_PLAYBOOK_DB,
    PathGuard,
    SafetyConfig,
    UnsafeConfiguration,
    prepare_environment,
    verify_disposable_marker,
    verify_resolved_settings,
)

BACKEND_DIR = Path(safety.BACKEND_DIR)
TESTS_DIR = BACKEND_DIR / "tests"
DISPOSABLE = "bolt://127.0.0.1:17687"
PRODUCTION_PASSWORD = "production-password-value"
LIVE = {
    LIVE_OPT_IN: "1",
    LIVE_URI: DISPOSABLE,
    LIVE_USERNAME: "disposable-user",
    LIVE_PASSWORD: "disposable-password-value",
    LIVE_MARKER: "disposable-marker-value",
}


@pytest.fixture
def dotenv(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "NEO4J_URI=bolt://localhost:7690\n"
        "NEO4J_USERNAME=neo4j\n"
        f"NEO4J_PASSWORD={PRODUCTION_PASSWORD}\n"
    )
    return path


def _child(code, cwd, env=None):
    prelude = (
        f"import sys\nsys.path.insert(0, {str(BACKEND_DIR)!r})\n"
        "from tests._production_safety import (install_child_guards, "
        "ProtectedResourceAccess, BlockedConnection, UnsafeConfiguration)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True, text=True, cwd=cwd, timeout=60, env=env,
    )


def _nested_pytest(args, tmp_path, extra_env=None):
    log = tmp_path / "net.log"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        capture_output=True, text=True, cwd=BACKEND_DIR, timeout=300,
        env={**os.environ, NET_LOG: str(log), **(extra_env or {})},
    )
    return result, (log.read_text().splitlines() if log.exists() else [])


# ---------------------------------------------------------------- 1. response-shape test

def test_correlation_shape_test_passes_without_any_connection_attempt(tmp_path):
    result, attempts = _nested_pytest(
        ["-q", "tests/test_dashboard_api_routes.py::test_correlation_campaigns_response_shape"],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout[-1500:]
    assert "1 passed" in result.stdout
    assert attempts == []


def test_correlation_shape_test_keeps_its_original_assertions():
    source = (TESTS_DIR / "test_dashboard_api_routes.py").read_text(encoding="utf-8")
    body = source.split("def test_correlation_campaigns_response_shape", 1)[1].split("\ndef ", 1)[0]
    assert "assert resp.status_code == 200" in body
    assert 'assert data["campaign_id"] == "CAMP_1"' in body
    assert 'assert data["similar_campaigns"][0]["campaign_id"] == "CAMP_2"' in body


# ---------------------------------------------------------------- 2-4. listener log

def test_guard_covers_the_path_the_dashboard_uses():
    """Reads the source, because the session redirects the imported constant."""
    tree = ast.parse((BACKEND_DIR / "dashboard_api.py").read_text(encoding="utf-8"))
    assigned = [node for node in tree.body if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "LISTENER_LOG_PATH"]
    assert len(assigned) == 1
    parts = [arg.value for arg in assigned[0].value.args if isinstance(arg, ast.Constant)]
    assert os.path.join(BACKEND_DIR, *parts) == REAL_LISTENER_LOG


def test_guard_covers_the_log_name_the_listener_uses():
    source = (BACKEND_DIR / "listener" / "wazuh_listener.py").read_text(encoding="utf-8")
    assert re.search(r'LOG_DIR = "logs"', source)
    assert re.search(r'LOG_FILE = os\.path\.join\(LOG_DIR, "prerana_listener\.log"\)', source)
    for path in (BACKEND_DIR / "logs", BACKEND_DIR / "listener" / "logs"):
        assert str(path / "prerana_listener.log") in REAL_LISTENER_LOGS


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("path", [*REAL_LISTENER_LOGS, *REAL_MISP_CACHES, REAL_PLAYBOOK_DB])
def test_real_state_files_are_refused_for_both_reads_and_writes(path, write):
    assert PathGuard().refusal(path, write=write)


def test_a_relative_spelling_of_the_real_log_is_refused(monkeypatch):
    monkeypatch.chdir(BACKEND_DIR)
    assert PathGuard().refusal("logs/prerana_listener.log")


def test_a_temporary_log_with_the_same_name_is_permitted(tmp_path):
    log = tmp_path / "logs" / "prerana_listener.log"
    log.parent.mkdir()
    log.write_text("written")
    assert log.read_text() == "written"
    with open(log, "a") as handle:
        handle.write(" and appended")
    assert log.read_text() == "written and appended"
    assert not PathGuard().refusal(log)


READ_ATTEMPTS = """[
    lambda: open(target),
    lambda: open(target, "rb"),
    lambda: pathlib.Path(target).read_text(),
    lambda: pathlib.Path(target).read_bytes(),
    lambda: os.open(target, os.O_RDONLY),
    lambda: shutil.copy(target, target + ".copy"),
    lambda: open(link),
    lambda: sqlite3.connect(target),
]"""
WRITE_ATTEMPTS = """[
    lambda: open(target, "w"),
    lambda: open(target, "a"),
    lambda: open(target, "r+"),
    lambda: pathlib.Path(target).write_text("changed"),
    lambda: pathlib.Path(link).write_text("changed"),
    lambda: os.open(target, os.O_WRONLY | os.O_TRUNC),
    lambda: os.truncate(target, 0),
    lambda: os.remove(target),
    lambda: os.link(target, target + ".hardlink"),
    lambda: os.symlink(target, target + ".symlink"),
]"""


@pytest.mark.parametrize("attempts", [READ_ATTEMPTS, WRITE_ATTEMPTS], ids=["read", "write"])
def test_every_access_method_is_refused_on_a_protected_file(tmp_path, attempts):
    target = tmp_path / "prerana_listener.log"
    target.write_text("original")
    link = tmp_path / "innocent_name.log"
    link.symlink_to(target)
    code = f"""
    import os, pathlib, shutil, sqlite3
    target = {str(target)!r}
    link = {str(link)!r}
    guards = install_child_guards(extra_protected=[target])
    for index, attempt in enumerate({" ".join(attempts.split())}):
        try:
            attempt()
            print("NOT BLOCKED", index)
        except ProtectedResourceAccess:
            pass
    print("done")
    """
    result = _child(code, str(tmp_path))

    assert "done" in result.stdout, result.stderr
    assert "NOT BLOCKED" not in result.stdout
    assert target.read_text() == "original"
    assert not (tmp_path / "prerana_listener.log.copy").exists()
    assert not (tmp_path / "prerana_listener.log.hardlink").exists()
    assert not (tmp_path / "prerana_listener.log.symlink").exists()


def test_a_swallowed_refusal_still_fails_the_test(tmp_path):
    """The code under test catches the exception. The test must not pass anyway."""
    target = tmp_path / "protected.log"
    target.write_text("original")
    test_file = tmp_path / "test_swallows.py"
    test_file.write_text(
        f"def test_swallows():\n"
        f"    try:\n"
        f"        open({str(target)!r})\n"
        f"    except OSError:\n"
        f"        pass\n"
    )
    result, _ = _nested_pytest(
        ["-q", "-p", "tests.conftest", str(test_file)], tmp_path,
        extra_env={EXTRA_PROTECTED: str(target)},
    )
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout[-1500:]
    assert "reached protected resources" in result.stdout


def test_dashboard_never_uses_the_real_log_in_a_test_session():
    import dashboard_api
    from tests.conftest import _redirect_dashboard_listener_log

    _redirect_dashboard_listener_log()
    assert dashboard_api.LISTENER_LOG_PATH != REAL_LISTENER_LOG
    with dashboard_api.app.test_client() as client:
        data = client.get("/api/audit/logs").get_json()
    assert data["entries"] == []
    assert data["log_path"].startswith(safety.ACTIVE.scratch_dir)


# ---------------------------------------------------------------- MISP cache and playbook database

def test_misp_cache_lives_in_the_scratch_directory():
    import misp_cache
    assert str(misp_cache.cache.cache_file).startswith(safety.ACTIVE.scratch_dir)


def test_playbook_store_connects_inside_the_scratch_directory():
    from soar.memory import memory_store
    connection = memory_store._connect()
    try:
        path = connection.execute("PRAGMA database_list").fetchone()[2]
    finally:
        connection.close()
    assert os.path.realpath(path).startswith(os.path.realpath(safety.ACTIVE.scratch_dir))


def test_scratch_directory_is_outside_the_backend():
    assert not os.path.realpath(safety.ACTIVE.scratch_dir).startswith(str(BACKEND_DIR))


# ---------------------------------------------------------------- 5. child processes

def test_the_parent_guard_does_not_carry_into_a_child():
    """Why children must install their own. Connects to nothing."""
    code = "import socket; print('connect' in socket.socket.__dict__)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
    )
    assert result.stdout.strip() == "False"
    assert "connect" in socket.socket.__dict__


def test_guarded_child_inherits_safe_settings_and_cannot_reach_them(tmp_path, deliberate_network):
    code = """
    import socket
    from urllib.parse import urlsplit
    guards = install_child_guards()
    import config
    print("SECRETS", bool(config.NEO4J_PASSWORD or config.MISP_API_KEY or config.SHUFFLE_API_KEY))
    print("SHUFFLE", repr(config.SHUFFLE_WEBHOOK), repr(config.SHUFFLE_BASE_URL))
    for url in (config.NEO4J_URI, config.MISP_URL):
        parts = urlsplit(url)
        print("PORT", parts.port)
        try:
            socket.create_connection((parts.hostname, parts.port), timeout=0.2)
            print("CONNECTED")
        except BlockedConnection:
            print("BLOCKED")
    for port in (7687, 8443):
        print("PRODUCTION_ALLOWED", guards.network.is_allowed(socket.AF_INET, ("127.0.0.1", port)))
    """
    result = _child(code, str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("BLOCKED") == 2
    assert "CONNECTED" not in result.stdout
    assert "SECRETS False" in result.stdout
    assert "SHUFFLE '' ''" in result.stdout
    assert result.stdout.count("PORT 1") == 2
    assert "PRODUCTION_ALLOWED True" not in result.stdout


def test_guarded_child_blocks_an_unsafe_connection(tmp_path, deliberate_network):
    code = """
    import socket
    install_child_guards()
    for address in (("127.0.0.1", 9), ("192.0.2.1", 9)):
        try:
            socket.create_connection(address, timeout=0.2)
            print("CONNECTED")
        except BlockedConnection:
            print("BLOCKED")
    """
    result = _child(code, str(tmp_path))
    assert result.stdout.count("BLOCKED") == 2, result.stdout + result.stderr


@pytest.mark.parametrize("name, value", [
    ("NEO4J_URI", "bolt://localhost:7687"),
    ("MISP_URL", "https://localhost:8443"),
    ("SHUFFLE_WEBHOOK", "https://shuffler.io/hooks/abc"),
])
def test_child_refuses_to_start_on_inherited_unsafe_settings(tmp_path, name, value):
    code = """
    try:
        install_child_guards()
        print("STARTED")
    except UnsafeConfiguration:
        print("REFUSED")
    """
    result = _child(code, str(tmp_path), env={**os.environ, name: value})
    assert "REFUSED" in result.stdout, result.stdout + result.stderr
    assert "STARTED" not in result.stdout


def test_a_failed_activation_is_not_treated_as_success_on_a_second_call(tmp_path):
    code = """
    for _ in range(2):
        try:
            install_child_guards()
            print("STARTED")
        except UnsafeConfiguration:
            print("REFUSED")
    """
    result = _child(code, str(tmp_path), env={**os.environ, "NEO4J_URI": "bolt://localhost:7687"})
    assert result.stdout.count("REFUSED") == 2, result.stdout + result.stderr


def test_boundary_child_installs_the_guard_before_importing_the_backend():
    source = (TESTS_DIR / "test_soc_engine_package_boundary.py").read_text(encoding="utf-8")
    child_code = source.split("code = (", 1)[1]
    assert child_code.index("install_child_guards()") < child_code.index("from soc_engine.model import Autoencoder")


def _calls_subprocess(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and getattr(node.func.value, "id", "") == "subprocess"):
            yield node


def test_every_test_file_that_starts_a_python_child_guards_it():
    """A nested pytest guards itself through conftest. Any other child needs its own."""
    offenders = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not list(_calls_subprocess(ast.parse(source))):
            continue
        if "install_child_guards" not in source and "pytest" not in path.name:
            offenders.append(path.name)
    assert offenders == []


# ---------------------------------------------------------------- 6. live-mode credentials

def test_complete_live_settings_are_used(dotenv):
    env = dict(LIVE)
    config = prepare_environment(env, dotenv)

    assert config.live_uri == DISPOSABLE
    assert config.live_marker == "disposable-marker-value"
    assert env["NEO4J_URI"] == DISPOSABLE
    assert env["NEO4J_USERNAME"] == "disposable-user"
    assert env["NEO4J_PASSWORD"] == "disposable-password-value"


@pytest.mark.parametrize("missing", [LIVE_USERNAME, LIVE_PASSWORD, LIVE_MARKER])
def test_live_mode_rejects_missing_credentials(dotenv, missing):
    env = {key: value for key, value in LIVE.items() if key != missing}
    with pytest.raises(UnsafeConfiguration, match=missing):
        prepare_environment(env, dotenv)


@pytest.mark.parametrize("blank", [LIVE_USERNAME, LIVE_PASSWORD, LIVE_MARKER])
def test_live_mode_rejects_blank_credentials(dotenv, blank):
    with pytest.raises(UnsafeConfiguration, match=blank):
        prepare_environment({**LIVE, blank: "  "}, dotenv)


def test_live_mode_rejects_the_production_password(dotenv):
    with pytest.raises(UnsafeConfiguration, match="production"):
        prepare_environment({**LIVE, LIVE_PASSWORD: PRODUCTION_PASSWORD}, dotenv)


def test_live_mode_does_not_fall_back_to_dotenv_credentials(dotenv):
    env = {LIVE_OPT_IN: "1", LIVE_URI: DISPOSABLE, LIVE_MARKER: "m"}
    with pytest.raises(UnsafeConfiguration) as caught:
        prepare_environment(env, dotenv)
    assert LIVE_USERNAME in str(caught.value) and LIVE_PASSWORD in str(caught.value)
    assert "NEO4J_PASSWORD" not in env


@pytest.mark.parametrize("name", ["NEO4J_USERNAME", "NEO4J_PASSWORD"])
def test_live_mode_rejects_conflicting_exported_credentials(dotenv, name):
    with pytest.raises(UnsafeConfiguration, match=name):
        prepare_environment({**LIVE, name: "something-else"}, dotenv)


def test_live_mode_rejects_credentials_embedded_in_the_uri(dotenv):
    with pytest.raises(UnsafeConfiguration, match="must not contain credentials"):
        prepare_environment({**LIVE, LIVE_URI: "bolt://user:pw@127.0.0.1:17687"}, dotenv)


def test_loopback_restriction_and_port_rules_still_apply_with_good_credentials(dotenv):
    for uri in ("bolt://localhost:7687", "bolt://localhost:7690", "bolt://10.0.0.5:17687"):
        with pytest.raises(UnsafeConfiguration, match=LIVE_URI):
            prepare_environment({**LIVE, LIVE_URI: uri}, dotenv)


def test_credentials_without_opt_in_change_nothing(dotenv):
    env = {key: value for key, value in LIVE.items() if key != LIVE_OPT_IN}
    config = prepare_environment(env, dotenv)

    assert config.live_uri is None
    assert env["NEO4J_URI"] == safety.CLOSED_NEO4J_URI
    assert env["NEO4J_PASSWORD"] == ""


def test_dotenv_is_not_read_without_live_mode(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(safety, "_dotenv_settings", lambda path: calls.append(path) or {})
    prepare_environment({}, tmp_path / ".env")
    assert calls == []


class _FakeSession:
    def __init__(self, count, error=None):
        self.count, self.error, self.queries = count, error, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        if self.error:
            raise self.error
        self.queries.append((query, params))
        return self

    def single(self):
        return {"n": self.count}


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


def test_marker_check_passes_on_exactly_one_matching_node():
    session = _FakeSession(1)
    verify_disposable_marker(_FakeDriver(session), "token")
    assert session.queries[0][1] == {"token": "token"}


def test_marker_check_only_reads():
    session = _FakeSession(1)
    verify_disposable_marker(_FakeDriver(session), "token")
    assert not re.search(r"\b(CREATE|MERGE|SET|DELETE|REMOVE)\b", session.queries[0][0])


@pytest.mark.parametrize("count", [0, 2])
def test_marker_check_fails_closed_on_a_wrong_count(count):
    with pytest.raises(UnsafeConfiguration):
        verify_disposable_marker(_FakeDriver(_FakeSession(count)), "token")


def test_marker_check_fails_closed_when_it_cannot_ask():
    with pytest.raises(UnsafeConfiguration):
        verify_disposable_marker(_FakeDriver(_FakeSession(1, error=OSError("down"))), "token")


# ---------------------------------------------------------------- 7. secrets in messages

SECRETS = [PRODUCTION_PASSWORD, "disposable-password-value", "disposable-marker-value",
           "hunter2", "webhook-id-value", "misp-key-value"]


@pytest.mark.parametrize("env", [
    {**LIVE, LIVE_PASSWORD: PRODUCTION_PASSWORD},
    {**LIVE, "NEO4J_PASSWORD": "hunter2"},
    {**LIVE, LIVE_URI: "bolt://user:hunter2@localhost:7687"},
    {"NEO4J_URI": "bolt://user:hunter2@localhost:7687"},
    {"SHUFFLE_WEBHOOK": "https://shuffler.io/hooks/webhook-id-value"},
    {"MISP_API_KEY": "misp-key-value"},
    {"NEO4J_PASSWORD": "hunter2"},
], ids=["production-password", "conflict", "uri-credentials", "uri", "webhook", "misp-key", "neo4j-password"])
def test_refusal_messages_never_show_secret_values(dotenv, env):
    with pytest.raises(UnsafeConfiguration) as caught:
        prepare_environment(dict(env), dotenv)
    for secret in SECRETS:
        assert secret not in str(caught.value)


def test_marker_failure_message_never_shows_the_token_or_the_error_text():
    with pytest.raises(UnsafeConfiguration) as caught:
        verify_disposable_marker(_FakeDriver(_FakeSession(1, error=OSError("hunter2"))), "token-value")
    assert "token-value" not in str(caught.value)
    assert "hunter2" not in str(caught.value)
    with pytest.raises(UnsafeConfiguration) as caught:
        verify_disposable_marker(_FakeDriver(_FakeSession(0)), "token-value")
    assert "token-value" not in str(caught.value)


def test_marker_does_not_appear_in_the_config_repr():
    assert "token-value" not in repr(SafetyConfig(live_marker="token-value"))


def test_pytest_usage_error_does_not_show_secret_values(tmp_path):
    result, _ = _nested_pytest(
        ["--collect-only", "-q", "tests/test_production_safety.py"], tmp_path,
        extra_env={"MISP_API_KEY": "misp-key-value",
                   "SHUFFLE_WEBHOOK": "https://shuffler.io/hooks/webhook-id-value"},
    )
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "MISP_API_KEY" in output
    for secret in SECRETS:
        assert secret not in output


def test_refused_command_arguments_are_not_recorded(tmp_path):
    code = """
    import subprocess
    guards = install_child_guards()
    try:
        subprocess.run(["docker", "login", "--password", "hunter2"])
    except ProtectedResourceAccess as error:
        print("MESSAGE", error)
    print("VIOLATIONS", guards.paths.violations)
    """
    result = _child(code, str(tmp_path))
    assert "VIOLATIONS ['command docker']" in result.stdout, result.stdout + result.stderr
    assert "hunter2" not in result.stdout


@pytest.mark.parametrize("name", ["NEO4J_PASSWORD", "MISP_API_KEY", "SHUFFLE_API_KEY", "NEO4J_USERNAME"])
def test_resolved_credentials_are_refused_without_live_mode(name):
    class Resolved:
        NEO4J_URI = safety.CLOSED_NEO4J_URI
        MISP_URL = safety.CLOSED_HTTP_URL
        ALERT_FILE = safety.NONEXISTENT_ALERT_FILE

    setattr(Resolved, name, "value-that-must-not-print")
    with pytest.raises(UnsafeConfiguration, match=name) as caught:
        verify_resolved_settings(Resolved, SafetyConfig())
    assert "value-that-must-not-print" not in str(caught.value)
