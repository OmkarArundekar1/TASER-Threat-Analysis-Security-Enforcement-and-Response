"""
Stops a test run from touching production resources on this machine.

Three independent layers, because none of them covers what the others do:

1. Configuration. prepare_environment() points every external setting at a
   closed port or at nothing, before backend modules read .env. Explicit unsafe
   settings stop the run instead of being overridden.
2. Network. NetworkGuard refuses IP connections from this interpreter, including
   to localhost, where the production services listen. A closed port would not
   protect against code that hard-codes an address.
3. Files and commands. PathGuard refuses access to the real Wazuh alert files,
   offset file, listener logs, MISP caches and playbook database, and to commands
   that control services. A closed port does nothing for these: a listener with no
   database still advances the real offset. Backend modules that open real state
   when imported are redirected to a scratch directory, because refusing them would
   break the import.

Live Neo4j mode needs all of: CYUKTI_TEST_LIVE_NEO4J=1, CYUKTI_TEST_NEO4J_URI,
CYUKTI_TEST_NEO4J_USERNAME, CYUKTI_TEST_NEO4J_PASSWORD and CYUKTI_TEST_NEO4J_MARKER.
The database must hold a (:CyuktiDisposableTestDatabase {token: <marker>}) node,
which the live_neo4j fixture checks before any test uses it. Whoever creates that
node is declaring the database disposable. A loopback URI, separate credentials and
the marker together make a mistake unlikely. They do not prove a database is
disposable: anyone could add the node to the wrong one.

Limits. These guards act inside one Python interpreter. They do not see native
libraries that open sockets or files themselves, os.stat() and other metadata calls,
hard links, or child processes that do not call install_child_guards().

Imports only the standard library, and must stay that way: conftest.py loads it
before any backend module.
"""

from __future__ import annotations

import atexit
import errno
import importlib.util
import ipaddress
import json
import os
import re
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CLOSED_PORT = 1
CLOSED_NEO4J_URI = f"bolt://127.0.0.1:{CLOSED_PORT}"
CLOSED_HTTP_URL = f"http://127.0.0.1:{CLOSED_PORT}"
NONEXISTENT_ALERT_FILE = "/nonexistent/cyukti-tests/alerts.json"

LIVE_OPT_IN = "CYUKTI_TEST_LIVE_NEO4J"
LIVE_URI = "CYUKTI_TEST_NEO4J_URI"
LIVE_USERNAME = "CYUKTI_TEST_NEO4J_USERNAME"
LIVE_PASSWORD = "CYUKTI_TEST_NEO4J_PASSWORD"
LIVE_MARKER = "CYUKTI_TEST_NEO4J_MARKER"
NET_LOG = "CYUKTI_TEST_NET_LOG"
EXTRA_PROTECTED = "CYUKTI_TEST_EXTRA_PROTECTED"
DELIBERATE_NETWORK = "CYUKTI_TEST_DELIBERATE_NETWORK"
ROOT_PID = "CYUKTI_TEST_ROOT_PID"
UNKNOWN_TEST = "unknown"

REAL_ALERT_ROOT = "/var/ossec"
REAL_OFFSET_FILE = os.path.join(BACKEND_DIR, "listener", "offset.dat")
# dashboard_api.LISTENER_LOG_PATH. The listener itself writes relative to its working
# directory, so the copy under listener/ is covered too.
REAL_LISTENER_LOG = os.path.join(BACKEND_DIR, "logs", "prerana_listener.log")
REAL_LISTENER_LOGS = (
    REAL_LISTENER_LOG,
    os.path.join(BACKEND_DIR, "listener", "logs", "prerana_listener.log"),
)
REAL_MISP_CACHES = (
    os.path.join(BACKEND_DIR, "misp_cache.json"),
    os.path.join(BACKEND_DIR, "listener", "misp_cache.json"),
)
REAL_PLAYBOOK_DB = os.path.join(BACKEND_DIR, "soar", "playbook_memory.db")

# Setting -> value used when the setting is absent. Setting it here also stops
# load_dotenv() supplying the production value, since it never overrides.
_ENDPOINT_DEFAULTS = {
    "NEO4J_URI": CLOSED_NEO4J_URI,
    "MISP_URL": CLOSED_HTTP_URL,
    "SHUFFLE_WEBHOOK": "",
    "SHUFFLE_BASE_URL": "",
    "REPLAY_MISP_URL": CLOSED_HTTP_URL,
}
# Absent in a normal test run, so that load_dotenv() cannot put production
# credentials into a test process.
_SECRET_SETTINGS = ("NEO4J_USERNAME", "NEO4J_PASSWORD", "MISP_API_KEY", "SHUFFLE_API_KEY")
_DEFAULT_PORTS = {"bolt": 7687, "neo4j": 7687, "http": 80, "https": 443}
_BUILT_IN_PRODUCTION_PORTS = {7687, 8443}

_DENIED_COMMANDS = {
    "cypher-shell", "curl", "docker", "docker-compose", "ip6tables", "iptables",
    "neo4j", "neo4j-admin", "nft", "service", "sudo", "systemctl", "wazuh-control",
    "wget",
}


class UnsafeConfiguration(Exception):
    """Test configuration points at a resource tests must not touch."""


class BlockedConnection(ConnectionRefusedError):
    """Raised instead of connecting. Subclasses ConnectionRefusedError so that
    code under test sees what a closed port would give it."""


class ProtectedResourceAccess(PermissionError):
    """Raised instead of opening a protected path or running a denied command."""


def _is_loopback(host):
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _endpoint(value):
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError:
        port = None
    return parts.scheme.lower(), (parts.hostname or "").lower(), port or _DEFAULT_PORTS.get(parts.scheme.lower())


def _describe(value):
    """Scheme, host and port only. URLs here can carry credentials or webhook ids."""
    scheme, host, port = _endpoint(value)
    return f"{scheme}://{host}:{port}"


def _is_closed_sentinel(value):
    _, host, port = _endpoint(value)
    return _is_loopback(host) and port == CLOSED_PORT


def _dotenv_settings(dotenv_path):
    """The .env file as a dict. Parsed by hand, so that nothing here loads it into
    the environment. Read only when live mode is requested."""
    found = {}
    try:
        with open(dotenv_path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return found
    for line in lines:
        key, separator, value = line.strip().partition("=")
        if separator:
            found[key.strip()] = value.strip().strip("'\"")
    return found


def _dotenv_endpoints(settings):
    found = set()
    for name in _ENDPOINT_DEFAULTS:
        if settings.get(name):
            _, host, port = _endpoint(settings[name])
            found.add(("loopback" if _is_loopback(host) else host, port))
    return found


def _live_uri_problem(uri, production_endpoints):
    scheme, host, port = _endpoint(uri)
    parts = urlsplit(uri)
    if parts.username or parts.password:
        return "must not contain credentials. Use the separate settings"
    if scheme != "bolt":
        return "must use bolt://. Other schemes, such as neo4j://, can route to other servers"
    if not _is_loopback(host):
        return "must be on loopback, so that it can only be a local disposable instance"
    if port is None or port in _BUILT_IN_PRODUCTION_PORTS or port == CLOSED_PORT:
        return f"port {port} is not usable: 7687 and 8443 are production ports here"
    if ("loopback", port) in production_endpoints:
        return "it is the endpoint configured in backend/.env, which is production"
    return None


def _live_credential_problems(environ, production_settings):
    """Names settings only. Must never include a value."""
    problems = []
    username = environ.get(LIVE_USERNAME, "")
    password = environ.get(LIVE_PASSWORD, "")
    for name, value in ((LIVE_USERNAME, username), (LIVE_PASSWORD, password),
                        (LIVE_MARKER, environ.get(LIVE_MARKER, ""))):
        if not value.strip():
            problems.append(f"Live mode requires {name}. Production credentials are never reused.")
    production_password = production_settings.get("NEO4J_PASSWORD", "")
    if password and production_password and password == production_password:
        problems.append(f"{LIVE_PASSWORD} is the same as NEO4J_PASSWORD in backend/.env, which is production.")
    for name, value in (("NEO4J_USERNAME", username), ("NEO4J_PASSWORD", password)):
        if environ.get(name, value) != value:
            problems.append(f"{name} is set and differs from the disposable credentials. Unset it.")
    return problems


@dataclass
class SafetyConfig:
    live_uri: str | None = None
    live_skip_reason: str | None = None
    live_marker: str | None = field(default=None, repr=False)


def prepare_environment(environ, dotenv_path):
    """Applies safe defaults to `environ`, or raises UnsafeConfiguration listing
    every problem. Explicit values are never replaced, and messages name settings
    without showing secret values."""
    problems = []
    config = SafetyConfig()

    opted_in = environ.get(LIVE_OPT_IN, "").strip().lower() in ("1", "true", "yes")
    requested = environ.get(LIVE_URI, "").strip()
    if not opted_in:
        config.live_skip_reason = (
            f"live Neo4j tests need {LIVE_OPT_IN}=1 and {LIVE_URI}=<disposable bolt URI>"
        )
    elif not requested:
        config.live_skip_reason = f"{LIVE_OPT_IN} is set, but {LIVE_URI} is not"
    else:
        production_settings = _dotenv_settings(dotenv_path)
        problem = _live_uri_problem(requested, _dotenv_endpoints(production_settings))
        if problem:
            problems.append(f"{LIVE_URI}={_describe(requested)} is refused: {problem}.")
        else:
            config.live_uri = requested
        credential_problems = _live_credential_problems(environ, production_settings)
        problems.extend(credential_problems)
        if not credential_problems:
            environ["NEO4J_USERNAME"] = environ[LIVE_USERNAME]
            environ["NEO4J_PASSWORD"] = environ[LIVE_PASSWORD]
            config.live_marker = environ[LIVE_MARKER]

    for name, default in _ENDPOINT_DEFAULTS.items():
        value = environ.get(name)
        if value is None:
            environ[name] = config.live_uri if (name == "NEO4J_URI" and config.live_uri) else default
            continue
        value = value.strip()
        if not value and not default:
            continue
        if name == "NEO4J_URI" and config.live_uri:
            if value != config.live_uri:
                problems.append(
                    f"NEO4J_URI={_describe(value)} differs from {LIVE_URI}. Unset it, or make them equal."
                )
            continue
        if value and _is_closed_sentinel(value):
            continue
        problems.append(
            f"{name}={_describe(value) if value else '<empty>'} is not allowed in a test run. "
            f"Unset it, or use {default or 'an empty value'}."
        )

    for name in _SECRET_SETTINGS:
        if config.live_uri and name in ("NEO4J_USERNAME", "NEO4J_PASSWORD"):
            continue
        if environ.get(name):
            problems.append(f"{name} is set. Tests must not hold real credentials, so unset it.")
        else:
            environ[name] = ""

    alert_file = environ.get("WAZUH_ALERT_FILE")
    if alert_file is None:
        environ["WAZUH_ALERT_FILE"] = NONEXISTENT_ALERT_FILE
    elif os.path.abspath(alert_file).startswith(REAL_ALERT_ROOT + os.sep):
        problems.append(f"WAZUH_ALERT_FILE={alert_file} is the real Wazuh alert file.")

    if problems:
        raise UnsafeConfiguration(
            "Refusing to run tests: configuration points at production resources.\n  - "
            + "\n  - ".join(problems)
        )
    return config


def verify_resolved_settings(module, safety_config):
    """Checks what backend config actually resolved, after load_dotenv() ran."""
    problems = []
    neo4j_uri = getattr(module, "NEO4J_URI", "")
    if neo4j_uri != safety_config.live_uri and not _is_closed_sentinel(neo4j_uri):
        problems.append(f"config.NEO4J_URI resolved to {_describe(neo4j_uri)}")
    misp_url = getattr(module, "MISP_URL", "")
    if not _is_closed_sentinel(misp_url):
        problems.append(f"config.MISP_URL resolved to {_describe(misp_url)}")
    for name in ("SHUFFLE_WEBHOOK", "SHUFFLE_BASE_URL"):
        value = getattr(module, name, "")
        if value and not _is_closed_sentinel(value):
            problems.append(f"config.{name} resolved to {_describe(value)}")
    if os.path.abspath(getattr(module, "ALERT_FILE", "")).startswith(REAL_ALERT_ROOT + os.sep):
        problems.append("config.ALERT_FILE resolved to the real Wazuh alert file")
    for name in _SECRET_SETTINGS:
        if safety_config.live_uri and name in ("NEO4J_USERNAME", "NEO4J_PASSWORD"):
            continue
        if getattr(module, name, ""):
            problems.append(f"config.{name} resolved to a non-empty value")
    if problems:
        raise UnsafeConfiguration(
            "Refusing to run tests: backend config resolved to production resources.\n  - "
            + "\n  - ".join(problems)
        )


def verify_disposable_marker(driver, token):
    """Raises UnsafeConfiguration unless the database holds the marker node. Fails
    closed on any error. See the module docstring for what this does not prove."""
    try:
        with driver.session() as session:
            record = session.run(
                "MATCH (m:CyuktiDisposableTestDatabase {token: $token}) RETURN count(m) AS n",
                token=token,
            ).single()
        found = record["n"]
    except Exception as error:
        raise UnsafeConfiguration(
            f"Could not verify that the live Neo4j is disposable: {type(error).__name__}."
        ) from None
    if found != 1:
        raise UnsafeConfiguration(
            "The live Neo4j has no matching disposable marker. Refusing to use it."
        )


def _sanitized_destination(target):
    """Host and port only. Anything else, such as a URL passed as a host name, could
    hold credentials, so it isn't recorded."""
    return target if re.fullmatch(r"[A-Za-z0-9._:\[\]-]{1,255}", target) else "<unrecordable>"


def _current_test():
    """(node id, phase) from the variable pytest sets for each test phase. Child
    processes inherit it. Between tests, or in a process pytest didn't start, the test
    is unknown. Parameter values are dropped from the id, because they can be anything,
    credentials included."""
    current = os.environ.get("PYTEST_CURRENT_TEST", "")
    if not current:
        return UNKNOWN_TEST, None
    node_id, _, phase = current.rpartition(" (")
    return re.sub(r"\[.*\]$", "[<param>]", node_id), phase.rstrip(")")


class NetworkGuard:
    """Refuses IP traffic from this interpreter. Allowed: Unix sockets, loopback
    ports this process bound itself, and ports added through allow_port().

    Each refusal is recorded with the test running at that moment. A background thread
    outliving its test is recorded against whichever test is current, so `thread` is
    kept to make that visible. `deliberate` is true only when the test asked for it
    through the deliberate_network fixture."""

    def __init__(self, log_path=None):
        self.attempts = []
        self.records = []
        self.allowed_ports = set()
        self.bound_ports = set()
        self.log_path = log_path

    def allow_port(self, port):
        self.allowed_ports.add(port)

    def is_allowed(self, family, address):
        if family not in (socket.AF_INET, socket.AF_INET6):
            return True
        host, port = address[0], address[1]
        return _is_loopback(host) and (port in self.allowed_ports or port in self.bound_ports)

    def _record(self, kind, target):
        destination = _sanitized_destination(target)
        self.attempts.append((kind, destination))
        test, phase = _current_test()
        record = {
            "test": test,
            "phase": phase,
            "operation": kind,
            "destination": destination,
            "deliberate": os.environ.get(DELIBERATE_NETWORK) == "1",
            "child": os.environ.get(ROOT_PID, str(os.getpid())) != str(os.getpid()),
            "thread": threading.current_thread().name,
        }
        self.records.append(record)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def _check(self, sock, address, kind):
        if not self.is_allowed(sock.family, address):
            self._record(kind, f"{address[0]}:{address[1]}")
            raise BlockedConnection(
                errno.ECONNREFUSED, f"test network guard blocked {kind} to {address[0]}:{address[1]}"
            )

    def install(self):
        guard = self
        real_connect = socket.socket.connect
        real_connect_ex = socket.socket.connect_ex
        real_sendto = socket.socket.sendto
        real_bind = socket.socket.bind
        real_getaddrinfo = socket.getaddrinfo

        def connect(sock, address):
            guard._check(sock, address, "connect")
            return real_connect(sock, address)

        def connect_ex(sock, address):
            try:
                guard._check(sock, address, "connect_ex")
            except BlockedConnection:
                return errno.ECONNREFUSED
            return real_connect_ex(sock, address)

        def sendto(sock, data, *args):
            guard._check(sock, args[-1], "sendto")
            return real_sendto(sock, data, *args)

        def bind(sock, address):
            result = real_bind(sock, address)
            if sock.family in (socket.AF_INET, socket.AF_INET6) and _is_loopback(address[0]):
                guard.bound_ports.add(sock.getsockname()[1])
            return result

        def getaddrinfo(host, *args, **kwargs):
            name = host.decode() if isinstance(host, bytes) else host
            if not name or _is_loopback(name) or _is_ip_literal(name):
                return real_getaddrinfo(host, *args, **kwargs)
            guard._record("resolve", str(name))
            raise socket.gaierror(socket.EAI_NONAME, f"test network guard blocked lookup of {name}")

        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.socket.sendto = sendto
        socket.socket.bind = bind
        socket.getaddrinfo = getaddrinfo


def _is_ip_literal(name):
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


class PathGuard:
    """Refuses protected paths and commands through a Python audit hook, which
    also covers os.open and every library that goes through them."""

    def __init__(self, extra_protected=()):
        self.no_access = {
            spelling
            for path in (REAL_OFFSET_FILE, *REAL_LISTENER_LOGS, *REAL_MISP_CACHES,
                         REAL_PLAYBOOK_DB, *extra_protected)
            for spelling in (os.path.abspath(path), os.path.realpath(path))
        }
        self.violations = []

    def refusal(self, path, write=False):
        """Why this access is refused, or None. Follows symlinks, and never opens the
        file. Reads and writes are treated alike; `write` is kept for callers that
        state their intent."""
        path = os.fsdecode(os.fspath(path))
        for resolved in (os.path.abspath(path), os.path.realpath(path)):
            if resolved == REAL_ALERT_ROOT or resolved.startswith(REAL_ALERT_ROOT + os.sep):
                return "real Wazuh data"
            if resolved in self.no_access:
                return "protected file"
        return None

    def command_refusal(self, argv):
        if isinstance(argv, (bytes, str)):
            argv = os.fsdecode(argv).split()
        if not argv:
            return None
        name = os.path.basename(os.fsdecode(argv[0]))
        return f"command {name}" if name in _DENIED_COMMANDS else None

    def _refuse(self, reason, target=None):
        """`target` is a path, or None for a command, whose arguments may hold secrets."""
        self.violations.append(reason if target is None else f"{reason}: {target}")
        raise ProtectedResourceAccess(
            errno.EACCES, f"tests must not access {reason}", None if target is None else str(target)
        )

    def _hook(self, event, args):
        if event == "open":
            path, mode, flags = args
            if path is None or isinstance(path, int):
                return
            write = (
                any(c in mode for c in "wax+") if isinstance(mode, str)
                else bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
            )
            reason = self.refusal(path, write)
            if reason:
                self._refuse(reason, path)
        elif event in ("os.remove", "os.truncate", "os.utime", "os.chmod", "os.chown"):
            if not isinstance(args[0], int):
                reason = self.refusal(args[0], True)
                if reason:
                    self._refuse(reason, args[0])
        elif event in ("os.rename", "os.link", "os.symlink"):
            for path in args[:2]:
                reason = self.refusal(path, True)
                if reason:
                    self._refuse(reason, path)
        elif event == "sqlite3.connect":
            if isinstance(args[0], (str, bytes, os.PathLike)) and args[0] not in (":memory:", ""):
                reason = self.refusal(args[0], True)
                if reason:
                    self._refuse(reason, args[0])
        elif event in ("os.listdir", "os.scandir"):
            if args[0] is not None and not isinstance(args[0], int):
                reason = self.refusal(args[0], False)
                if reason:
                    self._refuse(reason, args[0])
        elif event == "subprocess.Popen":
            reason = self.command_refusal(args[1] or args[0])
            if reason:
                self._refuse(reason)
        elif event == "os.system":
            reason = self.command_refusal(args[0])
            if reason:
                self._refuse(reason)

    def install(self):
        sys.addaudithook(self._hook)


def _redirect_playbook_db(scratch_dir):
    """soar.memory opens its database when imported. Sends that to scratch_dir."""
    real_connect = sqlite3.connect
    real_paths = {os.path.abspath(REAL_PLAYBOOK_DB), os.path.realpath(REAL_PLAYBOOK_DB)}
    replacement = os.path.join(scratch_dir, "playbook_memory.db")

    def connect(database, *args, **kwargs):
        if isinstance(database, (str, bytes, os.PathLike)):
            name = os.fsdecode(os.fspath(database))
            if name != ":memory:" and os.path.abspath(name) in real_paths:
                database = replacement
        return real_connect(database, *args, **kwargs)

    sqlite3.connect = connect
    return replacement


def _isolate_misp_cache(scratch_dir):
    """misp_cache builds its cache when imported, from a path relative to the working
    directory, so importing it from backend/ reads the real file. Loads it once from
    scratch_dir and keeps it there. Stays standard-library only: so does misp_cache."""
    if "misp_cache" in sys.modules:
        raise RuntimeError("misp_cache was imported before the test safety layer")
    spec = importlib.util.spec_from_file_location(
        "misp_cache", os.path.join(BACKEND_DIR, "misp_cache.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["misp_cache"] = module
    working_directory = os.getcwd()
    os.chdir(scratch_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(working_directory)
    module.cache.cache_file = Path(scratch_dir) / "misp_cache.json"


@dataclass
class Safety:
    network: NetworkGuard
    paths: PathGuard
    scratch_dir: str
    config: SafetyConfig = field(default_factory=SafetyConfig)


ACTIVE = None
_GUARDS = None


def activate(extra_protected=()):
    """Installs the guards first, then checks configuration, so they are in place even
    if configuration is unsafe. ACTIVE stays None until configuration has passed, and
    a later call checks it again rather than assuming."""
    global ACTIVE, _GUARDS
    if ACTIVE is not None:
        return ACTIVE
    os.environ.setdefault(ROOT_PID, str(os.getpid()))
    if _GUARDS is None:
        from_environment = [p for p in os.environ.get(EXTRA_PROTECTED, "").split(os.pathsep) if p]
        network = NetworkGuard(log_path=os.environ.get(NET_LOG))
        paths = PathGuard((*extra_protected, *from_environment))
        network.install()
        paths.install()
        scratch_dir = tempfile.mkdtemp(prefix="cyukti-test-state-")
        atexit.register(shutil.rmtree, scratch_dir, ignore_errors=True)
        _redirect_playbook_db(scratch_dir)
        _isolate_misp_cache(scratch_dir)
        _GUARDS = Safety(network, paths, scratch_dir)
    config = prepare_environment(os.environ, os.path.join(BACKEND_DIR, ".env"))
    if config.live_uri:
        _GUARDS.network.allow_port(_endpoint(config.live_uri)[2])
    _GUARDS.config = config
    ACTIVE = _GUARDS
    return ACTIVE


def install_child_guards(extra_protected=()):
    """For a Python subprocess started by a test: call before any backend import."""
    return activate(extra_protected)
