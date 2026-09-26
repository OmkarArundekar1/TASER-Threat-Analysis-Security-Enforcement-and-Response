"""
Static safety audit (Section 14): greps active_response/'s own source
for banned patterns. This is a real, load-bearing test -- if someone
later adds a generic command-execution path, this test fails, it
doesn't just document a hope.
"""

import os

import active_response

_PKG_DIR = os.path.dirname(active_response.__file__)

_BANNED_PATTERNS = [
    "os.system(",
    "shell=True",
    "eval(",
    "exec(",
    "paramiko",       # arbitrary SSH execution
    "subprocess.Popen",
]

# subprocess.run IS allowed, but only inside firewall_backend.py's
# documented, fixed-argument-list iptables calls -- never with a
# caller-supplied command string.
_ALLOWED_SUBPROCESS_FILE = "firewall_backend.py"


def _all_source_files():
    for root, _, files in os.walk(_PKG_DIR):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def test_no_banned_execution_patterns_anywhere_in_the_package():
    violations = []
    for path in _all_source_files():
        with open(path) as f:
            content = f.read()
        for pattern in _BANNED_PATTERNS:
            if pattern in content:
                violations.append(f"{path}: contains banned pattern {pattern!r}")
    assert not violations, "\n".join(violations)


def test_subprocess_run_only_appears_in_the_firewall_backend_with_a_fixed_argv():
    for path in _all_source_files():
        with open(path) as f:
            content = f.read()
        if "subprocess.run" in content:
            assert os.path.basename(path) == _ALLOWED_SUBPROCESS_FILE, (
                f"{path} calls subprocess.run but is not the documented firewall backend -- "
                f"this is exactly the generic-command-execution risk Section 14 prohibits."
            )


def test_containment_request_has_no_command_or_executable_path_field():
    from active_response.client_agent import ContainmentRequest
    field_names = set(ContainmentRequest.__dataclass_fields__)
    for banned in ("command", "cmd", "executable", "shell_command", "script"):
        assert banned not in field_names


def test_no_generic_execute_endpoint_pattern_string_anywhere():
    for path in _all_source_files():
        with open(path) as f:
            content = f.read()
        assert "/execute?command=" not in content
