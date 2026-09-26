"""
active_response/firewall_backend.py
======================================
FirewallBackend: the ONLY interface ClientResponseAgent is allowed to
call to actually enforce BLOCK_SOURCE_IP. Every implementation must
support both `block()` and an independent `is_blocked()` check --
enforcement and verification are deliberately separate methods so a
caller can never "verify" a block just by trusting the return value of
`block()` itself.

Two implementations:
  - InMemoryFirewallBackend: real, fully-tested, safe to run anywhere
    (dev, CI, this session) -- a real in-memory rule table, not a
    fake success stub. Every unit test in this package that needs a
    "real" firewall uses this.
  - IptablesFirewallBackend: real Linux iptables commands via
    subprocess -- code-complete and unit-tested (with subprocess
    mocked), but NEVER invoked against a live host in this session.
    Modifying a shared lab machine's actual firewall rules is exactly
    the kind of consequential, hard-to-reverse action this project's
    operating rules say to get explicit confirmation for first: this
    session did not have that confirmation, and the intended target
    (the separate Ubuntu client VM) was not reachable from here anyway
    (no shell access to it). See
    review/phaseX_active_response_report.md's environment section.
"""

from __future__ import annotations

import ipaddress
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass


class InvalidSourceIP(ValueError):
    pass


def validate_ip(ip: str) -> str:
    try:
        ipaddress.ip_address(ip)
    except ValueError as e:
        raise InvalidSourceIP(f"{ip!r} is not a valid IP address") from e
    return ip


@dataclass
class FirewallOperationResult:
    success: bool
    ip: str
    rule_reference: str | None   # backend-specific handle to the rule actually created, for later verification/rollback
    detail: str
    raw_output: str = ""


class FirewallBackend(ABC):
    @abstractmethod
    def block(self, ip: str) -> FirewallOperationResult: ...

    @abstractmethod
    def unblock(self, ip: str) -> FirewallOperationResult: ...

    @abstractmethod
    def is_blocked(self, ip: str) -> bool:
        """Independent state check -- must re-read the actual firewall
        state, never just return whatever `block()` last reported."""


class InMemoryFirewallBackend(FirewallBackend):
    """A real rule table (a set), not a stub that always says yes.
    Deliberately supports simulated failure via `fail_next_block` for
    negative-path tests (Section 16: 'firewall failure')."""

    def __init__(self):
        self._blocked: set[str] = set()
        self.fail_next_block = False

    def block(self, ip: str) -> FirewallOperationResult:
        validate_ip(ip)
        if self.fail_next_block:
            self.fail_next_block = False
            return FirewallOperationResult(False, ip, None, "Simulated firewall failure (test injection).")
        self._blocked.add(ip)
        return FirewallOperationResult(True, ip, f"inmemory-rule:{ip}", f"Blocked {ip} (in-memory backend).")

    def unblock(self, ip: str) -> FirewallOperationResult:
        validate_ip(ip)
        was_present = ip in self._blocked
        self._blocked.discard(ip)
        return FirewallOperationResult(
            success=not was_present or ip not in self._blocked,
            ip=ip, rule_reference=None,
            detail=f"Unblocked {ip} (in-memory backend)." if was_present else f"{ip} was not blocked.",
        )

    def is_blocked(self, ip: str) -> bool:
        validate_ip(ip)
        return ip in self._blocked


class IptablesFirewallBackend(FirewallBackend):
    """Real Linux iptables commands. Code-complete, unit-tested via a
    mocked `_run`, never exercised against a live system this session.
    Uses a uniquely-named, idempotent rule comment so is_blocked() can
    re-derive real state from `iptables -L` rather than trusting its
    own memory of what it did."""

    CHAIN = "CYUKTI_CONTAINMENT"

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(["iptables", *args], capture_output=True, text=True, timeout=10)

    def block(self, ip: str) -> FirewallOperationResult:
        validate_ip(ip)
        result = self._run(["-A", self.CHAIN, "-s", ip, "-j", "DROP", "-m", "comment", "--comment", f"cyukti:{ip}"])
        return FirewallOperationResult(
            success=result.returncode == 0, ip=ip,
            rule_reference=f"iptables:{self.CHAIN}:{ip}" if result.returncode == 0 else None,
            detail="iptables rule added." if result.returncode == 0 else f"iptables failed: {result.stderr}",
            raw_output=result.stdout + result.stderr,
        )

    def unblock(self, ip: str) -> FirewallOperationResult:
        validate_ip(ip)
        result = self._run(["-D", self.CHAIN, "-s", ip, "-j", "DROP", "-m", "comment", "--comment", f"cyukti:{ip}"])
        return FirewallOperationResult(
            success=result.returncode == 0, ip=ip, rule_reference=None,
            detail="iptables rule removed." if result.returncode == 0 else f"iptables removal failed: {result.stderr}",
            raw_output=result.stdout + result.stderr,
        )

    def is_blocked(self, ip: str) -> bool:
        validate_ip(ip)
        result = self._run(["-L", self.CHAIN, "-n", "--line-numbers"])
        return result.returncode == 0 and f"cyukti:{ip}" in result.stdout
