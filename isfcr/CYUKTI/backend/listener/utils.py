"""Source identity of a Wazuh alert, as the listener reads it.

An alert's identity is its manager and its Wazuh id, both exactly as the alert gives them:

    wazuh:<manager>/<id>                         source_key, what the listener's duplicate check uses
    cyukti.v1|wazuh:<manager>/<id>|<operation>   request_key, for a later write coordinator. Nothing uses it yet

Content is never a substitute. Two real occurrences can have identical content, and hashing a few fields merged 70 of 9,015
sample alerts (research/design/phase_2_6a_decisions/D2_ALERT_DEDUPLICATION_INVESTIGATION.md). So an alert without a usable
manager and id has no identity. It's still delivered, with no duplicate suppression, and a re-read would deliver it again.

What this doesn't establish: that a Wazuh id is unique. It looks like event seconds plus a file offset, and its behaviour
across restarts, rotation, restores and clock changes on the deployed version hasn't been validated.
"""
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

KEY_VERSION = "cyukti.v1"
SOURCE_KIND = "wazuh"

VALID = "VALID"
MISSING_ID = "MISSING_ID"
EMPTY_ID = "EMPTY_ID"
MALFORMED_ID = "MALFORMED_ID"
MISSING_MANAGER = "MISSING_MANAGER"
MALFORMED_MANAGER = "MALFORMED_MANAGER"

_WAZUH_ID = re.compile(r"[0-9]+\.[0-9]+")
_MAX_ID_LENGTH = 64
_MAX_ENCODED_MANAGER_LENGTH = 200
_OPERATION = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SourceIdentity:
    status: str
    manager: Optional[str] = None
    source_id: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.status == VALID

    @property
    def source_key(self) -> Optional[str]:
        if not self.valid:
            return None
        return f"{SOURCE_KIND}:{quote(self.manager, safe='-._~')}/{self.source_id}"

    def as_dict(self) -> dict:
        return {"status": self.status, "manager": self.manager, "id": self.source_id, "source_key": self.source_key}


def extract_source_identity(alert: Any) -> SourceIdentity:
    """Reads only alert["id"] and alert["manager"]["name"]. Nothing is parsed, reconstructed or defaulted, and agent names and
    hostnames are never used: they don't identify the manager. Exact string types, because JSON gives nothing else for a real one."""
    source_id = alert.get("id")
    if source_id is None:
        return SourceIdentity(MISSING_ID)
    if type(source_id) is not str:
        return SourceIdentity(MALFORMED_ID)
    if not source_id:
        return SourceIdentity(EMPTY_ID)
    if len(source_id) > _MAX_ID_LENGTH or not _WAZUH_ID.fullmatch(source_id):
        return SourceIdentity(MALFORMED_ID)
    manager_block = alert.get("manager")
    if manager_block is None:
        return SourceIdentity(MISSING_MANAGER)
    if type(manager_block) is not dict:
        return SourceIdentity(MALFORMED_MANAGER)
    manager = manager_block.get("name")
    if manager is None or manager == "":
        return SourceIdentity(MISSING_MANAGER)
    if type(manager) is not str:
        return SourceIdentity(MALFORMED_MANAGER)
    try:
        encoded = quote(manager, safe="-._~")
    except UnicodeEncodeError:  # a lone surrogate, which JSON can carry
        return SourceIdentity(MALFORMED_MANAGER)
    if len(encoded) > _MAX_ENCODED_MANAGER_LENGTH:
        return SourceIdentity(MALFORMED_MANAGER)
    return SourceIdentity(VALID, manager, source_id)


def assign_source_identity(alert: dict) -> SourceIdentity:
    """Puts the identity on the alert, where it travels through the queue to process_alert. alert_id is the source key, or None."""
    identity = extract_source_identity(alert)
    alert["source_identity"] = identity.as_dict()
    alert["alert_id"] = identity.source_key
    return identity


def request_key(source_key: str, operation: str) -> str:
    """The key one logical write would use. An operation is separate from the alert: one alert makes several."""
    if type(source_key) is not str or not source_key.startswith(f"{SOURCE_KIND}:"):
        raise ValueError("a request key needs a source key")
    if type(operation) is not str or not _OPERATION.fullmatch(operation):
        raise ValueError("operation must be a plain name")
    return f"{KEY_VERSION}|{source_key}|{operation}"


def source_ref(alert: dict) -> str:
    """How logs name an alert. A source key holds a host name, so it isn't logged."""
    key = alert.get("alert_id")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] if key else "no-identity"


def _first(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def network_endpoints(alert: dict) -> tuple[Any, Any, bool]:
    """(source, destination, whether both spellings were present and disagree). Values are returned unchanged.

    Precedence: data.src_ip, data.srcip, then the same two at the top level, and likewise dest_ip and dstip. That's the order
    realtime_socgraph.extract_iocs uses for data, which decides the attacker and victim, so the two can't disagree. Suricata
    alerts spell them src_ip and dest_ip. In the sample no alert has both spellings, but nothing rules it out."""
    data = alert.get("data")
    data = data if isinstance(data, dict) else {}
    source = (data.get("src_ip"), data.get("srcip"), alert.get("src_ip"), alert.get("srcip"))
    destination = (data.get("dest_ip"), data.get("dstip"), alert.get("dest_ip"), alert.get("dstip"))
    conflict = any(len({str(value) for value in group if value}) > 1 for group in (source, destination))
    return _first(*source), _first(*destination), conflict


def content_fingerprint(alert: dict) -> str:
    """For diagnostics only: lets someone reading the log see whether alerts lacking an identity repeat. It must never decide
    whether an alert is delivered, because distinct alerts can share it."""
    rule = alert.get("rule")
    agent = alert.get("agent")
    source, destination, _ = network_endpoints(alert)
    raw = "|".join(str(value) for value in (
        alert.get("timestamp", ""), rule.get("id", "") if isinstance(rule, dict) else "", agent.get("id", "") if isinstance(agent, dict) else "",
        source or "", destination or "", alert.get("location", "")))
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()
