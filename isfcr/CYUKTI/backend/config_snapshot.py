"""
config_snapshot.py
=====================
Safe configuration/report mechanism (CONFIGURATION_SNAPSHOT.md,
cross-cutting engineering audit section 9). Records the configuration
relevant to reproducing an experiment or a benchmark run WITHOUT ever
exposing API keys, passwords, tokens, or private credentials --
enforced by an explicit allowlist of config attribute names (not a
denylist), so a newly-added secret in config.py is excluded by
default rather than accidentally exposed.
"""

from __future__ import annotations

import platform
import sys

# Explicit allowlist -- config.py attributes safe to report verbatim.
# Deliberately excludes: NEO4J_USERNAME, NEO4J_PASSWORD, MISP_API_KEY,
# SHUFFLE_API_KEY. SHUFFLE_WEBHOOK/NEO4J_URI/MISP_URL are included as
# plain connection endpoints (no embedded credentials in this
# codebase's usage), but SHUFFLE_WEBHOOK specifically is reported only
# as configured/not-configured (its URL path segment is a Shuffle-
# generated hook identifier, effectively a bearer token for that
# workflow).
_SAFE_SCALAR_FIELDS = [
    "NEO4J_URI",
    "MISP_URL",
    "VERIFY_MISP_SSL",
    "GNN_ENABLED",
    "GNN_MODEL_PATH",
    "CAMPAIGN_TIMEOUT",
    "OPERATION_TIMEOUT",
    "DEDUP_WINDOW",
    "ENABLE_DUPLICATE_BUFFER",
    "LOG_LEVEL",
]


def get_safe_config_snapshot() -> dict:
    """Never raises -- a config attribute that no longer exists (a
    future config.py refactor) is simply omitted, not a crash."""
    import config

    snapshot: dict = {}
    for field in _SAFE_SCALAR_FIELDS:
        if hasattr(config, field):
            snapshot[field] = getattr(config, field)

    snapshot["misp_credential_configured"] = bool(getattr(config, "MISP_API_KEY", ""))
    snapshot["shuffle_webhook_configured"] = bool(getattr(config, "SHUFFLE_WEBHOOK", ""))
    snapshot["shuffle_api_configured"] = bool(
        getattr(config, "SHUFFLE_BASE_URL", "") and getattr(config, "SHUFFLE_API_KEY", "")
    )

    snapshot["software_versions"] = get_software_versions()
    return snapshot


def get_software_versions() -> dict:
    """Real, introspected versions -- never hardcoded/guessed. A
    package that isn't installed is reported as `None`, not omitted
    silently (so a snapshot always documents what it checked)."""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    def _pkg_version(module_name: str) -> str | None:
        try:
            import importlib.metadata
            return importlib.metadata.version(module_name)
        except Exception:
            return None

    for pkg in ["flask", "neo4j", "torch", "xgboost", "scikit-learn", "requests"]:
        versions[pkg] = _pkg_version(pkg)

    return versions
