"""
manifest.py
=============
ExperimentManifest: records the environment an evaluation run actually
executed in -- git commit, Python version, package versions, dataset
versions -- so results are reproducible/auditable. Never records
secrets (API keys, passwords, tokens): only version/identity strings.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict:
    versions = {}
    for pkg in ("neo4j",):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def _neo4j_server_version() -> str:
    try:
        from neo4j_client import driver
        with driver.session() as s:
            row = s.run("CALL dbms.components() YIELD name, versions WHERE name = 'Neo4j Kernel' "
                        "RETURN versions[0] AS v").single()
            return row["v"] if row else "unknown"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


@dataclass
class ExperimentManifest:
    timestamp: str
    git_commit: str
    python_version: str
    platform: str
    package_versions: dict
    neo4j_server_version: str
    dataset_versions: dict = field(default_factory=dict)
    n_samples: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    failures: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_manifest(dataset_versions: dict, n_samples: dict, duration_seconds: float,
                    failures: list, skipped: list) -> ExperimentManifest:
    return ExperimentManifest(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=_package_versions(),
        neo4j_server_version=_neo4j_server_version(),
        dataset_versions=dataset_versions,
        n_samples=n_samples,
        duration_seconds=duration_seconds,
        failures=failures,
        skipped=skipped,
    )
