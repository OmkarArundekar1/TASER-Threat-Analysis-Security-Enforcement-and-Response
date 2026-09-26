"""
run_all.py
============
Reproducible evaluation runner: python run_all.py (run from evaluation/,
with backend/ importable -- see README.md for the exact command, since
this needs both the evaluation/ package and Neo4j-connected backend/
modules on sys.path, and the neo4j driver's .env is resolved relative
to backend/ as CWD).

Runs all 7 evaluation categories and writes:
    results/summary.json, results/summary.md
    results/<task>_metrics.json  (one per category)
    results/manifest.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from manifest import build_manifest
from ground_truth import store
from ground_truth.schema import ReviewStatus

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _dataset_version(dataset_name: str) -> str:
    _, status = store.load_best_available(dataset_name)
    return status.value


def run() -> dict:
    start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from evaluators import (
        mitre_eval, prediction_eval, attribution_eval, campaign_correlation_eval,
        rag_eval, investigation_eval, threat_qualification_eval,
    )

    results = {}
    failures = []
    skipped = []

    def _run(name, fn):
        try:
            result = fn()
            results[name] = result.to_dict()
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            results[name] = {"task": name, "status": "ERROR", "reason": str(e)}

    _run("mitre_mapping", lambda: mitre_eval.evaluate(allow_preliminary=True))
    _run("prediction", lambda: prediction_eval.evaluate())
    _run("attribution", lambda: attribution_eval.evaluate(allow_preliminary=True))
    _run("campaign_correlation", lambda: campaign_correlation_eval.evaluate(allow_preliminary=True))
    _run("threat_qualification", lambda: threat_qualification_eval.evaluate(allow_preliminary=True))

    for source in ("mitre_semantic", "campaign_narrative", "gnn_topology"):
        result = rag_eval.evaluate(source)
        results[f"rag_{source}"] = result.to_dict()
        if result.status.value == "NOT_MEASURED":
            skipped.append(f"rag_{source}: {result.reason}")

    inv_result = investigation_eval.evaluate()
    results["investigation"] = inv_result.to_dict()
    if inv_result.status.value == "NOT_MEASURED":
        skipped.append(f"investigation: {inv_result.reason}")

    for name in ("mitre_mapping_v0", "threat_qualification_v0", "campaign_correlation_v0", "attribution_v0"):
        _, status = store.load_best_available(name)
        # only report versions for datasets that actually produced a result above

    dataset_versions = {
        "mitre_mapping_v0": _dataset_version("mitre_mapping_v0"),
        "threat_qualification_v0": _dataset_version("threat_qualification_v0"),
        "campaign_correlation_v0": _dataset_version("campaign_correlation_v0"),
        "attribution_v0": _dataset_version("attribution_v0"),
    }
    n_samples = {k: v.get("n", 0) for k, v in results.items()}

    duration = time.time() - start
    manifest = build_manifest(dataset_versions, n_samples, duration, failures, skipped)

    with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    for name, result in results.items():
        with open(os.path.join(RESULTS_DIR, f"{name}_metrics.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    summary = {"manifest": manifest.to_dict(), "results": {k: {kk: vv for kk, vv in v.items() if kk != "metrics"} for k, v in results.items()}}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _write_summary_md(results, manifest)
    return results


def _write_summary_md(results: dict, manifest) -> None:
    lines = [
        "# CYUKTI Evaluation Run Summary",
        "",
        f"Run at {manifest.timestamp} | commit `{manifest.git_commit[:12]}` | Neo4j {manifest.neo4j_server_version} | duration {manifest.duration_seconds:.1f}s",
        "",
        "| Task | Status | n | Headline metric |",
        "|---|---|---|---|",
    ]
    headline_keys = {
        "mitre_mapping": "f1", "prediction": None, "attribution": "accuracy",
        "campaign_correlation": "pairwise_f1", "threat_qualification": "accuracy",
    }
    for name, result in results.items():
        status = result.get("status", "?")
        n = result.get("n", 0)
        metrics = result.get("metrics", {}) or {}
        key = headline_keys.get(name)
        headline = f"{metrics[key]:.3f}" if key and key in metrics else "-"
        lines.append(f"| {name} | {status} | {n} | {headline} |")

    if manifest.failures:
        lines += ["", "## Failures", ""] + [f"- {f}" for f in manifest.failures]
    if manifest.skipped:
        lines += ["", "## Skipped / not measured", ""] + [f"- {s}" for s in manifest.skipped]

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    results = run()
    print(f"Wrote results for {len(results)} tasks to {RESULTS_DIR}/")
    for name, result in results.items():
        print(f"  {name}: {result.get('status')} (n={result.get('n', 0)})")
