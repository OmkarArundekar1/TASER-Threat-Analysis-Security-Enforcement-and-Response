"""
benchmarks/run_benchmarks.py
==============================
Reproducible latency benchmark harness (BENCHMARKS.md, cross-cutting
engineering audit section 5). Measures real function calls against
whatever this environment actually has available (live Neo4j, a
trained XGBoost model, an enabled GNN model) -- never fabricates a
number for a component that isn't reachable; that component's row is
reported as `"skipped"` with the reason, not a guessed latency.

Usage:
    cd backend && python benchmarks/run_benchmarks.py [--iterations 50]

Writes a timestamped JSON report to benchmarks/results/ and prints a
p50/p95/p99/throughput summary table to stdout. Each result also
records which subsystems were actually available when the benchmark
ran, so a report is never read out of context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _time_it(fn, iterations: int) -> dict | None:
    """Runs fn() `iterations` times, discarding the first call (warm-up
    -- import caches, lazy model loads) from the statistics. Returns
    None if fn() raised on the very first (warm-up) call, so a
    genuinely unavailable component is reported as skipped rather than
    a crash."""
    try:
        fn()
    except Exception as e:
        return {"status": "skipped", "reason": f"{type(e).__name__}: {e}"}

    durations_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        durations_ms.append((time.perf_counter() - start) * 1000)

    return {
        "status": "measured",
        "iterations": iterations,
        "p50_ms": round(_percentile(durations_ms, 50), 3),
        "p95_ms": round(_percentile(durations_ms, 95), 3),
        "p99_ms": round(_percentile(durations_ms, 99), 3),
        "mean_ms": round(mean(durations_ms), 3),
        "throughput_per_sec": round(1000.0 / mean(durations_ms), 2) if mean(durations_ms) > 0 else None,
    }


def benchmark_mitre_resolution():
    from mitre_resolver import resolve_mitre
    alert = {"rule": {"id": "100500", "mitre": {"id": ["T1595"]}}}
    return _time_it(lambda: resolve_mitre(alert), ITERATIONS)


def benchmark_ioc_extraction():
    from realtime_socgraph import extract_iocs
    alert = {"data": {"src_ip": "192.168.56.106", "dest_ip": "192.168.56.105"}, "agent": {"name": "host01"}}
    return _time_it(lambda: extract_iocs(alert), ITERATIONS)


def benchmark_deduplication():
    import dedup_engine as dedup_module
    dedup_module.find_recent_duplicate = lambda fingerprint: None  # isolate from live Neo4j for this specific measurement
    engine = dedup_module.DeduplicationEngine()
    return _time_it(lambda: engine.is_duplicate("1.2.3.4", "10.0.0.5", "T1110", "5716", "001"), ITERATIONS)


def benchmark_neo4j_simple_query():
    from neo4j_client import driver
    def _run():
        with driver.session() as session:
            session.run("MATCH (c:Campaign) RETURN count(c) AS n").single()
    return _time_it(_run, ITERATIONS)


def benchmark_campaign_selection_scoring():
    from campaign_selection import BestCampaignSelector
    selector = BestCampaignSelector()
    def _run():
        candidates = [
            selector.build_candidate(f"CAMP_{i}", topology_similarity=0.5 + i * 0.01,
                                      technique_similarity=0.6, temporal_similarity=0.2,
                                      attacker_similarity=1.0, host_similarity=0.5)
            for i in range(10)
        ]
        selector.select(candidates)
    return _time_it(_run, ITERATIONS)


def benchmark_campaign_selection_live_candidate_discovery():
    """Live query + real signal computation for one real campaign, if
    one exists -- distinct from the pure in-memory scoring benchmark
    above."""
    from neo4j_client import driver
    from dashboard_api import _load_campaign_context, _discover_campaign_selection_candidates
    from campaign_selection import build_candidates_from_campaigns

    with driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        return {"status": "skipped", "reason": "No real campaign exists in this Neo4j instance."}
    campaign_id = row["id"]

    def _run():
        with driver.session() as session:
            context = _load_campaign_context(session, campaign_id)
            candidate_ids = _discover_campaign_selection_candidates(campaign_id, session)
            build_candidates_from_campaigns(context, candidate_ids, session)
    return _time_it(_run, ITERATIONS)


def benchmark_severity_prediction():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "models", "xgb_severity.json")
    if not os.path.exists(model_path):
        return {"status": "skipped", "reason": "No trained XGBoost model artifact present."}
    from datetime import datetime, timezone, timedelta
    from ml.runtime_predictor import RuntimeCampaignPredictor
    from campaign_context import CampaignContext
    predictor = RuntimeCampaignPredictor.from_model_path(model_path)
    now = datetime.now(timezone.utc)
    context = CampaignContext(campaign_id="BENCH", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
                               risk_score=500.0, last_technique="T1110", techniques={"T1110"},
                               first_seen=now - timedelta(minutes=5), last_seen=now)
    return _time_it(lambda: predictor.predict_for_campaign(context, "T1110", "bench-evt"), ITERATIONS)


def benchmark_gnn_embedding():
    from ml.gnn.inference import gnn_inference_service
    if not gnn_inference_service.available:
        return {"status": "skipped", "reason": "GNN_ENABLED is false or no model artifact loaded."}
    from neo4j_client import driver
    with driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        return {"status": "skipped", "reason": "No real campaign exists in this Neo4j instance."}
    campaign_id = row["id"]
    return _time_it(lambda: gnn_inference_service.embed_campaign(campaign_id, use_cache=False), ITERATIONS)


def benchmark_response_plan_generation():
    from campaign_context import CampaignContext
    from soar.generator import PlaybookGenerator
    from soar.response_plan import ResponsePlanGenerator
    campaign = CampaignContext(campaign_id="BENCH", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
                                risk_score=500.0, last_technique="T1110", techniques={"T1110"})
    playbook = PlaybookGenerator().generate(campaign)
    generator = ResponsePlanGenerator()
    return _time_it(lambda: generator.generate(campaign, playbook=playbook), ITERATIONS)


def benchmark_incident_overview_end_to_end_http():
    """Real HTTP round trip against the live dashboard_api process, if
    one is listening -- the closest available stand-in for "total
    end-to-end latency" without re-running the live alert pipeline
    (which has real side effects and isn't a pure function to time)."""
    import urllib.request
    from neo4j_client import driver
    with driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        return {"status": "skipped", "reason": "No real campaign exists in this Neo4j instance."}
    campaign_id = row["id"]
    url = f"http://localhost:5002/api/incidents/{campaign_id}/overview"

    def _run():
        with urllib.request.urlopen(url, timeout=10) as response:
            response.read()
    return _time_it(_run, ITERATIONS)


BENCHMARKS = {
    "mitre_resolution": benchmark_mitre_resolution,
    "ioc_extraction": benchmark_ioc_extraction,
    "deduplication": benchmark_deduplication,
    "neo4j_simple_query": benchmark_neo4j_simple_query,
    "campaign_selection_scoring_in_memory": benchmark_campaign_selection_scoring,
    "campaign_selection_live_candidate_discovery": benchmark_campaign_selection_live_candidate_discovery,
    "severity_prediction": benchmark_severity_prediction,
    "gnn_embedding": benchmark_gnn_embedding,
    "response_plan_generation": benchmark_response_plan_generation,
    "incident_overview_end_to_end_http": benchmark_incident_overview_end_to_end_http,
}

ITERATIONS = 20


def main():
    global ITERATIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    ITERATIONS = args.iterations

    results = {}
    print(f"{'benchmark':<45} {'status':<10} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'ops/sec':>10}")
    print("-" * 100)
    for name, fn in BENCHMARKS.items():
        result = fn()
        results[name] = result
        if result["status"] == "measured":
            print(f"{name:<45} {'ok':<10} {result['p50_ms']:>10} {result['p95_ms']:>10} {result['p99_ms']:>10} {result['throughput_per_sec']:>10}")
        else:
            print(f"{name:<45} {'skipped':<10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}  ({result['reason']})")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": ITERATIONS,
        "results": results,
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
