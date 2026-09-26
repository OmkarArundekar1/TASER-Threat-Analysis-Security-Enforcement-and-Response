"""
review/export_review_queue.py
================================
Exports human-fillable review queues (Phase 11) -- the mechanism that
actually unblocks the NOT_MEASURED items in results/summary.md. Never
pre-fills a verdict/relevance judgment itself; only assembles the real
evidence a reviewer needs to make one, in a structured, low-friction
format (CSV).

Usage:
    cd evaluation && python review/export_review_queue.py --which investigation
    cd evaluation && python review/export_review_queue.py --which rag --source mitre_semantic
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# The 3 real Phase 21 investigations -- see review/phase21_real_investigation_validation.md.
# Frozen, historical, quoted verbatim (not re-derived) so exporting this queue never touches
# that file or re-runs the investigation.
_PHASE21_INVESTIGATIONS = [
    {"investigation_id": "CAMP_427A075C", "attacker": "192.168.56.106", "victim": "192.168.56.105",
     "final_investigation_confidence": 0.1835, "top_hypothesis_technique": "T1595",
     "evidence_summary": "Reconnaissance-only campaign (T1595), risk_score 8330. See "
                          "review/phase21_real_investigation_validation.md Section 7 for the full evidence trail."},
    {"investigation_id": "CAMP_1429ADB4", "attacker": "192.168.56.105", "victim": "pes1ug23cs411-VirtualBox",
     "final_investigation_confidence": 0.0687, "top_hypothesis_technique": "T1210",
     "evidence_summary": "Mixed exploitation campaign (T1055/T1059/T1059.007/T1190/T1210/T1595.002), "
                          "risk_score 34820. See review/phase21_real_investigation_validation.md Section 7."},
    {"investigation_id": "CAMP_D8605E81", "attacker": "192.168.56.106", "victim": "pes1ug23cs411-VirtualBox",
     "final_investigation_confidence": 0.0963, "top_hypothesis_technique": "T1110",
     "evidence_summary": "Brute-force campaign (T1110/T1110.001), risk_score 630. "
                          "See review/phase21_real_investigation_validation.md Section 7."},
]

# Real, observed-in-this-project candidate query texts, grounded in the actual
# scenarios this lab has run (evaluation/scenarios/scenarios.json) -- NOT
# pre-judged for relevance; a reviewer fills in relevant_ids after actually
# running each query against the retriever and inspecting its results.
_CANDIDATE_QUERIES = {
    "mitre_semantic": [
        "network reconnaissance active scanning technique",
        "SSH brute force repeated failed authentication",
        "service exhaustion denial of service flood",
        "process injection privilege escalation",
        "command and scripting interpreter JavaScript execution",
    ],
    "campaign_narrative": [
        "campaign targeting pes1ug23cs411-VirtualBox from 192.168.56.106",
        "campaign with mixed exploitation techniques and high risk score",
        "repeated brute-force campaign against the same victim host",
    ],
    "gnn_topology": [
        "campaign topologically similar to CAMP_D8605E81 (brute force)",
        "campaign topologically similar to CAMP_1429ADB4 (mixed exploitation)",
    ],
}


def export_investigation_queue() -> str:
    path = os.path.join(_OUT_DIR, "investigation_verdict_queue.csv")
    fields = ["investigation_id", "attacker", "victim", "final_investigation_confidence",
              "top_hypothesis_technique", "evidence_summary",
              "verdict (fill in: CORRECT|INCORRECT|PARTIALLY_CORRECT|INSUFFICIENT_EVIDENCE)", "reviewer"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in _PHASE21_INVESTIGATIONS:
            writer.writerow({
                **row,
                "verdict (fill in: CORRECT|INCORRECT|PARTIALLY_CORRECT|INSUFFICIENT_EVIDENCE)": "",
                "reviewer": "",
            })
    print(f"Wrote {len(_PHASE21_INVESTIGATIONS)} rows -> {path}")
    print("Fill in the verdict + reviewer columns, then run "
          "review/import_reviewed.py --which investigation to promote this into "
          "evaluation/labels/investigation_verdicts.json.")
    return path


def export_rag_queue(source: str) -> str:
    if source not in _CANDIDATE_QUERIES:
        raise ValueError(f"Unknown RAG source {source!r}; choose from {list(_CANDIDATE_QUERIES)}")
    path = os.path.join(_OUT_DIR, f"rag_query_queue_{source}.csv")
    fields = ["query_id", "query_text", "relevant_ids (fill in, comma-separated doc/campaign ids actually relevant)",
              "reviewer"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, text in enumerate(_CANDIDATE_QUERIES[source]):
            writer.writerow({
                "query_id": f"{source.upper()}-Q{i+1:02d}",
                "query_text": text,
                "relevant_ids (fill in, comma-separated doc/campaign ids actually relevant)": "",
                "reviewer": "",
            })
    print(f"Wrote {len(_CANDIDATE_QUERIES[source])} candidate queries -> {path}")
    print(f"For each query: run it against the real {source} retriever, inspect the results, fill in which "
          f"returned items are ACTUALLY relevant, then run review/import_reviewed.py --which rag --source {source}.")
    return path


def export_mitre_queue(dataset_name: str = "mitre_mapping_v0") -> str:
    """Phase 3: one row per AUTO_PROPOSED MITRE record, showing the raw
    alert evidence and the AI-proposed expected_mitre_techniques
    alongside CYUKTI's own live resolution (`cyukti_prediction`, kept
    in a clearly separate column) -- so a reviewer can see both without
    either one being mistaken for the other. The reviewer's decision
    (ACCEPT/MODIFY/REJECT/UNCERTAIN) determines what, if anything, gets
    promoted to HUMAN_REVIEWED; a MODIFY row's corrected_techniques
    column is what actually gets imported, not the original proposal."""
    from ground_truth import store, raw_alerts as raw_alerts_module
    from mitre_resolver import resolve_mitre

    records, _ = store.load_best_available(dataset_name)
    raw = raw_alerts_module.load_raw_alerts(dataset_name)
    if not records:
        print(f"No records found for {dataset_name} -- nothing to export.")
        return ""

    path = os.path.join(_OUT_DIR, f"mitre_review_queue_{dataset_name}.csv")
    fields = [
        "review_id", "sample_id", "scenario_id", "timestamp", "raw_event_reference",
        "attacker", "victim", "raw_rule_id", "raw_rule_description", "raw_rule_native_mitre",
        "ai_proposed_expected_mitre_techniques", "cyukti_prediction (live resolve_mitre() output, NOT ground truth)",
        "labeling_method", "evidence_reference",
        "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)",
        "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)",
        "reviewer", "rationale",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(records):
            alert = raw.get(r.sample_id, {})
            rule = alert.get("rule", {}) or {}
            live = resolve_mitre(alert) if alert else None
            writer.writerow({
                "review_id": f"REV-MITRE-{i+1:04d}",
                "sample_id": r.sample_id,
                "scenario_id": r.scenario_id,
                "timestamp": r.timestamp,
                "raw_event_reference": r.raw_event_id,
                "attacker": r.attacker_identity,
                "victim": r.victim_identity,
                "raw_rule_id": rule.get("id"),
                "raw_rule_description": rule.get("description"),
                "raw_rule_native_mitre": json.dumps(rule.get("mitre")) if rule.get("mitre") else "",
                "ai_proposed_expected_mitre_techniques": ",".join(r.expected_mitre_techniques),
                "cyukti_prediction (live resolve_mitre() output, NOT ground truth)":
                    ",".join(live.technique_ids or []) if live else "",
                "labeling_method": r.labeling_method,
                "evidence_reference": r.evidence_reference,
                "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)": "",
                "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)": "",
                "reviewer": "",
                "rationale": "",
            })
    print(f"Wrote {len(records)} rows -> {path}")
    return path


def export_attribution_queue(dataset_name: str = "attribution_v0") -> str:
    """Phase 4: ground truth here is the campaign's own real,
    directly-observed attacker_ip -- shown for reference, but the
    reviewer independently confirms or overrides it (UNKNOWN/UNDETERMINED
    is a valid, honest answer), never derived from CYUKTI's own
    attribution/clustering output (shown separately as cyukti_prediction)."""
    from ground_truth import store

    records, _ = store.load_best_available(dataset_name)
    if not records:
        print(f"No records found for {dataset_name} -- nothing to export.")
        return ""

    path = os.path.join(_OUT_DIR, f"attribution_review_queue_{dataset_name}.csv")
    fields = [
        "review_id", "sample_id", "scenario_id", "raw_event_reference",
        "raw_observed_attacker_ip (network fact, NOT CYUKTI output)",
        "labeling_method", "evidence_reference",
        "attacker_identity (fill in: confirm the IP, correct it, or write UNKNOWN/UNDETERMINED)",
        "confidence (fill in: HIGH|MEDIUM|LOW)",
        "evidence_reference_for_decision", "reviewer",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(records):
            writer.writerow({
                "review_id": f"REV-ATTR-{i+1:04d}",
                "sample_id": r.sample_id,
                "scenario_id": r.scenario_id,
                "raw_event_reference": r.raw_event_id,
                "raw_observed_attacker_ip (network fact, NOT CYUKTI output)": r.expected_attribution,
                "labeling_method": r.labeling_method,
                "evidence_reference": r.evidence_reference,
                "attacker_identity (fill in: confirm the IP, correct it, or write UNKNOWN/UNDETERMINED)": "",
                "confidence (fill in: HIGH|MEDIUM|LOW)": "",
                "evidence_reference_for_decision": "",
                "reviewer": "",
            })
    print(f"Wrote {len(records)} rows -> {path}")
    return path


def export_threat_qualification_queue(dataset_name: str = "threat_qualification_v0") -> str:
    """Phase 6, highest priority (preliminary agreement was only 8.3%).
    Shows the real cti_score for transparency but the reviewer answers
    the ground-truth question ("was this actually a threat?")
    independently -- cti_score/CYUKTI's derived classification is never
    the label itself."""
    from ground_truth import store
    from neo4j_client import driver
    from cti_confidence_engine import PUBLISH_THRESHOLD, NOT_THREAT_THRESHOLD

    records, _ = store.load_best_available(dataset_name)
    if not records:
        print(f"No records found for {dataset_name} -- nothing to export.")
        return ""

    path = os.path.join(_OUT_DIR, f"threat_qualification_review_queue_{dataset_name}.csv")
    fields = [
        "review_id", "sample_id", "scenario_id", "raw_event_reference (real Campaign.campaign_id)",
        "attacker", "victim", "ai_proposed_expected_threat_status",
        "live_cti_score (reference only, NOT the label)",
        "cyukti_prediction (derived from cti_score via cti_confidence_engine thresholds, NOT ground truth)",
        "evidence_reference",
        "ground_truth_threat_status (fill in: NOT_THREAT|SUSPICIOUS|QUALIFIED_THREAT)",
        "confidence (fill in: HIGH|MEDIUM|LOW)", "rationale", "reviewer",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with driver.session() as s:
            for i, r in enumerate(records):
                row = s.run("MATCH (c:Campaign {campaign_id:$id}) RETURN c.cti_score AS cti_score",
                            id=r.raw_event_id).single()
                cti_score = row["cti_score"] if row else None
                if cti_score is None:
                    cyukti_pred = "N/A (not yet scored)"
                elif cti_score >= PUBLISH_THRESHOLD:
                    cyukti_pred = "QUALIFIED_THREAT"
                elif cti_score >= NOT_THREAT_THRESHOLD:
                    cyukti_pred = "SUSPICIOUS"
                else:
                    cyukti_pred = "NOT_THREAT"
                writer.writerow({
                    "review_id": f"REV-TQ-{i+1:04d}",
                    "sample_id": r.sample_id,
                    "scenario_id": r.scenario_id,
                    "raw_event_reference (real Campaign.campaign_id)": r.raw_event_id,
                    "attacker": r.attacker_identity,
                    "victim": r.victim_identity,
                    "ai_proposed_expected_threat_status": r.expected_threat_status,
                    "live_cti_score (reference only, NOT the label)": cti_score,
                    "cyukti_prediction (derived from cti_score via cti_confidence_engine thresholds, NOT ground truth)": cyukti_pred,
                    "evidence_reference": r.evidence_reference,
                    "ground_truth_threat_status (fill in: NOT_THREAT|SUSPICIOUS|QUALIFIED_THREAT)": "",
                    "confidence (fill in: HIGH|MEDIUM|LOW)": "",
                    "rationale": "",
                    "reviewer": "",
                })
    print(f"Wrote {len(records)} rows -> {path}")
    return path


def export_campaign_correlation_queue(dataset_name: str = "campaign_correlation_v0", pairs_per_session: int = 5) -> str:
    """Phase 5: pairwise same-campaign review. The 3 existing
    session-boundary records each cover MANY real Campaign nodes (up to
    25) -- reviewing all C(46,2)~1000 possible pairs is impractical, so
    a representative sample is drawn per session (within-session pairs,
    a real candidate for 'same real campaign, wrongly split' vs.
    cross-session pairs, a real candidate for 'correctly kept separate')
    -- disclosed explicitly here and in the resulting report, not
    presented as an exhaustive pairwise evaluation."""
    import random

    from ground_truth import store

    records, _ = store.load_best_available(dataset_name)
    if not records:
        print(f"No records found for {dataset_name} -- nothing to export.")
        return ""

    sessions = {r.expected_campaign_id: r.raw_event_id.split("|") for r in records if r.raw_event_id}
    rng = random.Random(42)  # fixed seed -- reproducible sample, not cherry-picked per run
    pairs = []
    session_keys = list(sessions)
    for session_key, campaign_ids in sessions.items():
        within = list(campaign_ids)
        rng.shuffle(within)
        for a, b in zip(within[::2], within[1::2]):
            pairs.append((a, b, session_key, session_key, "within_session (candidate: same real campaign)"))
            if len(pairs) >= pairs_per_session * (session_keys.index(session_key) + 1):
                break
    for i in range(len(session_keys)):
        for j in range(i + 1, len(session_keys)):
            a_pool, b_pool = sessions[session_keys[i]], sessions[session_keys[j]]
            if a_pool and b_pool:
                pairs.append((rng.choice(a_pool), rng.choice(b_pool), session_keys[i], session_keys[j],
                              "cross_session (candidate: different real campaign)"))

    path = os.path.join(_OUT_DIR, f"campaign_correlation_review_queue_{dataset_name}.csv")
    fields = [
        "review_id", "campaign_a", "campaign_b", "session_a", "session_b", "sampling_rationale",
        "same_campaign_ground_truth (fill in: SAME|DIFFERENT|UNCERTAIN)", "evidence_reference", "reviewer",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (a, b, sa, sb, rationale) in enumerate(pairs):
            writer.writerow({
                "review_id": f"REV-CC-{i+1:04d}", "campaign_a": a, "campaign_b": b,
                "session_a": sa, "session_b": sb, "sampling_rationale": rationale,
                "same_campaign_ground_truth (fill in: SAME|DIFFERENT|UNCERTAIN)": "",
                "evidence_reference": "", "reviewer": "",
            })
    print(f"Wrote {len(pairs)} sampled pairs (of {len(records)} session-boundary records covering "
          f"{sum(len(v) for v in sessions.values())} real campaigns) -> {path}")
    print("NOTE: this is a representative SAMPLE, not an exhaustive pairwise evaluation -- see the "
          "module docstring / final report for why.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["investigation", "rag", "mitre", "attribution",
                                             "threat_qualification", "campaign_correlation"], required=True)
    parser.add_argument("--source", choices=list(_CANDIDATE_QUERIES), default=None)
    args = parser.parse_args()

    if args.which == "investigation":
        export_investigation_queue()
    elif args.which == "mitre":
        export_mitre_queue()
    elif args.which == "attribution":
        export_attribution_queue()
    elif args.which == "threat_qualification":
        export_threat_qualification_queue()
    elif args.which == "campaign_correlation":
        export_campaign_correlation_queue()
    else:
        if not args.source:
            for source in _CANDIDATE_QUERIES:
                export_rag_queue(source)
        else:
            export_rag_queue(args.source)
