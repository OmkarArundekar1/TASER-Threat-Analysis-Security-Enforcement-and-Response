"""
phase18_next_technique_diagnosis.py
====================================
Read-only diagnosis of the next_technique / predicted_next / prediction_correct
pipeline. Does not write to Neo4j or the dataset. For every campaign, computes
the TRUE chronological event order directly from AttackEvent.first_seen (the
only field guaranteed not to be nulled at archive time -- see
neo4j_client.archive_campaign_db, which nulls Campaign.last_technique and
Campaign.predicted_next but never touches AttackEvent nodes), and compares it
against what build_real_dataset.py currently derives.
"""
import os
import sys
from collections import Counter

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from neo4j_client import driver  # noqa: E402
from mitre_mapper import MITRE_TO_STAGE  # noqa: E402

with driver.session() as session:
    campaigns = session.run(
        """
        MATCH (c:Campaign)
        OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)
        OPTIONAL MATCH (c)-[pred:LIKELY_NEXT]->(pt:Technique)
        WITH c, e, pt, pred
        ORDER BY e.first_seen
        WITH c, collect({attack_id: e.attack_id, first_seen: e.first_seen, event_id: e.event_id}) AS events,
             pt.attack_id AS likely_next, pred.confidence AS likely_next_confidence
        RETURN c.campaign_id AS campaign_id, c.status AS status,
               c.last_technique AS stored_last_technique,
               c.predicted_next AS stored_predicted_next_property,
               c.reconstructed AS reconstructed,
               events, likely_next, likely_next_confidence
        """
    ).data()

print(f"Total campaigns: {len(campaigns)}\n")

null_reasons = Counter()
rows_report = []

for c in campaigns:
    cid = c["campaign_id"]
    events = [e for e in c["events"] if e.get("attack_id")]
    events_sorted = sorted(events, key=lambda e: (e["first_seen"], e["event_id"]))
    n_events = len(events_sorted)
    technique_seq = [e["attack_id"] for e in events_sorted]

    # duplicate/equal timestamp check
    timestamps = [e["first_seen"] for e in events_sorted]
    has_tie = len(timestamps) != len(set(timestamps))

    true_final = technique_seq[-1] if technique_seq else None

    # what build_real_dataset.py currently computes
    stored_last = c["stored_last_technique"]
    alpha_fallback = sorted(set(technique_seq))[-1] if technique_seq else None
    current_pipeline_final = stored_last or alpha_fallback

    mismatch_vs_true = (current_pipeline_final != true_final)

    likely_next = c["likely_next"]

    if n_events == 0:
        reason = "MISSING_EVENT_ORDER"
    elif n_events == 1:
        reason = "SINGLE_EVENT_CAMPAIGN"
    elif likely_next is None and c["reconstructed"]:
        reason = "RECONSTRUCTED_CAMPAIGN"
    elif likely_next is None:
        reason = "NO_TRANSITION_RELATIONSHIP"
    else:
        reason = "HAS_PREDICTION"

    null_reasons[reason] += 1

    rows_report.append({
        "campaign_id": cid,
        "status": c["status"],
        "reconstructed": bool(c["reconstructed"]),
        "n_events": n_events,
        "technique_seq": technique_seq,
        "has_timestamp_tie": has_tie,
        "true_final_technique": true_final,
        "stored_last_technique_property": stored_last,
        "alphabetical_fallback_would_pick": alpha_fallback,
        "current_pipeline_final_attack_id": current_pipeline_final,
        "mismatch_vs_true_final": mismatch_vs_true,
        "likely_next_stored": likely_next,
        "likely_next_confidence": c["likely_next_confidence"],
        "prediction_would_be_correct_if_compared_to_true_final": (likely_next == true_final) if likely_next else None,
        "null_reason": reason,
    })

print("=== Reason distribution for next_technique nullability ===")
for reason, count in null_reasons.most_common():
    print(f"  {reason}: {count}")

print(f"\n=== Campaigns where current pipeline's final_attack_id != TRUE chronological final ===")
mismatches = [r for r in rows_report if r["mismatch_vs_true_final"]]
print(f"Count: {len(mismatches)} / {len(rows_report)}")
for r in mismatches:
    print(f"  {r['campaign_id']}: seq={r['technique_seq']} true_final={r['true_final_technique']} "
          f"stored_last_technique={r['stored_last_technique_property']} "
          f"alpha_fallback={r['alphabetical_fallback_would_pick']} -> "
          f"pipeline_used={r['current_pipeline_final_attack_id']}")

print(f"\n=== Campaigns with a stored LIKELY_NEXT prediction (n={sum(1 for r in rows_report if r['likely_next_stored'])}) ===")
for r in rows_report:
    if r["likely_next_stored"]:
        print(f"  {r['campaign_id']}: seq={r['technique_seq']} true_final={r['true_final_technique']} "
              f"likely_next={r['likely_next_stored']} (conf={r['likely_next_confidence']}) "
              f"pipeline_final_used={r['current_pipeline_final_attack_id']} "
              f"would_be_correct_vs_TRUE_final={r['prediction_would_be_correct_if_compared_to_true_final']}")

print(f"\n=== Timestamp ties (ambiguous event order) ===")
ties = [r for r in rows_report if r["has_timestamp_tie"]]
print(f"Count: {len(ties)}")
for r in ties:
    print(f"  {r['campaign_id']}: seq={r['technique_seq']}")

print(f"\n=== Single-event campaigns (next_technique should be NOT_APPLICABLE, not NULL/0) ===")
singles = [r for r in rows_report if r["n_events"] == 1]
print(f"Count: {len(singles)}")

print(f"\n=== Multi-event campaigns' full technique sequences (for parent/sub-technique transition review) ===")
multis = [r for r in rows_report if r["n_events"] > 1]
for r in multis:
    print(f"  {r['campaign_id']} ({r['status']}, reconstructed={r['reconstructed']}): {r['technique_seq']}")

# parent/sub-technique transition semantic check
print("\n=== Parent/sub-technique adjacency check across all sequences ===")
PAIRS_OF_INTEREST = [
    ("T1110.001", "T1110"), ("T1110", "T1110.001"),
    ("T1595", "T1595.002"), ("T1595.002", "T1595"),
    ("T1059", "T1059.007"), ("T1059.007", "T1059"),
]
for r in multis:
    seq = r["technique_seq"]
    for i in range(len(seq) - 1):
        pair = (seq[i], seq[i + 1])
        if pair in PAIRS_OF_INTEREST:
            print(f"  {r['campaign_id']}: {pair[0]} -> {pair[1]}  (adjacent in real chronological sequence)")

with driver.session() as session:
    nt = session.run(
        "MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique) RETURN a.attack_id AS src, b.attack_id AS dst, r.count AS count ORDER BY r.count DESC"
    ).data()
print(f"\n=== Current NEXT_TECHNIQUE edges (global, aggregated) ===")
for r in nt:
    print(f"  {r['src']} -> {r['dst']}  count={r['count']}")

print(f"\nTotal campaigns: {len(rows_report)}")
print(f"Campaigns with n_events>=2 (valid transition possible): {len(multis)}")
print(f"Campaigns with n_events==1: {len(singles)}")
print(f"Campaigns with n_events==0: {sum(1 for r in rows_report if r['n_events']==0)}")
