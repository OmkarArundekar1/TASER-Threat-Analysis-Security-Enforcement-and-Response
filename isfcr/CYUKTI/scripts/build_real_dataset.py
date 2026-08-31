"""
Runs the REAL ml.dataset_builder against the 33 real campaigns currently
in Neo4j (not synthetic fixtures). Reports exactly what happened for
each campaign — built, skipped, or errored, and why — then reports data
quality on the resulting dataset. Read-only against Neo4j; only writes
the output dataset CSV under ml/datasets/.
"""

import os
import sys
import traceback

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from campaign_context import CampaignContext  # noqa: E402
from neo4j_client import driver  # noqa: E402


def load_full_campaign_context(session, campaign_id):
    record = session.run(
        """
        MATCH (c:Campaign {campaign_id: $campaign_id})
        OPTIONAL MATCH (a:Attacker)-[:LAUNCHED]->(c)
        OPTIONAL MATCH (c)-[:TARGETS]->(h:Host)
        OPTIONAL MATCH (c)-[pred:LIKELY_NEXT]->(pt:Technique)
        RETURN c, a.ip AS attacker_ip, h.ip AS victim_ip,
               pt.attack_id AS predicted_next, pred.confidence AS prediction_confidence
        """,
        campaign_id=campaign_id,
    ).single()

    if record is None or record["c"] is None:
        return None, None, None

    # Real chronological order, straight from AttackEvent.first_seen -- NOT
    # from collect(DISTINCT t.attack_id), which is an unordered, deduplicated
    # set and loses both event order and repeat occurrences (see Phase 18
    # investigation: this used to feed a `sorted(set(...))[-1]` alphabetical
    # fallback for "final technique", which is not chronological at all and
    # gets real parent/sub-technique pairs like T1110/T1110.001 backwards).
    # event_id is a stable tie-break; no two real events currently share a
    # first_seen timestamp, but ties are not guaranteed to stay impossible.
    event_rows = session.run(
        """
        MATCH (:Campaign {campaign_id: $campaign_id})-[:HAS_EVENT]->(e:AttackEvent)
        RETURN e.attack_id AS attack_id, e.first_seen AS first_seen, e.event_id AS event_id
        ORDER BY e.first_seen, e.event_id
        """,
        campaign_id=campaign_id,
    ).data()
    technique_sequence = [r["attack_id"] for r in event_rows if r["attack_id"]]
    sample_event_id = event_rows[0]["event_id"] if event_rows else None

    c = record["c"]
    context = CampaignContext(
        campaign_id=campaign_id,
        attacker_ip=record["attacker_ip"] or "",
        victim_ip=record["victim_ip"] or "",
        risk_score=c.get("risk_score", 0.0) or 0.0,
        last_technique=c.get("last_technique"),
        first_seen=c.get("first_seen").to_native() if c.get("first_seen") else None,
        last_seen=c.get("last_seen").to_native() if c.get("last_seen") else None,
        predicted_next=record["predicted_next"],
        prediction_confidence=record["prediction_confidence"] or 0.0,
    )
    context.techniques = set(technique_sequence)
    context.attack_chain = technique_sequence
    return context, sample_event_id, technique_sequence


def main():
    from ml.dataset_builder import builder
    from ml.dataset_utils import FEATURE_COLUMNS

    with driver.session() as session:
        campaign_ids = [r["campaign_id"] for r in session.run(
            "MATCH (c:Campaign) RETURN c.campaign_id AS campaign_id ORDER BY c.first_seen"
        )]

    built, skipped, errored = [], [], []

    for campaign_id in campaign_ids:
        with driver.session() as session:
            context, sample_event_id, technique_sequence = load_full_campaign_context(session, campaign_id)

        if context is None:
            skipped.append((campaign_id, "campaign not found"))
            continue
        if not technique_sequence:
            skipped.append((campaign_id, "no techniques linked"))
            continue

        # The true last technique is the chronologically last AttackEvent,
        # not context.last_technique (nulled by neo4j_client.archive_campaign_db
        # once a campaign is actually archived) and not an alphabetical sort
        # of the technique set (not chronological -- see Phase 18).
        final_attack_id = technique_sequence[-1]

        try:
            record = builder.build(
                campaign_context=context,
                final_attack_id=final_attack_id,
                attacker_ip=context.attacker_ip,
                event_id=sample_event_id or f"{campaign_id}-evt",
                attributed_actor="unknown",
                technique_sequence=technique_sequence,
                actual_actor=None,
            )
            built.append((campaign_id, record))
        except Exception as e:
            errored.append((campaign_id, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    print(f"\n=== Dataset build summary ===")
    print(f"Campaigns discovered: {len(campaign_ids)}")
    print(f"Built:   {len(built)}")
    print(f"Skipped: {len(skipped)} -> {skipped}")
    print(f"Errored: {len(errored)} -> {errored}")

    if not built:
        print("\nNo records built — nothing further to report.")
        return

    print(f"Total rows now in dataset: {builder.dataset_size()}")
    n_with_pred = sum(1 for _, r in built if getattr(r, "next_technique", ""))
    print(f"Rows with next_technique populated: {n_with_pred} / {len(built)}")

    df = builder.load_dataset()
    print(f"\n=== Data quality on the {len(df)}-row dataset ===")
    print("Missing values per column (non-zero only):")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing.to_dict() if len(missing) else "  none")

    print(f"\nDuplicate rows (all columns identical): {df.duplicated().sum()}")
    print(f"Duplicate rows (feature columns only, ignoring ids/labels): "
          f"{df.duplicated(subset=[c for c in FEATURE_COLUMNS if c in df.columns]).sum()}")

    print(f"\nSeverity label distribution:\n{df['severity'].value_counts().to_dict()}")
    print(f"prediction_correct distribution (NaN = NOT_APPLICABLE):\n{df['prediction_correct'].value_counts(dropna=False).to_dict()}")
    print(f"attribution_correct distribution:\n{df['attribution_correct'].value_counts().to_dict()}")
    print(f"\nUnique campaign_ids in dataset: {df['campaign_id'].nunique()} (leakage check: should equal row count "
          f"if one row per campaign) -> rows={len(df)}")


if __name__ == "__main__":
    main()
