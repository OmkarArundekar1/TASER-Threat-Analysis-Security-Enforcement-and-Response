"""
chain_updater.py
===================
Records observed ATT&CK technique transitions as
(:Technique)-[:NEXT_TECHNIQUE {count, confidence, created_at, last_seen}]->(:Technique).

The transition's previous technique is supplied explicitly by the caller as
an immutable campaign_context.TransitionContext captured by CampaignManager
when the event was attached to its campaign. It is deliberately NOT read
back from Campaign.last_technique: by the time this runs, that property has
already been overwritten with the current technique by
neo4j_client.update_campaign_activity() and create_attack_event(), which
made every live transition look like a self-transition and disabled
learning entirely (research/BASELINE_AUDIT.md finding A2b;
research/STEP1_CHAIN_LEARNING.md).

This module therefore no longer reads or writes Campaign.last_technique.
That property keeps its meaning ("most recent resolved technique") and is
still maintained by update_campaign_activity()/create_attack_event().

Learning policy is unchanged by the fix:
  - no transition for the first resolved event of a new/reopened campaign
    (TransitionContext.segment_start);
  - no transition when there is no previous resolved technique;
  - self-transitions (A -> A) are not recorded.
"""

from neo4j_client import driver


def update_attack_chain(
    campaign_id,
    current_technique,
    transition
):
    if transition is None:
        raise ValueError(
            "update_attack_chain() requires the TransitionContext returned by "
            "CampaignManager.get_or_create_campaign_with_transition(); the previous "
            "technique cannot be recovered from Campaign.last_technique."
        )
    if (
        transition.campaign_id != campaign_id
        or transition.current_technique != current_technique
    ):
        raise ValueError(
            f"TransitionContext mismatch: got campaign={transition.campaign_id!r} "
            f"technique={transition.current_technique!r}, expected "
            f"campaign={campaign_id!r} technique={current_technique!r}"
        )

    previous = transition.previous_technique

    print("\n========== CHAIN DEBUG ==========")
    print(f"Campaign : {campaign_id}")
    print(f"Previous : {previous}")
    print(f"Current  : {current_technique}")
    print(f"Segment  : {'start' if transition.segment_start else 'continuation'}")
    print("=================================\n")

    if transition.segment_start:
        print(
            "[CHAIN] New/Reopened campaign. "
            "Skipping first transition."
        )
        return None

    if previous is None:
        print(f"[CHAIN] Starting new attack chain with {current_technique}")
        return None

    if previous == current_technique:
        print(f"[CHAIN] Duplicate technique ignored: {current_technique}")
        return None

    try:
        with driver.session() as session:
            print(f"[CHAIN] Learning {previous} -> {current_technique}")
            session.run(
                """
                MATCH (a:Technique {attack_id: $src})
                MATCH (b:Technique {attack_id: $dst})

                MERGE (a)-[r:NEXT_TECHNIQUE]->(b)

                ON CREATE SET
                    r.count = 1,
                    r.created_at = datetime(),
                    r.last_seen = datetime(),
                    r.confidence = 1.0

                ON MATCH SET
                    r.count = r.count + 1,
                    r.last_seen = datetime()
                """,
                src=previous,
                dst=current_technique
            )

            session.run(
                """
                MATCH (a:Technique {attack_id: $src})-[r:NEXT_TECHNIQUE]->()

                WITH collect(r) AS rels,
                     sum(r.count) AS total

                UNWIND rels AS rel

                SET rel.confidence =
                    CASE
                        WHEN total = 0 THEN 0.0
                        ELSE toFloat(rel.count) / total
                    END
                """,
                src=previous
            )

            transition_row = session.run(
                """
                MATCH (a:Technique {attack_id: $src})
                      -[r:NEXT_TECHNIQUE]->
                      (b:Technique {attack_id: $dst})

                RETURN
                    r.count AS count,
                    r.confidence AS confidence,
                    r.created_at AS created_at,
                    r.last_seen AS last_seen
                """,
                src=previous,
                dst=current_technique
            ).single()

            if transition_row is None:
                # MERGE only fires when both Technique nodes exist; say so
                # instead of reporting a transition that was not stored.
                print(
                    f"[CHAIN WARNING] {previous} -> {current_technique} not stored: "
                    "a Technique node is missing."
                )
                return None

            print("\n========== TRANSITION LEARNED ==========")
            print(f"Source      : {previous}")
            print(f"Destination : {current_technique}")
            print(f"Count       : {transition_row['count']}")
            print(f"Confidence  : {transition_row['confidence']:.2f}")
            print(f"Created     : {transition_row['created_at']}")
            print(f"Last Seen   : {transition_row['last_seen']}")
            print("========================================")
            return (previous, current_technique)

    except Exception as e:
        print(f"[CHAIN ERROR] {e}")
        return None
