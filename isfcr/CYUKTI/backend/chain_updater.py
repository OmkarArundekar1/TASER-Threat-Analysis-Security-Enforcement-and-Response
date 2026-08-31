from neo4j_client import driver
from campaign_manager import campaign_manager

def update_attack_chain(
    campaign_id,
    current_technique
):
    try:
        with driver.session() as session:

            result = session.run(
                """
                MATCH (c:Campaign {
                    campaign_id: $campaign_id
                })
                RETURN c.last_technique AS previous
                """,
                campaign_id=campaign_id
            
            )

            record = result.single()

            if not record:
                print(f"[CHAIN] Campaign not found: {campaign_id}")
                return

            previous = record["previous"] or None
            context = campaign_manager.get_context_by_campaign_id(
                campaign_id
            )
            print("\n========== CHAIN DEBUG ==========")
            print(f"Campaign : {campaign_id}")
            print(f"Previous : {previous}")
            print(f"Current  : {current_technique}")
            
            if context:
                print(f"Skip Flag: {context.skip_next_chain_update}")
            else:
                print("Context : None")
            
            print("=================================\n")
            if context and context.skip_next_chain_update:
                print(
                    "[CHAIN] New/Reopened campaign. "
                    "Skipping first transition."
                )
                context.skip_next_chain_update = False
                print("[CHAIN] Skip flag cleared.")
                session.run(
                    """
                    MATCH (c:Campaign {
                        campaign_id:$campaign_id
                    })          
                    SET c.last_technique = $current
                    """,
                    campaign_id=campaign_id,
                    current=current_technique
                )
            
                return

            if previous is None:
                print(f"[CHAIN] Starting new attack chain with {current_technique}")

            elif previous == current_technique:
                print(f"[CHAIN] Duplicate technique ignored: {current_technique}")

            else:
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
            
                transition = session.run(
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
            
                if transition:
                    print("\n========== TRANSITION LEARNED ==========")
                    print(f"Source      : {previous}")
                    print(f"Destination : {current_technique}")
                    print(f"Count       : {transition['count']}")
                    print(f"Confidence  : {transition['confidence']:.2f}")
                    print(f"Created     : {transition['created_at']}")
                    print(f"Last Seen   : {transition['last_seen']}")
                    print("========================================")
            session.run(
                """
                MATCH (c:Campaign {
                    campaign_id: $campaign_id
                })
                SET c.last_technique = $current
                """,
                campaign_id=campaign_id,
                current=current_technique
            )
            

    except Exception as e:
        print(f"[CHAIN ERROR] {e}")