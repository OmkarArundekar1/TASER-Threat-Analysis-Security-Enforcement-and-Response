from typing import List

from neo4j_client import driver
from attribution_models import HistoricalCampaign

def _deduplicate_preserve_order(items):
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result
    
class AttributionContext:
    def load_historical_campaigns(
        self,
    ) -> List[HistoricalCampaign]:

        with driver.session() as session:

            result = session.run(
                """
                MATCH (c:Campaign)

                WHERE c.status IN ['INACTIVE','ARCHIVED']

                MATCH
                    (c)-[:HAS_EVENT]->
                    (e:AttackEvent)
                    -[:MATCHES]->
                    (t:Technique)

                WITH
                    c,
                    e,
                    t

                ORDER BY
                    c.campaign_id,
                    e.first_seen

                RETURN
                    c.campaign_id AS campaign_id,
                    c.attacker_ip AS attacker,
                    c.victim_ip AS victim,
                    collect(
                        t.attack_id
                    ) AS techniques,

                    collect(
                        toString(
                            e.first_seen
                        )
                    ) AS timestamps
                """
            )
            campaigns = []
            for row in result:
                techniques = _deduplicate_preserve_order(
                    row["techniques"]
                )
                campaigns.append(
                    HistoricalCampaign(
                        campaign_id=row["campaign_id"],
                        attacker=row["attacker"],
                        victim=row["victim"],
                        techniques=techniques,
                        timestamps=row["timestamps"],
                    )
                )
            return campaigns

context=AttributionContext()