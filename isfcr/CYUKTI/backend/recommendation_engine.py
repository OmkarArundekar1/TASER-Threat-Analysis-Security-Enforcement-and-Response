from neo4j_client import driver


def get_recommendations(technique_id):

    with driver.session() as session:

        result = session.run(
            """
            MATCH (t:Technique {attack_id:$attack_id})

            MATCH (m:CourseOfAction)-[:MITIGATES]->(t)

            RETURN
                m.mitigation_id AS mitigation_id,
                m.name AS mitigation_name,
                m.description AS description
            ORDER BY mitigation_name
            """,
            attack_id=technique_id
        )

        recommendations = []

        for row in result:

            recommendations.append({

                "recommendation": row["mitigation_name"],

                "priority": "MITRE",

                "mitre_mitigation": row["mitigation_id"],

                "reason": row["description"],

                "predicted_technique": technique_id,

                "traceability": (
                    f"MITRE ATT&CK → "
                    f"{row['mitigation_id']}"
                )

            })

        if recommendations:
            return recommendations

        return [{
            "recommendation": "No mitigation available",
            "priority": "LOW",
            "mitre_mitigation": "N/A",
            "reason": "No official ATT&CK mitigation found.",
            "predicted_technique": technique_id,
            "traceability": "MITRE ATT&CK"
        }]


def get_recommendation_strings(technique_id):

    return [
        r["recommendation"]
        for r in get_recommendations(technique_id)
    ]