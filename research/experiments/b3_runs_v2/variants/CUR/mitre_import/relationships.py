from neo4j import GraphDatabase
from mitre_import.config import (
    URI,
    USERNAME,
    PASSWORD
)
from mitre_import.parser import get_by_type
from mitre_import.utils import (
    RELATIONSHIP_TYPES,
    resolve_node_label
)
driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

QUERY = """
MATCH (source {stix_id:$source_ref})
MATCH (target {stix_id:$target_ref})
MERGE (source)-[r:%s]->(target)
ON CREATE SET
    r.created_at = datetime()
SET
    r.description = $description,
    r.created = datetime($created),
    r.modified = datetime($modified),
    r.revoked = $revoked,
    r.last_synced = datetime()
"""

def import_relationships():

    relationships = get_by_type("relationship")

    imported = 0
    skipped = 0

    with driver.session() as session:

        for obj in relationships:

            relationship_type = obj.get(
                "relationship_type"
            )
            neo4j_relationship = RELATIONSHIP_TYPES.get(
                relationship_type
            )
            if neo4j_relationship is None:
                skipped += 1
                continue
            source_ref = obj.get("source_ref")
            target_ref = obj.get("target_ref")
            if not source_ref or not target_ref:
                skipped += 1
                continue
            source_label = resolve_node_label(source_ref)
            target_label = resolve_node_label(target_ref)
            if source_label is None or target_label is None:
                skipped += 1
                continue
            query = QUERY % neo4j_relationship

            session.run(

                query,

                source_ref=source_ref,
                target_ref=target_ref,
                description=obj.get(
                    "description",
                    ""
                ),
                created=obj.get(
                    "created"
                ),
                modified=obj.get(
                    "modified"
                ),
                revoked=obj.get(
                    "revoked",
                    False
                )
            )
            imported += 1
            if imported % 500 == 0:
                print(
                    f"[Relationships] Imported {imported}"
                )
    print("\n========== Relationship Import ==========")
    print(f"Imported : {imported}")
    print(f"Skipped  : {skipped}")
    print("=========================================\n")

def close():
    driver.close()