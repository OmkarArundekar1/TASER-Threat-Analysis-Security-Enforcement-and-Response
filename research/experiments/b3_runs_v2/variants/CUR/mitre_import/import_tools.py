from neo4j import GraphDatabase
from mitre_import.config import (
    URI,
    USERNAME,
    PASSWORD
)
from mitre_import.parser import get_by_type
from mitre_import.mapper import map_tool

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)
QUERY = """
MERGE (t:Tool {
    stix_id:$stix_id
})
ON CREATE SET
    t.created_at = datetime()
SET
    t.name = $name,
    t.description = $description,
    t.aliases = $aliases,
    t.created = datetime($created),
    t.modified = datetime($modified),
    t.revoked = $revoked,
    t.last_synced = datetime()
"""
def import_tools():
    tools = get_by_type("tool")
    imported = 0
    skipped = 0
    with driver.session() as session:
        for obj in tools:
            properties = map_tool(obj)
            if properties is None:
                skipped += 1
                continue

            session.run(
                QUERY,
                **properties
            )
            imported += 1
            if imported % 25 == 0:
                print(f"[Tool] Imported {imported}")

    print("\n========== Tool Import ==========")
    print(f"Imported : {imported}")
    print(f"Skipped  : {skipped}")
    print("=================================\n")

def close():
    driver.close()