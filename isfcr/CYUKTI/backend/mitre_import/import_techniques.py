from neo4j import GraphDatabase

from mitre_import.config import (
    URI,
    USERNAME,
    PASSWORD
)

from mitre_import.parser import get_by_type
from mitre_import.mapper import map_technique


driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)


QUERY = """
MERGE (t:Technique {
    attack_id:$attack_id
})

ON CREATE SET
    t.created_at = datetime()

SET
    t.stix_id = $stix_id,
    t.name = $name,
    t.description = $description,
    t.url = $url,

    t.created = datetime($created),
    t.modified = datetime($modified),

    t.revoked = $revoked,
    t.deprecated = $deprecated,

    t.attack_spec_version = $attack_spec_version,
    t.object_version = $object_version,

    t.platforms = $platforms,
    t.domains = $domains,

    t.kill_chain_phases = $kill_chain_phases,

    t.is_subtechnique = $is_subtechnique,

    t.last_synced = datetime()
"""


def import_techniques():

    techniques = get_by_type("attack-pattern")

    imported = 0
    skipped = 0

    with driver.session() as session:

        for obj in techniques:

            properties = map_technique(obj)

            if properties is None:
                skipped += 1
                continue

            session.run(
                QUERY,
                **properties
            )

            imported += 1

            if imported % 100 == 0:
                print(f"[Technique] Imported {imported}")

    print("\n========== Technique Import ==========")
    print(f"Imported : {imported}")
    print(f"Skipped  : {skipped}")
    print("======================================\n")


def close():
    driver.close()