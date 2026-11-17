from neo4j import GraphDatabase

from mitre_import.config import (
    URI,
    USERNAME,
    PASSWORD
)

from mitre_import.parser import get_by_type


driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)


def get_attack_id(external_references):

    if not external_references:
        return None

    for ref in external_references:

        if ref.get("source_name") == "mitre-attack":

            return ref.get("external_id")

    return None



def close():

    driver.close()