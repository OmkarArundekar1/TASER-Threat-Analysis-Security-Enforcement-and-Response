NODE_LABELS = {
    "attack-pattern": "Technique",
    "course-of-action": "CourseOfAction",
    "intrusion-set": "ThreatActor",
    "tool": "Tool",
    "malware": "Malware"
}


RELATIONSHIP_TYPES = {
    "uses": "USES",
    "mitigates": "MITIGATES",
    "subtechnique-of": "SUBTECHNIQUE_OF",
    "revoked-by": "REVOKED_BY",
    "related-to": "RELATED_TO"
}
def get_stix_type(stix_id):
    return stix_id.split("--")[0]

def resolve_node_label(stix_id):
    stix_type = get_stix_type(stix_id)
    return NODE_LABELS.get(stix_type)

def get_attack_id(external_references):
    if not external_references:
        return None

    for ref in external_references:

        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")

    return None