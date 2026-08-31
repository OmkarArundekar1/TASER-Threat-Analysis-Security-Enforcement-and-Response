from mitre_import.utils import get_attack_id

def extract_kill_chain_phases(stix_obj):
    phases = []
    for phase in stix_obj.get("kill_chain_phases", []):
        name = phase.get("phase_name")
        if name:
            phases.append(name)
    return phases

def map_technique(stix_obj):
    attack_id = get_attack_id(
        stix_obj.get(
            "external_references",
            []
        )
    )
    if attack_id is None:
        return None
    return {
        "attack_id": attack_id,
        "stix_id": stix_obj["id"],
        "name": stix_obj.get(
            "name",
            ""
        ),
        "description": stix_obj.get(
            "description",
            ""
        ),
        "url": stix_obj.get(
            "external_references",
            [{}]
        )[0].get(
            "url",
            ""
        ),
        "created": stix_obj.get(
            "created"
        ),
        "modified": stix_obj.get(
            "modified"
        ),
        "revoked": stix_obj.get(
            "revoked",
            False
        ),
        "deprecated": stix_obj.get(
            "x_mitre_deprecated",
            False
        ),
        "platforms": stix_obj.get(
            "x_mitre_platforms",
            []
        ),
        "domains": stix_obj.get(
            "x_mitre_domains",
            []
        ),
        "attack_spec_version": stix_obj.get(
            "x_mitre_attack_spec_version",
            ""
        ),
        "object_version": stix_obj.get(
            "x_mitre_version",
            ""
        ),
        "is_subtechnique": stix_obj.get(
            "x_mitre_is_subtechnique",
            False
        ),
        "kill_chain_phases":
            extract_kill_chain_phases(
                stix_obj
            )
    }

def map_threat_actor(stix_obj):
    attack_id = get_attack_id(
        stix_obj.get(
            "external_references",
            []
        )
    )
    if attack_id is None:
        return None
    return {
        "actor_id": attack_id,
        "stix_id": stix_obj["id"],
        "name": stix_obj.get(
            "name",
            ""
        ),
        "description": stix_obj.get(
            "description",
            ""
        ),
        "url": stix_obj.get(
            "external_references",
            [{}]
        )[0].get(
            "url",
            ""
        ),
        "aliases": stix_obj.get(
            "aliases",
            []
        ),
        "created": stix_obj.get("created"),
        "modified": stix_obj.get("modified"),
        "revoked": stix_obj.get(
            "revoked",
            False
        )
    }

def map_course_of_action(stix_obj):
    mitigation_id = get_attack_id(
        stix_obj.get(
            "external_references",
            []
        )
    )
    if mitigation_id is None:
        return None
    return {
        "mitigation_id": mitigation_id,
        "stix_id": stix_obj["id"],
        "name": stix_obj.get(
            "name",
            ""
        ),
        "description": stix_obj.get(
            "description",
            ""
        ),
        "url": stix_obj.get(
            "external_references",
            [{}]
        )[0].get(
            "url",
            ""
        ),
        "created": stix_obj.get("created"),
        "modified": stix_obj.get("modified"),
        "revoked": stix_obj.get(
            "revoked",
            False
        )
    }    

def map_tool(stix_obj):
    return {
        "stix_id": stix_obj["id"],
        "name": stix_obj.get(
            "name",
            ""
        ),
        "description": stix_obj.get(
            "description",
            ""
        ),
        "aliases": stix_obj.get(
            "aliases",
            []
        ),
        "created": stix_obj.get("created"),
        "modified": stix_obj.get("modified"),
        "revoked": stix_obj.get(
            "revoked",
            False
        )
    }    

def map_malware(stix_obj):
    return {
        "stix_id": stix_obj["id"],
        "name": stix_obj.get(
            "name",
            ""
        ),
        "description": stix_obj.get(
            "description",
            ""
        ),
        "aliases": stix_obj.get(
            "aliases",
            []
        ),
        "created": stix_obj.get("created"),
        "modified": stix_obj.get("modified"),
        "revoked": stix_obj.get(
            "revoked",
            False
        )
    }    

    