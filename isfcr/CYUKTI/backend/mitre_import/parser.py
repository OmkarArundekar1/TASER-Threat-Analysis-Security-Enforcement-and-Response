import json
from mitre_import.config import STIX_FILE

def load_stix():
    with open(STIX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_objects():
    data = load_stix()
    return data["objects"]

def get_by_type(object_type):
    return [
        obj
        for obj in get_objects()
        if obj.get("type") == object_type

    ]