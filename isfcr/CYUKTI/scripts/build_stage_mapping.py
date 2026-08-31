import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from mitre_import.parser import get_by_type  # noqa: E402

OBSERVED = [
    "T1595", "T1110", "T1053.003", "T1110.001", "T1078",  # already mapped
    "T1055", "T1057", "T1059", "T1059.007", "T1114",
    "T1190", "T1210", "T1562.001", "T1595.002",
    "T1021.004",  # already mapped (Lateral Movement) but check consistency
]

techniques = {}
for obj in get_by_type("attack-pattern"):
    attack_id = None
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            attack_id = ref.get("external_id")
            break
    if attack_id in OBSERVED:
        techniques[attack_id] = obj

for aid in OBSERVED:
    obj = techniques.get(aid)
    if not obj:
        print(f"{aid}: NOT FOUND IN STIX CORPUS")
        continue
    phases = [p.get("phase_name") for p in obj.get("kill_chain_phases", [])]
    print(f"{aid:12s} name={obj.get('name'):35s} tactics={phases}")
