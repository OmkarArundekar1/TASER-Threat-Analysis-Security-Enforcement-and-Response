import json

DEFAULT_EVE_PATH = "../../data/raw_logs/eve.json"

# Mapping IDS signatures to attack stages
STAGE_MAP = {
    "Nmap": "recon",
    "SCAN": "recon",
    "Port": "recon",
    "SQL": "exploit",
    "Injection": "exploit",
    "Exploit": "exploit",
    "Brute": "credential",
    "SSH": "credential",
    "Login": "credential"
}


def _detect_stage(signature: str) -> str:
    for key, stage in STAGE_MAP.items():
        if key in signature:
            return stage
    return "unknown"


def parse_eve_json(path: str = DEFAULT_EVE_PATH) -> list[dict]:
    """Parse a Suricata eve.json file into normalized alert events."""
    events = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("event_type") != "alert":
                continue

            signature = data["alert"]["signature"]

            events.append({
                "timestamp": data.get("timestamp"),
                "src_ip": data.get("src_ip"),
                "dst_ip": data.get("dest_ip"),
                "signature": signature,
                "stage": _detect_stage(signature),
                "severity": data["alert"].get("severity"),
                "sensor": "suricata"
            })

    return events


if __name__ == "__main__":
    events = parse_eve_json()
    print("\nTotal Suricata alerts:", len(events))
    print("\nSample events:\n")
    for e in events[:5]:
        print(e)
