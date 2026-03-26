import json

file = "../data/raw_logs/eve.json"

events = []

# Mapping IDS signatures to attack stages
stage_map = {
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

with open(file, "r") as f:
    for line in f:

        line = line.strip()

        # skip empty lines
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # process only alert events
        if data.get("event_type") == "alert":

            signature = data["alert"]["signature"]

            stage = "unknown"

            # detect attack stage
            for key in stage_map:
                if key in signature:
                    stage = stage_map[key]
                    break

            event = {
                "timestamp": data.get("timestamp"),
                "src_ip": data.get("src_ip"),
                "dst_ip": data.get("dest_ip"),
                "signature": signature,
                "stage": stage,
                "severity": data["alert"].get("severity"),
                "sensor": "suricata"
            }

            events.append(event)

print("\nTotal Suricata alerts:", len(events))

print("\nSample events:\n")

for e in events[:5]:
    print(e)
