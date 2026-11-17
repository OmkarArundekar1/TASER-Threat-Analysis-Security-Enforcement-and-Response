"""Read only. On sample files already in the repository, measures whether generate_alert_id separates
alerts that Wazuh's own id says are different. Prints counts only, no alert content. Never reads /var/ossec."""
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, "listener")
from utils import generate_alert_id  # noqa: E402

for path in sys.argv[1:]:
    assert not os.path.abspath(path).startswith("/var/ossec")
    if not os.path.exists(path):
        print(path, "ABSENT")
        continue
    alerts, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    try:
        loaded = json.loads(text)
        alerts = loaded if isinstance(loaded, list) else [loaded]
    except ValueError:
        for line in text.splitlines():
            try:
                alerts.append(json.loads(line))
            except ValueError:
                bad += 1
    alerts = [a for a in alerts if isinstance(a, dict)]
    print(f"\n{path}: {len(alerts)} alerts, {bad} unparsable")
    with_id = [a for a in alerts if a.get("id")]
    print(f"  have a top-level id: {len(with_id)}; distinct ids: {len({a['id'] for a in with_id})}; have a timestamp: {sum(1 for a in alerts if a.get('timestamp'))}; have srcip in data: {sum(1 for a in alerts if isinstance(a.get('data'), dict) and a['data'].get('srcip'))}; have src_ip in data: {sum(1 for a in alerts if isinstance(a.get('data'), dict) and a['data'].get('src_ip'))}")
    groups = collections.defaultdict(list)
    for alert in alerts:
        groups[generate_alert_id(alert)].append(alert)
    print(f"  distinct alert_id: {len(groups)}")
    colliding = [g for g in groups.values() if len(g) > 1]
    different = 0
    different_content = 0
    for group in colliding:
        if len({a.get("id") for a in group}) > 1:
            different += len(group) - 1
        if len({hashlib.sha256(json.dumps(a, sort_keys=True, default=str).encode()).hexdigest() for a in group}) > 1:
            different_content += len(group) - 1
    print(f"  alert_id groups with more than one alert: {len(colliding)}; alerts lost to a collision though Wazuh's id differs: {different}; though the content differs: {different_content}")
    same_id = collections.defaultdict(set)
    for alert in with_id:
        same_id[alert["id"]].add(generate_alert_id(alert))
    print(f"  Wazuh ids mapping to more than one alert_id: {sum(1 for v in same_id.values() if len(v) > 1)}")
    if with_id:
        print(f"  id format example shape: {''.join('d' if c.isdigit() else c for c in str(with_id[0]['id']))}")
