"""Read only. What the archived alerts show about Wazuh's id. Prints counts and field names, never alert content.
usage (from isfcr/CYUKTI/backend): python -I d1_wazuh_id_analysis.py ALERT_FILE..."""
import collections
import datetime as dt
import hashlib
import json
import os
import re
import sys


def load(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, list) else [loaded]
    except ValueError:
        alerts = []
        for line in text.splitlines():
            try:
                alerts.append(json.loads(line))
            except ValueError:
                pass
        return alerts


def digest(alert, drop):
    return hashlib.sha256(json.dumps({k: v for k, v in alert.items() if k not in drop}, sort_keys=True, default=str).encode()).hexdigest()


for path in sys.argv[1:]:
    assert not os.path.abspath(path).startswith("/var/ossec")
    alerts = [a for a in load(path) if isinstance(a, dict)]
    print(f"\n### {path}")
    print(f"sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}; alerts {len(alerts)}")
    if not alerts:
        continue
    keys = collections.Counter(k for a in alerts for k in a)
    print("top-level keys:", dict(keys.most_common()))
    print("manager.name distinct values:", len({(a.get("manager") or {}).get("name") for a in alerts if isinstance(a.get("manager"), dict)}),
          "| alerts with manager.name:", sum(1 for a in alerts if isinstance(a.get("manager"), dict) and a["manager"].get("name")))
    print("cluster.name present:", sum(1 for a in alerts if isinstance(a.get("cluster"), dict) and a["cluster"].get("name")),
          "| cluster.node present:", sum(1 for a in alerts if isinstance(a.get("cluster"), dict) and a["cluster"].get("node")))
    with_id = [a for a in alerts if a.get("id")]
    print(f"with id: {len(with_id)}; distinct: {len({a['id'] for a in with_id})}; string typed: {sum(1 for a in with_id if isinstance(a['id'], str))}")
    shaped = [a for a in with_id if isinstance(a["id"], str) and re.fullmatch(r"\d+\.\d+", a["id"])]
    print(f"match seconds.offset: {len(shaped)}")
    if not shaped:
        continue
    same_second, off_by = 0, collections.Counter()
    regressions, equal_offsets, previous = 0, 0, None
    deltas = []
    seconds_list = []
    for alert in shaped:
        seconds, offset = alert["id"].split(".")
        seconds, offset = int(seconds), int(offset)
        seconds_list.append(seconds)
        try:
            stamp = dt.datetime.strptime(alert["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z")
            difference = seconds - int(stamp.timestamp())
            off_by[max(-3, min(3, difference))] += 1
            same_second += difference == 0
        except (KeyError, ValueError, TypeError):
            pass
        if previous is not None:
            if offset < previous:
                regressions += 1
            elif offset == previous:
                equal_offsets += 1
            else:
                deltas.append(offset - previous)
        previous = offset
    print(f"id seconds equal the timestamp's second: {same_second}; difference histogram (clamped to 3): {dict(sorted(off_by.items()))}")
    print(f"file order: offset went down {regressions} times, stayed equal {equal_offsets} times, rose {len(deltas)} times")
    if deltas:
        deltas.sort()
        print(f"offset rise: min {deltas[0]}, median {deltas[len(deltas)//2]}, max {deltas[-1]}")
    per_second = collections.Counter(seconds_list)
    print(f"alerts in the busiest second: {max(per_second.values())}; seconds with more than one alert: {sum(1 for v in per_second.values() if v > 1)}")
    span = max(seconds_list) - min(seconds_list)
    print(f"span {span} s ({span/86400:.2f} days), rate {len(shaped)/max(span,1)*86400:.0f} alerts per day over the span")
    active_days = collections.Counter(s // 86400 for s in seconds_list)
    print(f"distinct days {len(active_days)}, busiest day {max(active_days.values())} alerts")
    print("--- identical occurrences, the case a content hash can't separate")
    for label, drop in (("all fields except id", {"id"}), ("all except id and timestamp", {"id", "timestamp"})):
        groups = collections.Counter(digest(a, drop) for a in alerts)
        repeated = {k: v for k, v in groups.items() if v > 1}
        print(f"{label}: {len(groups)} distinct; {len(repeated)} groups repeat; {sum(repeated.values()) - len(repeated)} alerts would merge into another under a content hash")
