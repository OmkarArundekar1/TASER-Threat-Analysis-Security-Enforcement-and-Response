"""Writes the 8 synthetic alerts for the mini restart/policy check (NOT archive
data). usage: make_smoke_alerts.py <out.jsonl>

index  t(s)  alert                      expected path
0      0     T1595, rule 900001         RESOLVED
1      1     same                       DUPLICATE (buffered)
2      2     UNKNOWN, rule 31101        UNATTRIBUTED
3      3     same                       DUPLICATE_UNATTRIBUTED
   ---- simulated restart before index 4 ----
4      4     T1595, rule 900001         DUPLICATE via the Neo4j reload path
5      5     UNKNOWN, rule 31101        DUPLICATE_UNATTRIBUTED via the reload path
6      100   UNKNOWN, rule 31102        UNATTRIBUTED (activity for A only, in memory)
7      200   T1110, rule 900002         RESOLVED: 100 s after the UNKNOWN alert but
                                        200 s after the last resolved alert, so the
                                        activity policies should disagree here
"""
import json
import sys
from datetime import datetime, timedelta, timezone

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def alert(seconds, rule_id, technique=None):
    rule = {"id": rule_id, "level": 10, "description": f"smoke {rule_id}", "groups": []}
    if technique:
        rule["mitre"] = {"id": [technique], "technique": [technique], "tactic": []}
    when = T0 + timedelta(seconds=seconds)
    return {"timestamp": when.strftime("%Y-%m-%dT%H:%M:%S.000+0000"), "rule": rule,
            "agent": {"id": "099", "name": "smoke-victim"},
            "data": {"src_ip": "10.99.0.5", "dest_ip": "10.99.0.9"}}


rows = [alert(0, "900001", "T1595"), alert(1, "900001", "T1595"), alert(2, "31101"), alert(3, "31101"),
        alert(4, "900001", "T1595"), alert(5, "31101"), alert(100, "31102"), alert(200, "900002", "T1110")]
with open(sys.argv[1], "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
