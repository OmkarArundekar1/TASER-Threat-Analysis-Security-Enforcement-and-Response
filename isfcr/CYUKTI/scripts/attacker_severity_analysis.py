import os
import sys
from collections import defaultdict

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

import pandas as pd  # noqa: E402
from ml.dataset_utils import load_dataset  # noqa: E402

df = load_dataset()

print("=== Attacker x campaign count / severity / risk ===")
for attacker, group in df.groupby("attacker_ip"):
    severities = group["severity"].value_counts().to_dict()
    print(f"{attacker:28s} campaigns={len(group):3d} severities={severities} "
          f"mean_risk={group['risk_score'].mean():8.1f} max_risk={group['risk_score'].max():8.1f} "
          f"unique_techniques_total={group['unique_techniques'].sum()}")

print("\n=== Victim x campaign count / severity / risk ===")
for victim, group in df.groupby("victim_ip"):
    severities = group["severity"].value_counts().to_dict()
    print(f"{victim:28s} campaigns={len(group):3d} severities={severities} "
          f"mean_risk={group['risk_score'].mean():8.1f} max_risk={group['risk_score'].max():8.1f}")

print("\n=== Non-Low severity campaigns: which attacker? ===")
non_low = df[df["severity"] != "Low"][["campaign_id", "attacker_ip", "victim_ip", "severity", "risk_score"]]
print(non_low.to_string())

print("\n=== Attacker diversity among non-Low campaigns ===")
print(non_low["attacker_ip"].value_counts())
