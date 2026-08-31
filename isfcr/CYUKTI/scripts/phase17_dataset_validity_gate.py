"""
phase17_dataset_validity_gate.py
=================================
Read-only statistical validation of the production campaign_dataset.csv
before any severity-calibration work. Does not modify Neo4j, the CSV,
TPS_MAP, TPS_CEILING, or severity thresholds. Does not train models.

Writes an immutable snapshot (metadata + CSV copy) under
backend/analysis_snapshots/<timestamp>/ and prints every statistic the
Phase 17 report needs, computed directly from real data.
"""
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from ml.dataset_utils import load_dataset, FEATURE_COLUMNS, LABEL_COLUMNS, IDENTIFIER_COLUMNS, LEAKAGE_COLUMNS  # noqa: E402
from risk_scoring import TPS_CEILING, normalize_risk_score, risk_level_from_score  # noqa: E402
from mitre_mapper import MITRE_TO_STAGE  # noqa: E402
from config import TPS_MAP  # noqa: E402
from neo4j_client import driver  # noqa: E402

TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SNAP_DIR = os.path.join(BACKEND, "analysis_snapshots", TS)
os.makedirs(SNAP_DIR, exist_ok=True)

df = load_dataset()

def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

# ---------------------------------------------------------------- PHASE 1
print("=" * 100)
print("PHASE 1: IMMUTABLE SNAPSHOT")
print("=" * 100)

with driver.session() as session:
    neo4j_campaign_count = session.run("MATCH (c:Campaign) RETURN count(c) AS n").single()["n"]
    neo4j_event_count = session.run("MATCH (e:AttackEvent) RETURN count(e) AS n").single()["n"]
    neo4j_linked_event_count = session.run(
        "MATCH (:Campaign)-[:HAS_EVENT]->(e:AttackEvent) RETURN count(DISTINCT e) AS n"
    ).single()["n"]

snapshot_meta = {
    "generated_at_utc": TS,
    "dataset_csv_path": "backend/ml/datasets/campaign_dataset.csv",
    "dataset_row_count": len(df),
    "neo4j_campaign_count": neo4j_campaign_count,
    "neo4j_attackevent_count": neo4j_event_count,
    "neo4j_linked_attackevent_count": neo4j_linked_event_count,
    "neo4j_orphaned_attackevent_count": neo4j_event_count - neo4j_linked_event_count,
    "feature_columns": FEATURE_COLUMNS,
    "feature_column_count": len(FEATURE_COLUMNS),
    "label_columns": LABEL_COLUMNS,
    "identifier_columns": IDENTIFIER_COLUMNS,
    "leakage_columns_excluded": LEAKAGE_COLUMNS,
    "risk_score_formula": "normalize_risk_score(raw)=min(100, round(raw/TPS_CEILING*100)); risk_level_from_score: >=80 CRITICAL, >=60 HIGH, >=35 MEDIUM, else LOW",
    "tps_ceiling": TPS_CEILING,
    "severity_thresholds": {"CRITICAL": 80, "HIGH": 60, "MEDIUM": 35, "LOW": 0},
    "mitre_to_stage_hash": h(MITRE_TO_STAGE),
    "mitre_to_stage_size": len(MITRE_TO_STAGE),
    "tps_map_hash": h(TPS_MAP),
    "tps_map": TPS_MAP,
}
with open(os.path.join(SNAP_DIR, "snapshot_metadata.json"), "w") as f:
    json.dump(snapshot_meta, f, indent=2, default=str)
shutil.copy(os.path.join(BACKEND, "ml", "datasets", "campaign_dataset.csv"), os.path.join(SNAP_DIR, "campaign_dataset.csv"))
print(f"Snapshot written to: {SNAP_DIR}")
for k, v in snapshot_meta.items():
    if k not in ("tps_map",):
        print(f"  {k}: {v}")

# ---------------------------------------------------------------- PHASE 2
print("\n" + "=" * 100)
print("PHASE 2: CAMPAIGN INDEPENDENCE")
print("=" * 100)
print(f"Total rows: {len(df)}")
print(f"Distinct campaign_id: {df['campaign_id'].nunique()}")
print(f"Distinct attacker_ip: {df['attacker_ip'].nunique()}  -> counts:\n{df['attacker_ip'].value_counts().to_string()}")
print(f"Distinct victim_ip: {df['victim_ip'].nunique()}  -> counts:\n{df['victim_ip'].value_counts().to_string()}")
pair_counts = df.groupby(["attacker_ip", "victim_ip"]).size()
print(f"Distinct (attacker,victim) pairs: {len(pair_counts)}")
print(pair_counts.to_string())

# feature-vector duplicate detection (Phase 2/3 combined)
feat_df = df[FEATURE_COLUMNS].copy()
feat_hashes = feat_df.apply(lambda row: h(row.to_dict()), axis=1)
df["_feat_hash"] = feat_hashes
dup_groups = df.groupby("_feat_hash")["campaign_id"].apply(list)
dup_groups = dup_groups[dup_groups.apply(len) > 1]
print(f"\nDistinct feature vectors (over {len(FEATURE_COLUMNS)} feature cols): {df['_feat_hash'].nunique()} / {len(df)} rows")
print(f"Feature-duplicate groups (>=2 campaigns sharing identical feature vector): {len(dup_groups)}")
for gh, cids in dup_groups.items():
    sub = df[df["campaign_id"].isin(cids)][["campaign_id", "attacker_ip", "victim_ip", "severity", "risk_score"]]
    print(f"  group ({len(cids)} campaigns): {cids}")
    print(sub.to_string(index=False))

# ---------------------------------------------------------------- PHASE 4
print("\n" + "=" * 100)
print("PHASE 4: FEATURE COLLISION / VARIANCE AUDIT")
print("=" * 100)
variance_report = []
for col in FEATURE_COLUMNS:
    s = df[col]
    nunique = s.nunique(dropna=True)
    is_numeric = pd.api.types.is_numeric_dtype(s)
    zero_rate = float((s == 0).mean()) if is_numeric else None
    constant = nunique <= 1
    variance_report.append({
        "feature": col, "nunique": int(nunique), "zero_rate": zero_rate,
        "constant": constant, "dtype": str(s.dtype),
    })
vr_df = pd.DataFrame(variance_report).sort_values("nunique")
print(vr_df.to_string(index=False))
constant_feats = vr_df[vr_df["constant"]]["feature"].tolist()
print(f"\nCONSTANT (zero-variance) features across all {len(df)} rows: {constant_feats}")
near_const = vr_df[(vr_df["nunique"] > 1) & (vr_df["nunique"] <= 3)]["feature"].tolist()
print(f"Near-constant features (nunique 2-3): {near_const}")

# ---------------------------------------------------------------- PHASE 5
print("\n" + "=" * 100)
print("PHASE 5: ATTACKER / SEVERITY CONFOUNDING")
print("=" * 100)
ct = pd.crosstab(df["attacker_ip"], df["severity"])
print(ct.to_string())
try:
    chi2, p, dof, expected = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    min_dim = min(ct.shape) - 1
    cramers_v = float(np.sqrt((chi2 / n) / max(min_dim, 1))) if min_dim > 0 else float("nan")
    print(f"chi2={chi2:.3f} dof={dof} p={p:.4f} Cramer's V={cramers_v:.3f} (n={n})")
    print("NOTE: chi-square/Cramer's V assumptions (expected-cell-count >=5) are likely violated "
          "given small n and sparse cells -- reported for reference only, not as a definitive test.")
    print("Expected counts:\n", pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(2).to_string())
except Exception as e:
    print(f"chi2_contingency failed: {e}")

print("\nRisk-score distribution by attacker:")
print(df.groupby("attacker_ip")["risk_score"].describe().to_string())

# ---------------------------------------------------------------- PHASE 6/7 (real Neo4j timestamps + technique sets)
print("\n" + "=" * 100)
print("PHASE 6/7: TEMPORAL CLUSTERING + TECHNIQUE DIVERSITY (from Neo4j)")
print("=" * 100)
with driver.session() as session:
    rows = session.run(
        """
        MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)
        RETURN c.campaign_id AS campaign_id, c.attacker_ip AS attacker_ip,
               c.first_seen AS first_seen, c.last_seen AS last_seen,
               collect(DISTINCT e.attack_id) AS techniques
        """
    ).data()
camp_rows = pd.DataFrame(rows)

def to_dt(v):
    if v is None:
        return None
    try:
        return v.to_native() if hasattr(v, "to_native") else pd.to_datetime(str(v), utc=True, errors="coerce")
    except Exception:
        return None

camp_rows["first_seen_dt"] = camp_rows["first_seen"].apply(to_dt)
camp_rows = camp_rows.dropna(subset=["first_seen_dt"])
camp_rows["first_seen_dt"] = pd.to_datetime(camp_rows["first_seen_dt"], utc=True, errors="coerce")
camp_rows = camp_rows.dropna(subset=["first_seen_dt"])
camp_rows["date"] = camp_rows["first_seen_dt"].dt.date
print(f"Campaigns with resolvable first_seen: {len(camp_rows)} / {len(rows)}")
print("Campaigns per calendar date:")
print(camp_rows.groupby("date").size().to_string())
print("\nCampaigns per date x attacker:")
print(camp_rows.groupby(["date", "attacker_ip"]).size().to_string())

camp_rows_sorted = camp_rows.sort_values("first_seen_dt")
gaps = camp_rows_sorted["first_seen_dt"].diff().dt.total_seconds().dropna()
if len(gaps):
    print(f"\nInter-campaign start-time gaps (seconds): min={gaps.min():.1f} median={gaps.median():.1f} max={gaps.max():.1f}")
    print(f"Campaigns starting within 60s of the previous campaign: {int((gaps <= 60).sum())} / {len(gaps)}")

print("\nTechnique-set diversity:")
camp_rows["technique_set"] = camp_rows["techniques"].apply(lambda ts: frozenset(ts))
print(f"Distinct technique sets: {camp_rows['technique_set'].nunique()} / {len(camp_rows)}")
tset_groups = camp_rows.groupby("technique_set")["campaign_id"].apply(list)
tset_dups = tset_groups[tset_groups.apply(len) > 1]
print(f"Campaigns sharing an IDENTICAL technique set with >=1 other campaign: {sum(len(v) for v in tset_dups)} across {len(tset_dups)} groups")
for tset, cids in tset_dups.items():
    print(f"  {sorted(tset)} -> {cids}")

all_techniques = Counter()
for ts in camp_rows["techniques"]:
    all_techniques.update(ts)
print(f"\nDistinct techniques observed across all campaigns: {len(all_techniques)}")
print(f"Technique frequency: {dict(all_techniques.most_common())}")
unmapped = [t for t in all_techniques if t not in MITRE_TO_STAGE]
print(f"Observed techniques NOT in MITRE_TO_STAGE (excluded from risk_score by design): {unmapped}")

# per-attacker technique diversity
print("\nPer-attacker technique diversity:")
for attacker, group in camp_rows.groupby("attacker_ip"):
    techs = set()
    for ts in group["techniques"]:
        techs.update(ts)
    print(f"  {attacker}: campaigns={len(group)} distinct_techniques={len(techs)} -> {sorted(techs)}")

# ---------------------------------------------------------------- PHASE 8
print("\n" + "=" * 100)
print("PHASE 8: SEVERITY BOUNDARY STABILITY")
print("=" * 100)
df["_norm_score"] = df["risk_score"].apply(normalize_risk_score)
boundaries = [35, 60, 80]
for b in boundaries:
    dist = (df["_norm_score"] - b).abs()
    close = df[dist <= 5][["campaign_id", "risk_score", "_norm_score", "severity"]]
    print(f"\nCampaigns within 5 normalized points of boundary {b}: {len(close)}")
    if len(close):
        print(close.to_string(index=False))

# ---------------------------------------------------------------- PHASE 9
print("\n" + "=" * 100)
print("PHASE 9: RISK-SCORE DISTRIBUTION")
print("=" * 100)
print(df["risk_score"].describe().to_string())
print("\nPercentiles:")
print(df["risk_score"].quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
print(f"\nMax value: {df['risk_score'].max()} (campaign: {df.loc[df['risk_score'].idxmax(), 'campaign_id']})")
print(f"TPS_CEILING={TPS_CEILING}; values exceeding ceiling (clip to 100): {int((df['risk_score'] > TPS_CEILING).sum())} / {len(df)}")
print("\nSeverity label counts:")
print(df["severity"].value_counts().to_string())

# ---------------------------------------------------------------- PHASE 11 (NEXT_TECHNIQUE)
print("\n" + "=" * 100)
print("PHASE 11: NEXT_TECHNIQUE / prediction_correct VALIDITY (re-check post-reconstruction)")
print("=" * 100)
with driver.session() as session:
    nt_rows = session.run(
        """
        MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique)
        RETURN a.attack_id AS from_technique, b.attack_id AS to_technique, r.count AS count
        ORDER BY r.count DESC
        """
    ).data()
print(f"Distinct NEXT_TECHNIQUE edges: {len(nt_rows)}")
for r in nt_rows:
    print(f"  {r['from_technique']} -> {r['to_technique']}  count={r['count']}")
print(f"\nprediction_correct label distribution:\n{df['prediction_correct'].value_counts().to_string()}")
print(f"next_technique value distribution:\n{df['next_technique'].value_counts().to_string()}")

df.drop(columns=["_feat_hash", "_norm_score"], errors="ignore").to_csv(os.path.join(SNAP_DIR, "annotated_view_dropped_helper_cols.csv"), index=False)
print(f"\nDone. Snapshot + all raw outputs above are reproducible from: {SNAP_DIR}")
