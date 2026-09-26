# CYUKTI — Quantitative Results

> Last verified: 2026-09-26. Counts reflect live Neo4j query and pytest run from this date. **New, preliminary only**: attribution 74.4% (n=43), MITRE mapping F1=0.80 (n=100) — both from independent AI-built ground truth, human review pending (`review/paper_metrics_source_of_truth.md`). Not slide-ready yet as "verified" numbers — present them explicitly as preliminary if used.

17 strongest numbers for a presentation slide, each with a qualifier. Every number below is traceable to `quantitative_results.md`/`.csv`/`paper_submission_status.md` — none were invented for this list.

- **689/689 backend tests passing (65 files) + 109/109 frontend (17 files) = 798 total** — full regression suite, re-run 2026-09-25 (was 150/150 backend-only on 2026-09-12)
- **83 SOAR/playbook tests (9 files)** — new subsystem since the 2026-09-12 snapshot
- **858 ATT&CK techniques** — imported from vendored MITRE STIX, Enterprise v19.1 (unchanged — vendored corpus)
- **111 Campaigns / 212 AttackEvents / 50 Operations / 9 Attackers / 12 Hosts** — live graph, 2026-09-25 (was 65/124/43/5/4 on 2026-09-12)
- **2,539 total nodes / 20,804 total relationships** — full live graph scale, 2026-09-25
- **93.3% dedup ratio** — 212 distinct AttackEvents vs. 3,185 tracked occurrences
- **46 API routes** — 33 core + 13 SOAR
- **27 mounted frontend components, 1 orphaned** (`PredictionPanel.tsx`)
- **10 investigation actions** — not 8; `CAMPAIGN_NARRATIVE_SEARCH` and `GNN_TOPOLOGY_RETRIEVAL` were added later
- **0 orphaned AttackEvents** — 124/124 linked to a Campaign, live-verified 2026-09-12 (not re-run at the current 212-event scale)
- **44 orphaned events repaired → 0** — real production data-integrity defect found and durably fixed
- **MITRE coverage (post rule-fix + reboot): 26/120 (21.7%) native, 94/120 (78.3%) UNKNOWN** — 2026-09-25; earlier snapshots (391-alert 10.0%/90.0% from 2026-08-31, 239-alert, 13-alert) preserved in `quantitative_results.md` as historical, superseded traffic-composition snapshots
- **5/5 live UNKNOWN events re-verified intact** — 0 fabricated `attack_id`, after a full Neo4j outage/recovery cycle
- **3 NEXT_TECHNIQUE edges, unchanged before/after UNKNOWN ingestion** — zero attack-chain contamination, live-verified
- **XGBoost accuracy: 91.7%** — in-sample, n=60; **not** a generalization result
- **Macro F1: 0.676** — in-sample, n=60
- **Critical-class recall: 25%** — 3 of 4 Critical campaigns missed even in-sample, flagged not hidden
- **NEXT_TECHNIQUE: 4/12 correct (33.3%)** — self-assessed `INSUFFICIENT FOR SUPERVISED ML`
- **>500 real ATT&CK documents indexed** — RAG retriever, real corpus not synthetic
- **End-to-end latency: p50 35.5ms / p95 66.9ms** — real HTTP round trip, `GET /api/incidents/<id>/overview`, single-machine dev benchmark

## Numbers explicitly NOT included above (and why)

Not included because they don't exist as defensible measurements: attribution accuracy, RAG retrieval precision/recall, GNN performance on real data, any held-out ML metric, MISP live-publication counts, latency/throughput. See the "cannot claim" list in `quantitative_results.md`.
