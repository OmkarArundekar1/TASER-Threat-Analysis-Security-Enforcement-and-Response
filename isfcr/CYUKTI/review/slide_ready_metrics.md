# CYUKTI — Quantitative Results

15 strongest numbers for a presentation slide, each with a qualifier. Every number below is traceable to `quantitative_results.md`/`.csv` — none were invented for this list.

- **150/150 tests passing** — full backend regression suite, fresh run 2026-09-12
- **858 ATT&CK techniques** — imported from vendored MITRE STIX, Enterprise v19.1
- **0 orphaned AttackEvents** — 124/124 linked to a Campaign, live-verified 2026-09-12
- **44 orphaned events repaired → 0** — real production data-integrity defect found and durably fixed
- **391 live alerts evaluated** — largest available MITRE-resolution coverage snapshot (2026-08-31)
- **39/391 (10.0%) native Wazuh ATT&CK mappings**
- **352/391 (90.0%) preserved as UNKNOWN** — by design, not a deficiency
- **5/5 live UNKNOWN events re-verified intact** — 0 fabricated `attack_id`, after a full Neo4j outage/recovery cycle
- **3 NEXT_TECHNIQUE edges, unchanged before/after UNKNOWN ingestion** — zero attack-chain contamination, live-verified
- **XGBoost accuracy: 91.7%** — in-sample, n=60; **not** a generalization result
- **Macro F1: 0.676** — in-sample, n=60
- **Critical-class recall: 25%** — 3 of 4 Critical campaigns missed even in-sample, flagged not hidden
- **NEXT_TECHNIQUE: 4/12 correct (33.3%)** — self-assessed `INSUFFICIENT FOR SUPERVISED ML`
- **>500 real ATT&CK documents indexed** — RAG retriever, real corpus not synthetic
- **65 Campaigns / 124 AttackEvents / 858 Technique nodes** — full live graph scale, identical before and after an unplanned infrastructure outage

## Numbers explicitly NOT included above (and why)

Not included because they don't exist as defensible measurements: attribution accuracy, RAG retrieval precision/recall, GNN performance on real data, any held-out ML metric, MISP live-publication counts, latency/throughput. See the "cannot claim" list in `quantitative_results.md`.
