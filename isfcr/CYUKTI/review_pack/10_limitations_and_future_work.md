# CYUKTI — Known Limitations, Current Issues, and Final Verdict

## Current Issue 1: maintenance-worker `TypeError` (found live, 2026-08-31, NOT fixed)

**Status: present in the current code, confirmed still reachable.**

```
File "listener/wazuh_listener.py", line 109, in maintenance_worker
    campaign_manager.expire_active_campaigns()
File "campaign_manager.py", line 352, in expire_active_campaigns
    now - context.last_seen
TypeError: unsupported operand type(s) for -: 'datetime.datetime' and 'NoneType'
```

**Root cause**: the Phase 20 UNKNOWN-first-event path calls `campaign_manager.create_campaign_context(attacker_ip, victim_ip, current_technique=None)` directly, bypassing `activate_campaign()` (deliberately — that function also calls `append_technique()`, which must never run for UNKNOWN events). `activate_campaign()` is the function that normally sets `context.last_seen`; skipping it leaves the in-memory `CampaignContext` cached with `last_seen=None`. The maintenance thread's `expire_active_campaigns()` (unrelated to any Phase 20 file) then crashes computing `now - None` every ~5 seconds for as long as that context stays cached.

**Classification: non-blocking / technical debt / operational risk (not correctness risk).**
- UNKNOWN ingestion itself is unaffected — verified live, 5 real UNKNOWN events created successfully after this exception started recurring.
- The listener process remains alive and continues processing alerts (offset kept advancing).
- Neo4j's own `Campaign.last_seen` property is correct (set via `create_unattributed_attack_event`'s Cypher) — only the separate in-memory cache object is affected.
- The operational risk: campaign timeout/expiration is likely non-functional while any such context sits in the cache, since the exception is caught per-cycle but may abort the rest of that cycle's loop — meaning other legitimately-expirable campaigns may not get expired on schedule either, for as long as the listener runs without a restart.

**Disclosure for review**: this should be stated plainly if asked — "we found this live yesterday during our own activation testing, root-caused it precisely, and made the deliberate decision not to fix it under this task's scope so as not to touch `campaign_manager.py` without separate authorization." That is a defensible, disciplined answer, not something to hide.

## Current Issue 2: duplicate local rule IDs `100500`/`100501`

Still formally **unresolved — requires root-level `wazuh-analysisd -t` validation.** New live evidence (rule 100500 firing 25 times with native `T1595` present) suggests the Nmap-detection rule definition is currently active, but this is behavioral inference, not a direct confirmation, and the status of the `100501` duplicate remains unknown either way. See `05_phase20_results.md` for detail. Do not claim this is resolved.

## Current Issue 3: `T1548.003` missing from `MITRE_TO_STAGE`

Confirmed absent, live-checked this session. Native Wazuh attribution for it (via rule 5401) is valid; CYUKTI's own risk-stage taxonomy simply hasn't been extended to cover it yet. Not fixed, per instruction. See `05_phase20_results.md`.

## Dataset limitations (Phase 17 — carried forward, unchanged)

- 60 campaigns from only 3 attacker identities, 2 victims, 4 attacker/victim pairs.
- 23 of 57 declared features are zero-variance across the entire dataset.
- 11 exact feature-vector duplicate rows.
- Strong attacker/severity confounding (one attacker identity is 100% Low severity).
- Significant temporal clustering (45% of campaigns on 2 calendar dates).
- Zero High-severity examples.
- **Formal verdict, unchanged since Phase 17**: `NOT_READY_FOR_CALIBRATION`.

## ML/prediction limitations

- No held-out train/test split exists anywhere in the repository — the in-sample XGBoost metrics in `04_results_and_metrics.md` are not a generalization estimate.
- NEXT_TECHNIQUE learning has only 3 real edges and 12 evaluable predictions (4 correct) — Phase 18's own formal verdict: `INSUFFICIENT_FOR_SUPERVISED_ML`.
- GNN has no real-campaign benchmark, only synthetic-graph unit tests.
- No attribution accuracy metric exists (no ground-truth actor-identity dataset).

## Infrastructure/operational limitations

- MISP live-publish success not reconfirmed this session (`MISP_API_KEY` was empty as of a prior audit).
- Neo4j-unavailable fallback behavior not tested.
- No latency/throughput instrumentation exists anywhere in the pipeline.
- File-based audit log (`logs/prerana_listener.log`) observed at 0 bytes despite active logging to stdout — the file-handler is not reliably capturing a durable audit trail in the current environment.

## What must NOT be claimed in the review

- Do NOT claim the XGBoost model "achieves 91.7% accuracy" without immediately qualifying it as in-sample, n=60, no held-out split.
- Do NOT describe the 90% UNKNOWN rate as a coverage failure or a weakness to apologize for — it is the intended, correct outcome of refusing to fabricate attribution.
- Do NOT claim next-technique prediction works — the system's own audit says it doesn't have enough data yet.
- Do NOT claim attribution accuracy — no such metric exists.
- Do NOT claim the `100500`/`100501` duplicate-rule issue or the maintenance-worker bug are fixed.
- Do NOT claim MISP publication is confirmed working live.
- Do NOT claim novelty for the individual algorithms (Neo4j, TF-IDF, XGBoost, GraphSAGE) — only for the system-level attribution-resolution design.

## Final Technical Verdict

### Maturity classification: **Research Prototype**

**Justification**: CYUKTI is well past a bare prototype — it has a real, live, multi-service integration (Wazuh + Neo4j + MISP + a trained ML pipeline), 150 passing tests, a genuinely-fixed production defect discovered and repaired mid-project (Phase 18's next-technique pipeline, Phase 20's live blocker), and disciplined self-auditing (the Phase 17 validity gate explicitly refusing to proceed to calibration). It falls short of "Functional Prototype" or higher because: several core modules (attribution, GNN, prediction) have no benchmark beyond unit tests; the dataset itself is explicitly, formally judged not ready for the experiments it was built to support; and at least one live, disclosed defect (maintenance-worker crash) remains unresolved. It is not "Production-like" because there is no held-out evaluation methodology, no load/latency testing, and no confirmed MISP publish path.

### What is genuinely completed
Wazuh→Neo4j ingestion pipeline; Phase 20 MITRE resolution (resolver, registry, unattributed-event path, live-validated); campaign reconstruction (real 44-event repair); the corrected next-technique label pipeline (Phase 18); the evidence-aware confidence architecture (implemented and unit-tested); a real, trained ML stack (XGBoost, SSL autoencoder, GNN, SSFT) with real artifacts on disk; RAG retrieval over a real ATT&CK corpus; a working dashboard API and React frontend.

### What is experimentally validated
Wazuh listener resilience across restarts (Phases 20D-F); MITRE resolver correctness (unit + integration + live, on real alert traffic); campaign-reconstruction idempotency; the Phase 18 dataset-labeling fix (verified via real dataset rebuild); UNKNOWN-path safety invariants (verified against real, live-created Neo4j records, not just tests).

### What is implemented but not benchmarked
Threat attribution (no ground-truth accuracy dataset); campaign correlation/Operation matching (existence confirmed, quality not evaluated); GNN on real campaigns (synthetic-only tests); RAG retrieval quality (no labeled query set); MISP live publication (client initializes, publish not reconfirmed); XGBoost generalization (in-sample only).

### What remains
Deliberate dataset expansion (Phase 19 spec: 6-8 attackers, 5+ victims, ≥1 real High example, ≥20 technique compositions); a held-out ML evaluation methodology; fixing the two disclosed live issues; a real attribution-accuracy benchmark; latency/throughput instrumentation.

### Strongest results to show
1. Live UNKNOWN-path safety proof: 5 real Neo4j records, zero fabricated attribution, zero attack-chain contamination.
2. 150/150 passing test suite.
3. Real campaign-reconstruction repair (44 orphaned events → 27 campaigns, idempotent, still 0 orphans today).
4. The Phase 17→18 self-correction narrative: found a real ground-truth bug (next_technique was the model's own prediction, not reality), fixed it, and it's now demonstrably correct.

### Strongest architecture diagram
`cyukti_detailed_architecture.mmd` (Diagram 2) — it's the one that actually demonstrates the core contribution (the resolver fork and the structural non-contamination of the UNKNOWN branch), not just a generic pipeline.

### Strongest research contribution
Provenance-aware MITRE resolution with a structurally-enforced, live-verified UNKNOWN state (contributions #8, #9, #10 in `06_research_contribution.md`) — this is the one claim backed by both a control-flow proof and live data, and it directly addresses a real, named problem (the "detection paradox") rather than a generic capability.
