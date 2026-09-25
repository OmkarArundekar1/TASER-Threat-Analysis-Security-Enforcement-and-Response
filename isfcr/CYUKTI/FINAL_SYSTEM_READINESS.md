# CYUKTI Final System Readiness Audit

Snapshot as of 2026-09-25, with Neo4j and the CYUKTI backend/listener genuinely running live in this environment (not assumed — see "Live environment state" below).

## Live environment state (verified this phase)

- **Wazuh manager**: real, installed, running (`systemctl status wazuh-manager` → active since 04:58 UTC), producing real organic alerts (`rule 86601`, package-manager traffic, correctly left `UNKNOWN` by the resolver).
- **Neo4j**: was down at the start of this phase (`neo4j-soc` Docker container `Exited (255)`). Started via `docker start neo4j-soc` (the user's own account is in the `docker` group — no privileged access needed). Verified reachable, **100 real campaigns** present.
- **CYUKTI backend** (`dashboard_api.py`) and **listener** (`wazuh_listener.py`): started fresh this phase against the now-live Neo4j; both confirmed processing real data (listener advanced its offset on a real incoming alert; dashboard API served real `/api/incidents/<id>/overview` responses).
- **Wazuh manager restart** (`sudo /var/ossec/bin/wazuh-control restart`, needed to activate `MITRE_MAPPING.md`'s rule fixes): confirmed genuinely blocked — `sudo -n` requires an interactive password (`sudo -ln` also refused non-interactively), and no NOPASSWD sudoers entry exists. **Not attempted further; not claimed as done.**
- **Both the backend (port 5002) and frontend dev server (port 5173) were left running** at the end of this phase so the new Incident View can be viewed directly.

## Subsystem table

| Subsystem | Status | Evidence | Tests | Live verification | Known limitation |
|---|---|---|---|---|---|
| Wazuh ingestion | IMPLEMENTED + TESTED | `listener/wazuh_listener.py` | `test_realtime_*` | **Yes** — real listener processed a real alert this phase | Manager restart pending for the new rule mappings (see below) |
| MITRE mapping | IMPLEMENTED + TESTED | `mitre_resolver.py`, `mitre_rule_registry.py`, `backend/wazuh_rules/local_rules.xml` | `test_mitre_resolver.py`, `test_mitre_enrichment.py` | **Yes** (via `wazuh-logtest`, prior phase) | (1) Wazuh manager restart still pending to activate the fixed rules in the live pipeline (blocked, see above). (2) **New finding this phase**: ~31% (268/858) of imported ATT&CK STIX techniques have non-standard `kill_chain_phases` values (`"stealth"`, `"defense-impairment"` instead of the real tactic `"defense-evasion"`) — traced to the *vendored* `enterprise-attack.json` file itself (not an import bug in `mitre_import/mapper.py` — the raw STIX object for T1562.001 and T1055 genuinely contains `phase_name: "stealth"` under `kill_chain_name: "mitre-attack"`). Not silently "corrected" by guessing a replacement — flagged for the user to re-fetch the official bundle from MITRE's GitHub and compare. |
| IOC extraction | IMPLEMENTED + TESTED | `realtime_socgraph.extract_iocs` | existing suite | Yes (live pipeline today) | none new |
| Deduplication | IMPLEMENTED + TESTED | `dedup_engine.py` | existing suite | Yes (live pipeline today) | none new |
| Campaign Manager | IMPLEMENTED + TESTED | `campaign_manager.py` | existing suite | **Yes** — 100 real campaigns confirmed in live Neo4j, new one created live this phase | none new |
| Operation correlation | IMPLEMENTED + TESTED | `campaign_correlation_engine.py`, `operation_manager.py` | existing suite | **Yes** — real `operation_id` returned by `/api/incidents/<id>/overview` for a real campaign | none new |
| Neo4j | IMPLEMENTED + TESTED | `neo4j_client.py` | full suite depends on it | **Yes** — was down, started this phase, full 643-test suite passed cleanly against it | Was not running at session start; now running, not persisted across a host reboot without the user re-running `docker start neo4j-soc` |
| MITRE features | IMPLEMENTED + TESTED | `mitre_feature_engine.py` | existing suite | Yes (live pipeline) | none new |
| GNN | IMPLEMENTED + TESTED | `ml/gnn/*` | extensive prior-phase suite | **Yes** — real embeddings computed live in `campaign_selection`'s topology signal this phase | Disabled by default; additive only; severity-ablation null result preserved unmodified |
| Severity (XGBoost) | IMPLEMENTED + TESTED | `ml/train_xgboost.py`, `ml/runtime_predictor.py` | existing suite | Not re-verified live this phase (unchanged) | Frozen GNN-ablation null result preserved; GNN NOT added to XGBoost this phase, per explicit constraint |
| Attribution | IMPLEMENTED + TESTED | `threat_attribution_engine.py` | existing suite | Not re-verified live this phase | none new |
| Investigation | IMPLEMENTED + TESTED | `investigation/loop.py` | existing suite | Not re-run live this phase (deliberately — see `INCIDENT_VIEW.md` on side effects) | Reused as-is in the new Incident View, not modified |
| Next-Best-Evidence | IMPLEMENTED + TESTED | `investigation/next_best_evidence.py` | existing suite | Same as Investigation | none new |
| Multi-RAG | IMPLEMENTED + TESTED | `rag/*`, `/api/rag/search` | existing suite | Not re-verified live this phase | Now surfaced in the Incident View's "Multi-RAG" section |
| Threat qualification | IMPLEMENTED + TESTED; **ADVISORY ONLY** re: the live MISP gate | `threat_qualification.py`, `cti_confidence_engine.py` | 14 new tests this phase + prior phase's | **Yes** — real `cti_score`/classification returned for a real campaign (SUSPICIOUS, 29.15) | Deliberately not wired into `misp_sync.should_publish()`'s live decision (Task 5's own conditions for doing so — full understanding, regression-proof tests, live verification — are not all met yet; see `MISP_PUBLICATION_POLICY.md`) |
| Campaign selection | IMPLEMENTED + TESTED | `campaign_selection.py` | 18 tests (prior phase) + integration tests this phase | **Yes** — ranked 10 real historical candidates, selected one, real explanation, live this phase | Literal node-graph visualization now built (`IncidentView.tsx`), not just a ranked list |
| ResponsePlan | IMPLEMENTED + TESTED | `soar/response_plan.py`, new `/api/incidents/<id>/response-plan` | 7 + 3 new tests | **Yes** — real plan generated for a real campaign, correct `misp_status: BLOCKED` | none new |
| Shuffle | IMPLEMENTED + ENVIRONMENT BLOCKED | `soar/shuffle_client.py` | tested with mocked HTTP | **No** — `SHUFFLE_WEBHOOK` still unconfigured in this environment | Unchanged from the prior SOAR phase; no live execution has ever occurred |
| Playbook memory | IMPLEMENTED + TESTED | `soar/memory.py` | extensive prior-phase suite | Partial — real read/write against the SQLite store confirmed this phase (via `/api/incidents/<id>/overview`'s `soar` section), but holds zero real executions since Shuffle is blocked | Correctly reports empty/no-history state honestly rather than fabricating |
| MISP | IMPLEMENTED + TESTED (confidence gate); IMPLEMENTED + ENVIRONMENT BLOCKED (authenticated publish) | `misp_sync.py`, `cti_confidence_engine.py` | ~20 tests (prior phases) | Gate logic yes; authenticated publish round-trip **not** re-verified this phase (Neo4j/listener/MISP weren't all live together with the corrected key) | See `MISP_PUBLICATION_POLICY.md`'s exact next action |
| Dashboard/API | IMPLEMENTED + TESTED | `dashboard_api.py` (~30 routes) | full route-test suite | **Yes** — extensively curl-tested against the live server this phase, including the two brand-new aggregate routes | none new |
| Audit logging | IMPLEMENTED + TESTED | `listener/logs/prerana_listener.log`, `/api/audit/logs` | `test_audit_logging.py` | **Yes** — confirmed live-writing today by the real listener process | none new |

## Regression testing (Task 8)

Commands run, exact results:

```
$ python -m pytest -q          # backend, from isfcr/CYUKTI/backend, against LIVE Neo4j
643 passed in 148.42s (0:02:28)

$ npm run test -- --run        # frontend
Test Files  19 passed (19)
     Tests  106 passed (106)

$ npm run build                # frontend production build
✓ built in 18.60s, no errors
```

**One real flake investigated and resolved, not swept under the rug**: an earlier full-suite run (with `dashboard_api.py`/`wazuh_listener.py` also running live in the background, competing for CPU/memory alongside the loaded GNN/XGBoost models) showed 2 failures in `test_audit_logging.py`'s subprocess-based tests (a fixed 30s subprocess timeout). Re-running those two tests in isolation passed in 1.46s; re-running the full `test_audit_logging.py` file alone passed in 18.46s; stopping the two competing background processes and re-running the *entire* 643-test suite passed cleanly in 148.42s. Confirmed as system-load-induced test flakiness (a fixed subprocess timeout under heavy concurrent load), not a code regression from this phase's changes — neither `wazuh_listener.py` nor `test_audit_logging.py` was touched this phase.

Checked and confirmed:
- No existing API route's response shape changed for existing callers (only new routes added; two existing routes refactored to share helper functions with identical external behavior, covered by their original tests still passing).
- GNN remains additive and disabled-by-default; `GNN_ENABLED=true` in this environment's `.env`, unchanged from prior phases.
- Severity (XGBoost) behavior unchanged — not touched this phase.
- Campaign correlation (`campaign_correlation_engine.py`, `CAMPAIGN_WEIGHTS`) unchanged — not touched this phase.
- `misp_sync.should_publish()`'s live gate condition unchanged (still exactly `CTIConfidence.publish`, i.e. `score >= 40`) — verified by `test_cti_confidence_classification.py`'s explicit regression assertion.
- No secrets in any response body, log line, or Playbook Memory record this phase (grep-checked across every new file).
- No API keys exposed — `MISP_API_KEY`/`SHUFFLE_API_KEY` never referenced outside their own client construction.
- No fabricated production data — every number shown in this document and in the Incident View traces to a real, live query result quoted above.

## Security findings

- None new this phase beyond the MITRE-corpus data-quality finding above (which is a data-integrity concern, not a security vulnerability).
- Reconfirmed: `MISP_API_KEY` never leaves `cti_publisher.py`'s own HTTP client construction.

## Data-quality findings

- **The 268/858-technique `kill_chain_phases` anomaly** described above — the single most significant finding of this phase. Affects the "tactic" label shown anywhere `mitre_resolver.enrich_technique_metadata()` is used (the Incident View header, `/api/incidents/<id>/overview`'s `mitre` array). Does **not** affect `technique_name` (verified correct in all spot-checks) or `mitre_id` (unaffected — technique IDs are correct, only the STIX object's own declared kill-chain-phase name is non-standard for a large minority of techniques).
- The rule `100510` ("Multiple Failed Login Attempts... SSH brute force") has no `frequency`/`timeframe` repetition threshold despite its description — flagged in `MITRE_MAPPING.md`, not fixed this phase (out of scope: changing detection thresholds is a different risk category than fixing a MITRE mapping).

## Accuracy metrics actually measured

None with independent ground truth this phase for MITRE mapping/threat classification/campaign correlation — see `ACCURACY_EVALUATION.md`'s per-task (A)/(B)/(C) table. `evaluation_metrics.py` itself is correctness-tested (21 tests) against hand-computed values, which is not the same as measuring CYUKTI against real-world ground truth.

## Metrics intentionally unavailable

MITRE mapping accuracy, threat qualification precision/recall, campaign correlation pairwise P/R/F1, playbook Recall@K/Precision@K — all row-(C) in `ACCURACY_EVALUATION.md`, all genuinely blocked by absent independent ground truth, none fabricated.

## MISP status

Confidence-gated publication already live and unchanged (`should_publish()`). Threat-qualification checklist built and tested, advisory only (see Task 5 reasoning in `MISP_PUBLICATION_POLICY.md`). Authenticated round-trip not re-verified this phase.

## Shuffle status

Unchanged: `SHUFFLE_INFRASTRUCTURE_BLOCKED`, no webhook configured, zero live executions ever performed.

## Playbook memory status

Real SQLite store, read/write confirmed live this phase via the new aggregate endpoint. Zero real execution records exist (Shuffle blocked) — honestly reported as such everywhere it's surfaced ("No previous playbook found," "Execution unavailable").

## GUI status

New unified Incident View built and wired into the top nav (`IncidentView.tsx`), composing 9 sections (A–I) from real backend data, all frontend tests passing (106/106), production build clean. Not screenshotted in an actual browser this phase (no `chromium-cli`/Playwright available; judged not worth the install/setup time given endpoint-level live verification and full test coverage already obtained) — both the backend (port 5002) and frontend dev server (port 5173) are left running for direct inspection.

## Remaining architectural gaps

- Shuffle/authenticated-MISP live execution — infrastructure/credential blocked, not something local implementation can close further.
- Wazuh manager restart — privileged-access blocked.
- The STIX corpus data-quality issue — needs the user's decision (re-fetch vs. investigate further) rather than an autonomous fix.
- Threat-qualification-checklist-as-live-MISP-gate — deliberately deferred pending more live observation (Task 5's own stated conditions).
- No accuracy dashboard, by design (would show numbers with no real ground truth behind them).

## Recommended next operational test

Once the Wazuh manager can be restarted (with the user's own sudo password): fire one real SSH-failure or wget/curl event through the live pipeline, confirm the alert now carries the new native `<mitre>` tag from `local_rules.xml` (T1110/T1105 respectively), confirm CYUKTI's `NATIVE_WAZUH` resolver tier picks it up, and confirm it flows through to a real `Campaign`/`/api/incidents/<id>/overview` response — closing the loop this phase could only verify up to the manager-restart boundary.

## Git commit hash

See the final report message for this phase's exact commit hash (recorded at commit time, after this document itself is committed).
