# CYUKTI — Phase 20 MITRE Resolution: Verified Results

> Last verified: 2026-09-26. Counts reflect live Neo4j query and pytest run from this date. **New**: an independent MITRE-mapping accuracy evaluation now exists (`evaluation/`), n=100 real alerts, P=1.00/R=0.667/F1=0.80 — PRELIMINARY, human review pending. See `review/paper_metrics_source_of_truth.md`. The Phase 20 progression/discrepancy narrative below (239→391 alerts) is preserved as a historical record; a newer post-rule-fix, post-reboot coverage snapshot has been added as its own section rather than overwriting the Phase 20 numbers, consistent with this document's own practice of disclosing snapshot-to-snapshot discrepancies rather than silently replacing them.

## Newest snapshot (2026-09-25, post rule-fix + reboot) — see also `journal_ready_data.md` Section 24

`local_rules.xml` was fixed (six duplicate-rule-ID / missing-mapping issues, `journal_ready_data.md` Section 4.9) at 05:09:59 UTC; the environment then rebooted at 14:20:39 UTC and `wazuh-analysisd` restarted at 14:21:31 UTC — after the fix, so the corrected rules are loaded in the live manager. `scripts/mitre_coverage_report.py` was re-run against the current `alerts.json`:

| Provenance | Count | Percentage |
|---|---|---|
| Total alerts examined | 120 | 100% |
| NATIVE_WAZUH | 26 | 21.7% |
| REVIEWED_RULE_MAPPING | 0 | 0% |
| DETERMINISTIC_INFERENCE | 0 | 0% |
| AMBIGUOUS | 0 | 0% |
| UNKNOWN | 94 | 78.3% |

None of the six specifically-fixed rule IDs (100510, 100511, 100513, 210001, 210011, and the sixth from Section 4.9) appear in this 120-alert window — it's dominated by Suricata APT-repo noise (rule 86601), dpkg housekeeping (2902/2904), and rule 100500 (Nmap reconnaissance, already correctly resolving to T1595 via NATIVE_WAZUH *before* this fix — it was never one of the broken rules). This is **not** evidence the fix failed; the specific attack types the fix targeted (brute force, port-scan variants, DoS) simply haven't been re-triggered against this environment since the reboot. **Status: rules active in the live manager, not yet exercised end-to-end by a matching alert.**
*Planned:* re-run `scripts/mitre_coverage_report.py` once brute-force, port-scan, or DoS traffic is generated against the environment, to confirm the fixed rules resolve end-to-end.

## Phase progression (verified against repository/logs, not blindly trusted)

| Phase | Claim | Verification result |
|---|---|---|
| 20A | Architecture approved | Confirmed — design document exists in conversation history; matches implemented code |
| 20B | `mitre_resolver.py`, `mitre_rule_registry.py`, resolver tests, integration tests, coverage report, unattributed path implemented | **Confirmed present in repo**: all files exist, 15+8=23 tests exist and pass |
| 20C | 8/8 pre-live validations passed | Confirmed via conversation record; re-verified structurally this session (control-flow proof still holds in current code) |
| 20D | First live activation exposed a campaign-resolution `TypeError` | Confirmed — root cause was `get_recent_inactive_campaign_db()` returning `list[dict]` while `load_from_database()` indexed it as a single dict; only reachable via `include_inactive=True`, which had no caller before Phase 20B |
| 20E | Fixed by removing `include_inactive=True`; 150/150 tests passed | **Confirmed**: `realtime_socgraph.py`'s `_process_unattributed_alert()` calls `resolve_campaign_context(attacker_ip, victim_ip)` with no `include_inactive` argument (defaults to `False`) — verified by direct code read this session |
| 20F | Live UNKNOWN path validated; listener operational; UNKNOWN events persisted | **Confirmed**: live listener (current PID 10228) has created real UNKNOWN AttackEvents with all correct properties (see below) |

## Coverage numbers — DISCREPANCY DISCLOSED (Rule 13/14)

The task brief cited a historical snapshot: **239 alerts, 3 NATIVE_WAZUH, 236 UNKNOWN, 0 REVIEWED/INFERENCE/AMBIGUOUS, ~98.7% UNKNOWN.**

**Fresh live measurement, 2026-08-31** (`scripts/mitre_coverage_report.py` re-run against the current `alerts.json`):

| Provenance | Count | Percentage |
|---|---|---|
| Total alerts examined | 391 | 100% |
| NATIVE_WAZUH | 39 | 10.0% |
| REVIEWED_RULE_MAPPING | 0 | 0% |
| DETERMINISTIC_INFERENCE | 0 | 0% |
| AMBIGUOUS | 0 | 0% |
| UNKNOWN | 352 | 90.0% |

**These two snapshots disagree** because real time has passed and more live traffic has been generated between the historical snapshot and now (391 vs 239 alerts — 152 new alerts arrived). The NATIVE_WAZUH share rose from 1.3% to 10.0%, driven mainly by rule `100500` (25 new occurrences, native `T1595`/Active Scanning — see Part J below) and `sshd` auth rules (5760/5715, 3 each) newly appearing. **Report the current 10.0%/90.0% split as the live number; the 98.7% figure is stale and should not be quoted as current.**

### Rule-level breakdown (live, top entries)

| Rule ID | Count | Status | Description |
|---|---|---|---|
| 40704 | 151 | UNKNOWN | Systemd service failure |
| 52002 | 66 | UNKNOWN | Apparmor DENIED |
| 86601 | 60 (39+21, two distinct message types) | UNKNOWN | Suricata info-level alerts |
| **100500** | **25** | **resolved (NATIVE_WAZUH → T1595)** | "HIGH SEVERITY: Nmap Reconnaissance Detected" (local custom rule) |
| 31101 | 24 | UNKNOWN | Web server 400 error |
| 203 | 20 | UNKNOWN | Wazuh agent event queue full |
| 503 | 10 | UNKNOWN | Wazuh agent started |
| 5503 | 3 | resolved (NATIVE_WAZUH → T1110.001) | PAM login failed |
| 5760 | 3 | resolved (NATIVE_WAZUH → T1110 family, sshd auth failure) | sshd authentication failed |
| 5715 | 3 | resolved (NATIVE_WAZUH) | sshd authentication success |
| 506 | 4 | resolved | Wazuh agent stopped |

## Scientific interpretation of the UNKNOWN rate (do not call this "bad performance")

**MITRE attribution coverage** and **attack detection capability** are different things:

- The 90% "UNKNOWN" rate measures how much of Wazuh's *own* out-of-the-box ruleset carries an ATT&CK tag for the telemetry currently arriving. Most of that telemetry — systemd failures, Apparmor denials, disk-usage warnings, package-manager traffic — is **legitimate operational noise, not attacker behavior**. Wazuh's own rule authors did not tag these as ATT&CK techniques either (verified: rules 40704, 52002, 2904, 86601, 2902, 531 have no `<mitre>` block in the installed Wazuh ruleset itself).
- Phase 20's contribution is *not* to raise this percentage. It is to make CYUKTI **capable of ingesting and preserving** all of that telemetry as evidence — rather than the pre-Phase-20 behavior, which discarded every alert without a native MITRE tag outright.
- A lower "resolved" percentage achieved by refusing to guess is the intended, correct outcome, not a shortfall. Manufacturing ATT&CK tags to raise this number would be a research-validity regression, not an improvement.

## Rule 100500/100501 (Part J) — empirical evidence, formally unresolved

Prior audit found rule IDs `100500` and `100501` defined **twice** in `local_rules.xml`: once with `<mitre>` tags (Nmap→T1595, cron→T1053.003), and again later in the same file as level-0 "ignore dpkg" rules with no MITRE tag. Which definition Wazuh actually loads could not be confirmed without root access to `wazuh-analysisd -t`.

**New evidence this session**: live alert data shows rule `100500` firing 25 times with `rule.mitre = {"id": ["T1595"], "tactic": ["Reconnaissance"], "technique": ["Active Scanning"]}` present in the raw Wazuh alert. This is **behavioral evidence** that the Nmap-detection definition is the one currently active — but it is not the same as a direct `wazuh-analysisd -t` confirmation, and the *other* duplicate (`100501`) has not fired in this window, so its status remains unconfirmed either way.

**Status: unresolved — requires root-level `wazuh-analysisd -t` validation to confirm definitively.** Do not claim it is resolved; do not guess which definition wins for `100501`.
*Planned fix:* run `wazuh-analysisd -t` with root access in a maintenance window to get a direct confirmation, then remove or renumber whichever rule is the duplicate.

## T1548.003 (Part K)

- **Confirmed absent from `MITRE_TO_STAGE`** (`'T1548.003' in MITRE_TO_STAGE` → `False`, checked live this session).
- **Native Wazuh resolution**: valid — rule `5401` ("Failed attempt to run sudo") carries a genuine native Wazuh mapping to `T1548.003` (Abuse Elevation Control Mechanism: Sudo and Sudo Caching), tactics `Privilege Escalation` + `Defense Evasion`.
- **Stage mapping**: `mitre_mapper.MITRE_TO_STAGE` has no entry for it, so any resolved event carrying this technique contributes `tps=0` via the `MITRE_TO_STAGE.get(id, "Unknown")` fallback — not because the resolution is wrong, but because CYUKTI's risk-stage taxonomy hasn't been extended to include it.
- These are **separate concerns**: MITRE attribution correctness (native, valid) vs. CYUKTI's own risk-scoring coverage (incomplete). Not fixed, per instruction.
  *Planned fix:* add a `T1548.003` entry to `MITRE_TO_STAGE` in the same pass as the next taxonomy update.

## UNKNOWN safety invariants — live-verified, not just claimed

5 real UNKNOWN AttackEvents inspected directly in Neo4j this session, all confirmed:

| Invariant | Result |
|---|---|
| No fabricated `attack_id` | Confirmed — property absent entirely from all 5 records |
| No `Technique` node/`MATCHES` relationship created | Confirmed — `MATCHES` query on the relevant campaign returns `[]` |
| `mitre_status="UNKNOWN"`, `mitre_provenance="UNKNOWN"`, `mitre_confidence="NONE"` | Confirmed on all 5 |
| `mitre_reason` and `mitre_resolver_version` present | Confirmed (`"1.0.0"`) |
| `mitre_technique_ids=[]` | Confirmed |
| `tps=0` | Confirmed (hardcoded in the Cypher, not parameterized — cannot be non-zero) |
| Full raw alert preserved | Confirmed — `investigation_payload` contains verbatim rule_id/description/groups/agent/location/timestamp |
| Valid `HAS_EVENT` relationship to a Campaign | Confirmed |
| `Campaign.last_technique` not overwritten by UNKNOWN | Confirmed — remains `''` (empty string, the pre-existing `CAMPAIGN_DEFAULTS` value for a brand-new campaign, not a fabricated technique) |
| No `NEXT_TECHNIQUE` contamination | Confirmed — edge count stayed at 3 before and after |
| No prediction triggered | Confirmed structurally: `predict_next()` is never called in the UNKNOWN code branch (line-number control-flow proof from Phase 20C stands, re-checked against current code) |
| No MISP publication for UNKNOWN | Confirmed structurally: `sync.publish_campaign()` is never called in the UNKNOWN branch |
| Multi-technique native preservation | Confirmed by test (`test_multiple_native_mitre_ids_are_preserved`, `test_multi_technique_native_alert_uses_first_as_primary_but_preserves_all`) — no live multi-technique native alert has been observed to independently confirm this in production traffic yet. *Planned:* confirm against a live multi-technique native alert once one is observed in production traffic. |
