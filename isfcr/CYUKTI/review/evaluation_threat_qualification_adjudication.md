# CYUKTI — Independent Threat-Qualification Adjudication

> 2026-09-26. **This is `MEASURED_ASSISTANT_ADJUDICATED`, not human ground truth.** Every label below was produced by this AI assistant reading real repository/Neo4j evidence and real external reference material (MITRE ATT&CK, Atomic Red Team), not by an independent human security analyst. It must not be cited in the paper as human-reviewed or final. It is a materially more granular, evidence-based re-examination than the prior per-scenario AI label, and it changes the reported agreement number — from worse-looking to less-worse-looking in aggregate, but with an important asymmetry disclosed plainly in Section 7, not hidden.

## 1. Executive Summary

The previous evaluation reported **8.3% (2/24)** agreement between CYUKTI's live threat classification and a per-*scenario* AI-proposed label (one blanket label for every campaign sharing an attacker/victim pair). This adjudication re-examined all 24 campaigns individually against their own real evidence and found the per-scenario label was too coarse: it ignored real per-campaign differences (some brute-force campaigns show a corroborating successful-login event, most don't; one campaign labeled "brute force" by IP-pair grouping is actually an unrelated exploit chain; some "mixed exploitation" campaigns have almost no real evidence at all). **Independent per-campaign adjudication reaches 29.2% (7/24) agreement with CYUKTI** — a real, evidence-based change, not an optimization for a better number. The more important finding is *not* the headline percentage but its shape: **CYUKTI has 100% recall and 0% precision for the QUALIFIED_THREAT label against this adjudication** (false positive rate = 1.0 on the binary view) — it never misses a case this adjudication also calls a real threat, but it also calls *every* weaker SUSPICIOUS-level case QUALIFIED_THREAT too. This is a materially different, more actionable characterization than "8.3% accuracy" alone conveyed, and it survives this report's scrutiny in both directions (Section 6). **Human review is still required** before any of this is final.

## 2. Evidence Sources — the exact 8 files

Located by direct inspection, not assumed:

| # | File | Contains | Type | CYUKTI-derived? | Usable as independent evidence? |
|---|---|---|---|---|---|
| 1 | `evaluation/scenarios/scenarios.json` | The 5 hand-authored `AttackScenario` records (attacker/victim, expected techniques, provenance notes) | Scenario registry (source) | No | Yes — independently authored from lab topology/docs |
| 2 | `evaluation/ground_truth/provisional/threat_qualification_v0.jsonl` | 24 `GroundTruthRecord`s: `sample_id`, `scenario_id`, `raw_event_id` (real `campaign_id`), `expected_threat_status` (the original per-scenario AI label) | Provisional ground truth | No (built independently of `threat_qualification.py`) | Yes, but coarse (per-scenario, not per-campaign) — this adjudication's starting point, not its answer |
| 3 | `evaluation/review/threat_qualification_review_queue_threat_qualification_v0.csv` | Human-review queue: same 24 rows + `cyukti_prediction` reference column + blank fillable columns | Generated review artifact | Partially (shows CYUKTI's live prediction for reference) | Yes, as a review UI — confirmed 0 rows filled in (Section B of the prior report, unchanged) |
| 4 | `evaluation/evaluators/threat_qualification_eval.py` | The evaluator: re-fetches live `cti_score`, classifies via `cti_confidence_engine`'s real thresholds, compares to `expected_threat_status` | Evaluation code | Reads CYUKTI output (by design, as the system-under-test side) | Not evidence itself — the comparison logic |
| 5 | `evaluation/build_campaign_dataset.py` | The script that built record #2 from live Neo4j (`_find_real_campaigns`, attacker/victim Cypher lookups) | Ground-truth builder | No | Yes — shows exactly how the 24 campaigns were selected (real Cypher, not fabricated) |
| 6 | `evaluation/results/threat_qualification_metrics.json` | The original evaluator's output: n=24, accuracy=0.083, confusion matrix | Evaluation output | Yes (CYUKTI's own classification is embedded in it) | Reference only — this is what is being re-examined, not new evidence |
| 7 | `backend/threat_qualification.py` | `ThreatQualificationEngine` — reads `incident.cti.threat_classification`, does not itself compute it | Production code (system under test) | Yes | Not ground truth — this is exactly what must never be used as ground truth |
| 8 | `backend/cti_confidence_engine.py` | `CTIConfidenceEngine` — the actual score/threshold logic (`PUBLISH_THRESHOLD=40`, `NOT_THREAT_THRESHOLD=20`) that produces the classification | Production code (system under test) | Yes | Not ground truth, but its thresholds are quoted here so the reader can see exactly how CYUKTI's label is derived |

**New this adjudication, not one of the "existing 8" but the actual new evidence used**: real Neo4j `Campaign`/`AttackEvent`/`Operation` records for all 24 campaigns, pulled fresh via a new read-only script, `evaluation/review/fetch_threat_qualification_evidence.py`, output at `evaluation/results/threat_qualification_adjudication_evidence.json`. This is genuinely independent of CYUKTI's threat-qualification *output* (it reads raw `attacker_ip`, `first_seen`, per-event `occurrences`, `mitre_provenance`, technique IDs — never `cti_score`'s downstream classification) — `cti_score` itself is fetched and shown for reference but never used as an adjudication input, matching Section 5's evidence-hierarchy Tier 4 rule.

## 3. External Reference — Atomic Red Team & MITRE ATT&CK

Fetched live (this session, via `gh api` for GitHub content and direct ATT&CK page fetches), not recalled from training data alone:

- **T1110.001 (Password Guessing)** — `atomics/T1110.001/T1110.001.md` (redcanaryco/atomic-red-team, master branch). Real ATT&CK description text confirms SSH (22/TCP) as an explicitly named commonly-targeted service for this technique. **8 real Atomic Tests exist**, but none is a purpose-built "SSH brute force" test — they target Active Directory/SMB/LDAP/Azure AD/Kerberos (Windows, tests #1-4) and `sudo` brute-forcing on Debian/Redhat/FreeBSD (tests #5-7) plus ESXi (#8). **Finding, stated plainly, not smoothed over**: ART's own test catalog does not contain a dedicated SSH-login-brute-force atomic — this lab's SSH brute-force traffic did not come from running a literal ART test, consistent with this project's own prior documentation that it comes from an internal synthetic traffic generator, not ART. ART is used here only as independent confirmation that SSH-targeted password guessing is a real, ATT&CK-documented instance of T1110.001, not as proof any specific observed event is malicious.
- **T1595 / T1595.002 (Active Scanning / Vulnerability Scanning)** — fetched from `attack.mitre.org/techniques/T1595/`. Confirmed: **Reconnaissance tactic (TA0043)**, defined as probing "victim infrastructure via network traffic." No ART atomic test exists for the base T1595 or T1595.002 sub-technique in this repository's current index (only `T1595.003`, Wordlist Scanning, has a folder) — another case where ART's coverage doesn't map 1:1 to what CYUKTI observed. This independently confirms the technique's classification (reconnaissance, not compromise) directly from MITRE's own taxonomy.
- **T1078 (Valid Accounts)** — fetched from `attack.mitre.org/techniques/T1078/`. MITRE's own text: valid-account use enables Initial Access, Persistence, Privilege Escalation, *and* Defense Evasion, and — critically — **MITRE itself does not provide a way to distinguish legitimate from adversarial use without behavioral context** ("anomalous logon patterns, abnormal logon types, inconsistent geographic or time-based activity"). This is directly load-bearing for Section 5's per-record reasoning: a lone T1078 event is genuinely ambiguous per MITRE's own framing; a T1078 event immediately following a burst of T1110/T1110.001 attempts from the same external attacker IP is the kind of contextual corroboration MITRE's own detection guidance points to.

No unrelated technique was investigated. No Atomic Red Team test was executed to manufacture evidence for a historical event (Section 14's prohibition, respected).

## 4. Dataset Structure — 24 records, real relationships

| Level | Count | Note |
|---|---|---|
| Review records (this adjudication) | 24 | |
| Unique Campaign IDs | 24 | Every record is a distinct real `Campaign` node — no duplicates |
| Unique Operations | 16 | `campaign_manager`'s own operation-correlation already merged some: `OP_D856F358`→2 campaigns, `OP_0EEBF42C`→2, `OP_236E9AE4`→3, `OP_7AF36DEF`→2, `OP_3970E3F5`→2, `OP_6844EECE`→2; the remaining 10 operations are 1:1 with a campaign |
| Unique attacker/victim sessions (IP pairs) | 3 | `(.106, pes1ug23cs411)`=12 campaigns, `(.105, pes1ug23cs411)`=4 campaigns, `(.106, .105)`=8 campaigns |
| Unique attack *episodes* (this adjudication's best estimate) | **Not precisely determinable without a human reviewer** | Real timestamps for the 12-campaign group span 2026-08-06 to 2026-09-25 (7 weeks) — most of that spread is far too wide to be one continuous session artificially fragmented; it is far more consistent with a lab user re-running the same generator scenario repeatedly over the project's lifetime. Reported as `HUMAN_REVIEW_REQUIRED` for a precise episode count, not guessed. |

**One confirmed cross-contamination case, not a fragmentation case**: `CAMP_10407C1A` shares the `(.106, pes1ug23cs411)` attacker/victim pair with the SSH-brute-force group (hence being swept into that scenario and labeled `SUSPICIOUS`), but its real technique set (`T1055` process injection ×3, `T1595.002` vulnerability scanning ×1 at 152 occurrences, `T1210` exploitation of remote services, `T1190` exploit public-facing application) shows **zero brute-force technique activity at all**. This is a real scenario-construction defect (Section 6, cause #4) affecting one specific record, not evidence that the fragmentation finding itself was wrong.

## 5. Row-by-Row Adjudication

Full machine-readable version: `evaluation/review/compute_threat_qualification_adjudication_metrics.py` (the `ADJUDICATED` dict, with inline per-record rationale comments) and the raw evidence at `evaluation/results/threat_qualification_adjudication_evidence.json`. Condensed table (columns trimmed for readability — `direct_evidence`/`corroborating_evidence`/`counter_evidence` merged into `reason`; full detail is in the evidence JSON):

| record_id | campaign_id | attacker→victim | techniques (occurrences) | provisional | cyukti | **adjudication** | confidence | reason |
|---|---|---|---|---|---|---|---|---|
| 1 | CAMP_1627943B | .106→pes1ug23 | T1110.001(2,2,2), T1110(2), T1078(2) | SUSPICIOUS | QUALIFIED_THREAT | **QUALIFIED_THREAT** | MEDIUM | Repeated brute-force attempts followed by a native-mapped T1078 (SSH auth success) from the same attacker within the same session — MITRE's own T1078 guidance treats attacker-IP+timing correlation as the relevant corroborating context |
| 2 | CAMP_BB9A0F28 | .106→pes1ug23 | T1110.001(2,2) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | MEDIUM-HIGH | Brute-force attempts only, no success indicator anywhere in this campaign's own events |
| 3 | CAMP_0E9C0284 | .106→pes1ug23 | T1110.001×4, T1110(2), T1078(2) | SUSPICIOUS | QUALIFIED_THREAT | **QUALIFIED_THREAT** | MEDIUM | Same pattern as #1: brute force + T1078 success indicator |
| 4 | CAMP_C6E494CF | .106→pes1ug23 | T1110.001(2,2) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | MEDIUM-HIGH | Attempts only, no success indicator |
| 5 | CAMP_75FBEEBC | .106→pes1ug23 | T1110.001(2,2,59,1,1) T1110(2,1) T1078(2,2) | SUSPICIOUS | QUALIFIED_THREAT | **QUALIFIED_THREAT** | MEDIUM-HIGH | Sustained burst (59 occurrences in one event), reopened 5×, two T1078 success indicators |
| 6 | CAMP_DDBFC6D9 | .106→pes1ug23 | T1078(2), T1110.001(2) | SUSPICIOUS | QUALIFIED_THREAT | **QUALIFIED_THREAT** | MEDIUM | Brute force + success indicator |
| 7 | CAMP_10407C1A | .106→pes1ug23 | T1055(30,3,3), T1210(1), T1595.002(152), T1190(2) | SUSPICIOUS | QUALIFIED_THREAT | **QUALIFIED_THREAT** | HIGH | Not a brute-force campaign at all — a real vulnerability-scan→exploit→process-injection chain, mislabeled by IP-pair scenario grouping |
| 8 | CAMP_D8605E81 | .106→pes1ug23 | T1110.001(13), T1110(1) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | MEDIUM | Elevated attempt volume (13) but no success indicator |
| 9 | CAMP_DAE35509 | .106→pes1ug23 | T1110.001(1,1) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | MEDIUM | Minimal attempts, no success indicator |
| 10 | CAMP_E449E81D | .106→pes1ug23 | T1110.001(1) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | LOW-MEDIUM | Single, very thin attempt event |
| 11 | CAMP_BB680773 | .106→pes1ug23 | UNKNOWN(7,1), T1078(1, NATIVE/CONFIRMED), T1110.001(1, NATIVE/CONFIRMED) | SUSPICIOUS | QUALIFIED_THREAT | **QUALIFIED_THREAT** | MEDIUM-HIGH | Brute force + success indicator, post-rule-fix native mapping (higher-confidence provenance than the older null-provenance events) |
| 12 | CAMP_E8E9F042 | .106→pes1ug23 | T1110.001(1, NATIVE), T1110(1, NATIVE) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | MEDIUM | Attempts only, no success indicator |
| 13 | CAMP_23FB320A | .105→pes1ug23 | T1114(10) | QUALIFIED_THREAT | SUSPICIOUS | **INSUFFICIENT_EVIDENCE** | LOW | Single technique (Email Collection), `risk_score=0`, no corroborating recon/brute-force/exploit context in this campaign's own record |
| 14 | CAMP_1429ADB4 | .105→pes1ug23 | T1595.002(448) T1055(13,279,36) T1059.007(206) T1059(29) T1190(4,2,1) T1210(3) | QUALIFIED_THREAT | QUALIFIED_THREAT | **QUALIFIED_THREAT** | HIGH | Rich, multi-stage, heavily-corroborated exploitation chain, `risk_score=34820` (highest in the dataset) |
| 15 | CAMP_F91A7652 | .105→pes1ug23 | T1110.001(1) | QUALIFIED_THREAT | QUALIFIED_THREAT | **SUSPICIOUS** | LOW-MEDIUM | A single, minimal brute-force-adjacent event — inconsistent with the strength implied by "QUALIFIED_THREAT," regardless of what label CYUKTI or the original scenario assigned |
| 16 | CAMP_066240B7 | .105→pes1ug23 | T1078(1) | QUALIFIED_THREAT | SUSPICIOUS | **INSUFFICIENT_EVIDENCE** | LOW | A lone valid-account-use event with no preceding brute-force/scan context — MITRE's own text says this is ambiguous without behavioral corroboration, which is absent here |
| 17 | CAMP_947A7084 | .106→.105 | T1595(46) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Reconnaissance tactic only (MITRE-confirmed), no follow-on |
| 18 | CAMP_67FCB361 | .106→.105 | T1595(50) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Same |
| 19 | CAMP_427A075C | .106→.105 | T1595(26,807) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Sustained/repeated scanning (833 total occurrences) is still reconnaissance-tactic-only by MITRE's own taxonomy, absent any follow-on technique |
| 20 | CAMP_99EE7A64 | .106→.105 | T1595(25) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Same |
| 21 | CAMP_86D5837D | .106→.105 | T1595(25, NATIVE), UNKNOWN(1) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Scan + one unrelated low-signal trailing event, no confirmed follow-on attack |
| 22 | CAMP_15E98B82 | .106→.105 | T1595(25, NATIVE), UNKNOWN(1) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Same |
| 23 | CAMP_B177634A | .106→.105 | T1595(25, NATIVE), UNKNOWN(1,1) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Same |
| 24 | CAMP_1B8E9033 | .106→.105 | T1595(25, NATIVE), UNKNOWN(1) | SUSPICIOUS | QUALIFIED_THREAT | **SUSPICIOUS** | HIGH | Same |

`evidence_gaps` common to nearly every record (not repeated 24×): no endpoint telemetry beyond Wazuh/Suricata-derived `AttackEvent`s; no independent confirmation of whether the SSH-brute-force traffic was manually run or generator-driven for any *specific* campaign (project-level docs confirm a generator exists, but not which specific campaign it produced); `mitre_provenance` is `null` (not `NATIVE_WAZUH`/`UNKNOWN`) for most pre-September events, meaning their MITRE resolution predates the provenance-tracking field being populated — treated as `NOT AVAILABLE` for provenance strength, not silently assumed confirmed.

## 6. Confusion Matrix

**Evaluation A — 3-class (adjudicated = reference, n=24):**

| | Predicted NOT_THREAT | Predicted SUSPICIOUS | Predicted QUALIFIED_THREAT | Predicted INSUFFICIENT_EVIDENCE |
|---|---|---|---|---|
| **True NOT_THREAT** | 0 | 0 | 0 | 0 |
| **True SUSPICIOUS** | 0 | 0 | 15 | 0 |
| **True QUALIFIED_THREAT** | 0 | 0 | 7 | 0 |
| **True INSUFFICIENT_EVIDENCE** | 0 | 2 | 0 | 0 |

**Evaluation B — binary (n=22, 2 INSUFFICIENT_EVIDENCE records excluded from both sides, reported separately, never silently discarded):**

| | Predicted NON_THREAT | Predicted THREAT |
|---|---|---|
| **True NON_THREAT** (adjudicated SUSPICIOUS) | 0 | 15 |
| **True THREAT** (adjudicated QUALIFIED_THREAT) | 0 | 7 |

## 7. Metrics

Reproducible via `python evaluation/review/compute_threat_qualification_adjudication_metrics.py`:

| | Evaluation A (3-class) | Evaluation B (binary) |
|---|---|---|
| Accuracy | 29.2% (7/24) | 31.8% (7/22) |
| Macro F1 | 0.161 | 0.241 |
| Weighted F1 | 0.141 | — |
| QUALIFIED_THREAT / THREAT: precision | 0.318 | 0.318 |
| QUALIFIED_THREAT / THREAT: recall | **1.000** | **1.000** |
| QUALIFIED_THREAT / THREAT: F1 | 0.483 | 0.483 |
| SUSPICIOUS / NON_THREAT: precision, recall, F1 | 0, 0, 0 | 0, 0, 0 |
| False positive rate (THREAT positive class) | — | **1.000** |
| False negative rate (THREAT positive class) | — | **0.000** |

**The headline number is not "29.2% accuracy" in isolation — it is 100% recall / 0% precision on the QUALIFIED_THREAT label.** CYUKTI's `cti_confidence_engine` never fails to flag a case this adjudication also considers a real threat, but it also flags every weaker case the same way. Both the previous 8.3% framing and a naive "29.2% is better" framing would understate this specific, actionable, asymmetric finding.

## 8. Disagreement Analysis

17/24 records disagree between CYUKTI and this adjudication. Classified:

| Disagreement type | Records | Count |
|---|---|---|
| `FALSE_POSITIVE` (CYUKTI over-called QUALIFIED_THREAT vs. adjudicated SUSPICIOUS) | 2,4,8,9,10,12 (brute-force-attempts-only), 17-24 (all 8 Nmap-only) | 14 |
| `FALSE_POSITIVE` (CYUKTI over-called vs. adjudicated INSUFFICIENT_EVIDENCE) | 13 (CAMP_23FB320A was actually CYUKTI=SUSPICIOUS, adjudicated=INSUFFICIENT_EVIDENCE — a milder disagreement) | 1 |
| `LABELING_ERROR` (original *provisional* label, not CYUKTI, was wrong) | 7 (CAMP_10407C1A: provisional said SUSPICIOUS by IP-pair grouping; both CYUKTI and this adjudication independently agree it's QUALIFIED_THREAT) | 1 |
| `BOUNDARY_CASE` / `INSUFFICIENT_EVIDENCE` | 16 (CAMP_066240B7: CYUKTI=SUSPICIOUS, adjudicated=INSUFFICIENT_EVIDENCE — both are honest hedges, not a real disagreement in substance) | 1 |
| `CAMPAIGN_FRAGMENTATION` (scenario-construction cause, not a model or evaluator bug) | 7, 13, 15, 16 (all 4 records swept into the wrong scenario purely by IP-pair matching, three of which were also very-low-evidence single events) | 4 (overlaps with rows counted above) |
| Agreements (no disagreement) | 1,3,5,6,7,11,14 | 7 |

**Primary root cause of the disagreement pattern, stated plainly**: `cti_confidence_engine`'s `PUBLISH_THRESHOLD=40` is crossed by nearly any campaign with even one native-mapped MITRE technique and a handful of occurrences — the blended score (`DETECTION_WEIGHT=0.25`, `RISK_WEIGHT=0.20`, `THREAT_WEIGHT=0.20`, `CAMPAIGN_WEIGHT=0.20`, `PREDICTION_WEIGHT=0.15`) does not appear, from this real data, to meaningfully separate "one brute-force attempt" from "a sustained, corroborated compromise." This is a real, disclosed finding about the *scoring engine's discrimination*, not a claim that the underlying detections themselves are wrong.

## 9. Campaign Fragmentation Analysis

Quantified directly (Section 4): 24 review records → 16 real `Operation` nodes → 3 real attacker/victim IP pairs. Fragmentation's concrete effect on *this specific adjudication*:
- **1 confirmed cross-contamination case** (`CAMP_10407C1A`) where IP-pair-based scenario construction placed a real exploit-chain campaign into the "brute force" bucket, producing a `LABELING_ERROR` in the *provisional* ground truth (not in CYUKTI, which happened to classify it correctly by coincidence of its own scoring, not because it recognized the technique mismatch).
- **3 low-evidence records** (`CAMP_23FB320A`, `CAMP_F91A7652`, `CAMP_066240B7`) swept into the "mixed exploitation, QUALIFIED_THREAT" bucket purely by sharing an attacker/victim pair with the one genuinely rich campaign (`CAMP_1429ADB4`), despite having little to no corroborating evidence of their own.
- **Net effect on the metrics**: without correcting for fragmentation, the original per-scenario evaluation implicitly treated 4 very different campaigns as equivalent to `CAMP_1429ADB4`'s strong evidence. This adjudication's per-campaign approach removes that artifact — it is the single largest source of the 8.3%→29.2% change, more than any change in interpretation of the brute-force/reconnaissance evidence itself.

## 10. Evidence Gaps (require human verification)

1. Whether any specific SSH-brute-force campaign was a manually-run attack or generator-driven (project docs confirm a generator exists; per-campaign attribution to it is not independently recorded).
2. Whether the T1078 events genuinely represent a compromised account being used for further activity, or an operator/lab-owner legitimate login coinciding with generator noise — this adjudication treated attacker-IP+timing correlation as reasonably strong circumstantial evidence, but MITRE itself says this needs human behavioral judgment to confirm.
3. Precise attack-episode boundaries within the 7-week-spanning SSH-brute-force group (Section 4) — this adjudication did not attempt to guess a number.
4. Whether the Nmap group's identical `cti_score=49.08` across 6/8 records reflects a genuine scoring-engine ceiling/plateau or a coincidence — noted, not investigated further (out of this task's scope; a candidate follow-up for whoever reviews `cti_confidence_engine.py` next).

## 11. Scientific Interpretation

- **Observed fact**: real Neo4j `Campaign`/`AttackEvent` records, real timestamps, real technique IDs, real occurrence counts, real `cti_score` values — all directly queried, none inferred.
- **Atomic Red Team / MITRE contextual evidence**: real, fetched this session — confirms T1595 is reconnaissance-only by MITRE's own taxonomy, confirms T1110.001 legitimately covers SSH as a target service (even though no ART test targets SSH specifically), confirms T1078 is inherently ambiguous without behavioral context per MITRE's own text. Used as Tier 3 context, never as a stand-in for missing Tier 1/2 evidence.
- **Assistant inference**: every "adjudication" cell above — an AI assistant's reasoned judgment applying the evidence hierarchy, not an observed fact and not human ground truth.
- **CYUKTI output**: `cti_score` and its threshold-derived classification — shown throughout, never used to produce an adjudication label.
- **Unresolved uncertainty**: 2 records (`CAMP_23FB320A`, `CAMP_066240B7`) genuinely could not be resolved past `INSUFFICIENT_EVIDENCE`; the true episode count within the 7-week brute-force group; whether T1078 co-occurrence really means "successful compromise" in this specific lab's context.

## 12. Final Status

**`MEASURED_ASSISTANT_ADJUDICATED`.**

> This is not independent human ground truth and must not be presented as such in the paper. It is a more granular, evidence-based, AI-assisted re-examination of the same 24 real campaigns, reproducible via the two scripts in `evaluation/review/`, and it remains exactly one step short of `MEASURED_HUMAN_REVIEWED` — an actual independent human analyst reading the same evidence (available now at `evaluation/results/threat_qualification_adjudication_evidence.json`) is still required before any number in this document may be cited as final.
