# B3 Design Decision Report
## Campaign Activity Semantics, Restart Consistency, and Lifecycle Anomalies

**Date:** 4 November 2026
**Project:** CYUKTI
**Study:** B3 v2 — CUR, A, and B campaign activity policies
**Dataset:** 9,015 Wazuh alerts

### 1. Executive decision

**Decision: retain CUR as the reference implementation for now. Do not promote A or B as the selected production policy.**

The experiment establishes that all three policies can process the archive successfully and that campaign grouping was unchanged by the single graceful restart tested. It does not establish which policy produces more accurate campaigns.

The policies represent different activity-recording semantics:

- **CUR:** preserves the existing distinction between in-memory campaign activity and persisted Neo4j activity.
- **A:** extends campaign activity for UNKNOWN alerts and duplicates, including activity associated with campaigns that may already have closed.
- **B:** restricts activity extension to resolved, new alerts and removes the relevant duplicate/UNKNOWN activity updates.

B is the most restart-consistent implementation in this experiment. However, its zero `closed_at < last_seen` anomalies are achieved partly by not recording certain activity. That is not sufficient evidence that B represents campaign lifecycle correctly.

A reduces campaign count, but its reduction is concentrated in self-loop identities and it increases score-based closures. It should not be selected on the basis of fewer campaigns.

CUR is retained as a baseline, not declared correct.

### 2. Experimental findings

All four policy/restart experiments completed with exit code 0 and processed all 9,015 alerts. No tracebacks, alert errors, watchdog events or boundary violations were reported.

| Policy | No-restart campaigns | Restart campaigns | Restart grouping |
|---|---:|---:|---|
| CUR | 116 | 116 | Identical |
| A | 97 | 97 | Identical |
| B | 116 | 116 | Identical |

B's restart and no-restart output hashes are identical. CUR and A show limited persisted-field differences despite identical campaign grouping.

The per-alert paths and duplicate decisions are identical across all six runs. Therefore, the policy changes studied here do not alter which alerts are treated as duplicates.

#### CUR versus B

B produces the same campaign assignments, operations and transitions as CUR on this archive. Its changes are primarily in persisted activity timestamps and reopen scores.

B's byte-identical restart/no-restart output is a useful determinism result. It is not a campaign-quality result.

#### A versus CUR

A produces 19 fewer campaigns. Seventeen of those reductions are associated with two dominant self-loop identities:

- `Omkar → Omkar`: 9 fewer campaigns.
- `pes1ug23cs411-VirtualBox → itself`: 8 fewer campaigns.
- Two other pairs account for one fewer campaign each.

Only 85 campaign memberships remain identical between A and CUR. The largest campaign grows from 14 to 27 events, and two transitions each gain one occurrence.

A also changes lifecycle decisions:

| Decision | CUR/B | A |
|---|---:|---:|
| CLOSE_EXPIRED | 96 | 74 |
| CLOSE_BELOW_THRESHOLD | 21 | 35 |
| FAST_REOPEN | 10 | 22 |
| CONTINUE | 71 | 62 |

A therefore shifts decisions away from expiry closure and toward score-based closure and reopening. Since the chain-similarity score can exceed 1, this is a material concern. A should not be adopted until the score defect and fragmentation mechanism are addressed.

### 3. Restart and buffer interpretation

The single graceful restart did not change grouping under any policy. This supports a narrow conclusion: the tested restart path did not alter campaign assignments for this archive and restart point.

It does not establish correctness under multiple restarts, other restart locations, process crashes, unflushed buffers or periodic flushing.

The duplicate-buffer attribution issue explains the observed CUR and A `last_seen` differences. Buffered duplicate records can be credited to the campaign associated with the fingerprint buffered first. This can move timestamps between campaigns and create misleading persisted lifecycle fields.

A's C0025 `closed_at` difference appears consistent with `touch()` not updating the campaign after its in-memory context has been lost during restart. This is an inference from output, not a directly observed execution trace. It requires targeted instrumentation before being treated as a confirmed mechanism.

CUR's reopen-score difference also remains unexplained. It did not cross a decision threshold, but the cause should be traced before claiming full restart determinism.

**Design implication:** activity timestamps need a clearly specified ownership and persistence model. A future implementation should define whether duplicate activity belongs to the campaign that originally received the alert, the campaign currently associated with the fingerprint, or another explicit entity. Buffer flush timing must not silently determine attribution.

### 4. UNKNOWN-only operations

The earlier replay found 63 of 116 campaigns without an associated operation; all 63 were UNKNOWN-only campaigns.

This is not merely an operation-manager issue. It exposes a dependency between ATT&CK resolution and higher-level correlation.

If operation creation or association depends on resolved techniques, UNKNOWN-only campaigns may remain isolated even when their raw alerts contain useful correlation evidence, such as:

- source and destination identity;
- host identity;
- timestamps and temporal proximity;
- IOC or network indicators;
- Wazuh rule identifiers and raw event attributes.

**Recommendation: do not fabricate ATT&CK techniques to make these campaigns eligible for operations.**

Instead, evaluate a separate evidence-based operation association path. It should be possible to associate campaigns using non-ATT&CK evidence while retaining explicit uncertainty and provenance.

The operation layer should distinguish at least:

1. **Technique-supported association:** supported by resolved ATT&CK evidence.
2. **Telemetry-supported association:** supported by host, IOC, identity or temporal evidence.
3. **Unresolved association:** plausible correlation, but insufficient evidence for a confident link.

The operation model should preserve the basis and confidence of each association. An UNKNOWN-only campaign should not be forced into an operation merely to improve coverage.

Required validation:
- Measure operation association coverage separately for resolved and UNKNOWN-only campaigns.
- Measure false associations using labelled attacks.
- Test cross-host sequences where technique labels are absent.
- Report how many UNKNOWN-only campaigns remain unassociated and why.

### 5. Self-loop identity fallback

The self-loop fallback represents an important data-quality problem.

When attacker and victim identity cannot be resolved independently, the fallback can assign the same identity to both sides. This creates pairs such as `Omkar → Omkar` and `pes1ug23cs411-VirtualBox → itself`.

These self-loops are not reliable evidence of a real attacker-to-victim relationship. They can distort campaign grouping, make A's lower campaign count appear more attractive than it is, and contaminate downstream operation and attribution features.

**Recommendation: do not treat a self-loop fallback as a confirmed attacker-victim pair.**

Retain the underlying event, but distinguish identity provenance:

- `OBSERVED`: independently extracted from telemetry.
- `INFERRED`: derived using a documented correlation rule.
- `FALLBACK`: substituted because identity was unavailable.
- `UNKNOWN`: no defensible identity could be established.

A fallback identity should not be used as a strong campaign-pair key. Prefer a nullable or explicitly unknown endpoint identity, with host and event evidence retained separately.

Do not delete historical self-loop records or silently rewrite their identity. Any remediation should be a separately versioned data migration with before/after counts and provenance.

Required validation:
- Count self-loops by source and fallback mechanism.
- Measure how many disappear when fallback identities are excluded from pair-based correlation.
- Check whether legitimate same-host activity is being incorrectly discarded.
- Re-run campaign reconstruction against labelled data before changing production grouping.

### 6. `closed_at < last_seen` anomalies

The observed anomaly counts are:

| Policy | Anomalies |
|---|---:|
| CUR, no restart | 2 |
| CUR, restart | 3 |
| A, no restart | 6 |
| A, restart | 7 |
| B, both | 0 |

B's zero count must not be interpreted as proof that lifecycle handling is correct. B suppresses activity updates that could otherwise produce the anomaly.

A's higher count is consistent with its broader activity-extension semantics, including duplicates arriving after campaign closure. CUR also has known buffer-attribution effects.

The Step 2 report's earlier 0-anomaly and 26/101 figures were based on a harness that did not flush the duplicate buffer. They must be treated as superseded by the flushed B3 measurements, with the historical correction preserved as an addendum rather than silently rewriting the earlier report.

### 7. Decision matrix

| Criterion | CUR | A | B |
|---|---|---|---|
| Existing behavior/reference continuity | Strong | Modified | Modified |
| Grouping restart consistency in tested case | Yes | Yes | Yes |
| Byte-identical restart output | No | No | Yes |
| Duplicate/UNKNOWN activity persistence | Inconsistent across memory/DB paths | Broadly extended | Suppressed |
| `closed_at < last_seen` count | Nonzero | Higher | Zero, but partly by omission |
| Campaign count reduction | No | Yes, but self-loop dominated | No |
| Ready for production selection | No | No | No |

### 8. Recommended engineering sequence

**Step 1 — Freeze and preserve B3 evidence.**

Keep all six run outputs, checksums, comparison tables and historical corrections immutable. No further policy experiment should overwrite these artifacts.

**Step 2 — Instrument duplicate attribution and lifecycle transitions.**

Add diagnostic-only tracing in a separate development branch or isolated workspace. Capture:
- fingerprint and duplicate classification;
- original campaign association;
- campaign association at buffer time;
- buffer insertion and flush timestamps;
- whether the campaign is active, closed or absent from memory;
- in-memory and Neo4j `last_seen` before and after the operation;
- reopen score inputs and decision result.

Do not alter scoring or campaign assignment while collecting this evidence.

**Step 3 — Decide the activity contract.**

Specify which events count as campaign activity and which entity owns that activity. Define separate semantics for:
- resolved new alerts;
- UNKNOWN new alerts;
- resolved duplicates;
- UNKNOWN duplicates;
- late duplicates after campaign closure.

The contract must apply consistently across in-memory state, persistence, restart recovery and periodic maintenance.

**Step 4 — Correct campaign scoring and fragmentation.**

This remains the primary reviewer concern. The current B3 study did not fix or validate the chain-similarity score flaw.

The next scoring work should address:
- unseen transitions being treated as zero evidence;
- recording split transitions so the model can learn them;
- overlap and double-counting between prediction and chain evidence;
- score calibration and threshold selection;
- scores exceeding their intended range.

Do not use B3 campaign counts as a substitute for labelled evaluation.

**Step 5 — Validate on labelled data.**

Use a dataset with defensible ground truth, such as AIT if suitable and accessible, and expanded repeatable lab scenarios. Report:
- pairwise precision, recall and F1;
- campaign purity and inverse purity;
- fragmentation and over-merging;
- campaign count per ground-truth attack;
- confidence intervals or variation across repeated runs.

The target of one or two campaigns per attack can be a diagnostic observation, not a hardcoded optimization target.

**Step 6 — Re-evaluate downstream components.**

Only after campaign reconstruction is validated should operation correlation, severity, attribution, GNN and investigation results be treated as evidence of system-level performance.

### 9. Explicitly out of scope for this decision

This study does not establish:
- detection accuracy or campaign reconstruction quality;
- the best activity policy;
- behavior under crashes without graceful flush;
- behavior under multiple or differently placed restarts;
- behavior under periodic event-time buffer flushing;
- generalization to another archive;
- behavior with a populated ATT&CK corpus.

The replay environment's missing ATT&CK data also means ATT&CK-dependent risk and severity outputs should not be interpreted as representative production outcomes.

### 10. Final decision

Retain CUR as the frozen comparison reference. Do not merge A or B as the selected activity policy. Do not interpret B's zero lifecycle anomalies or A's reduced campaign count as evidence of superiority.

The immediate next engineering task is **instrumentation and specification of campaign activity ownership**, not another full-archive policy run. In parallel, the central paper remediation remains campaign-scoring correction and labelled reconstruction evaluation.

B3 is a successful temporal-semantics and restart experiment. It is not yet a campaign-quality validation.
