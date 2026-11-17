# Documentation reconciliation report: campaign activity contract

**Status:** documentation only. These are review-ready documents, not approved specifications. D1 to D12 remain open. No code, test, service, database or frozen evidence was changed, and no criterion was run.
**Date:** 2026-11-04.

## 1. Documents

The four drafts are kept unchanged. The review copies supersede them for review, but the drafts still contain the wording corrected below.

| Draft (kept) | sha256 | Review copy |
|---|---|---|
| `CAMPAIGN_ACTIVITY_CONTRACT_DRAFT.md` | `a783096966dc378c…` | `CAMPAIGN_ACTIVITY_CONTRACT_FINAL_DRAFT.md` |
| `EVIDENCE_TRACEABILITY.md` | `adb928eead34bd8c…` | `EVIDENCE_TRACEABILITY_FINAL.md` |
| `OPEN_POLICY_DECISIONS.md` | `b76dc3d9c15a65ae…` | `OPEN_POLICY_DECISIONS_FINAL.md` |
| `VALIDATION_CRITERIA.md` | `b2959e7d90c25845…` | `VALIDATION_CRITERIA_FINAL.md` |

**Source version checked:** HEAD `3afe62c` plus uncommitted Steps 1 to 2A, backend diff sha256 `804bdeb06d2f949c…`, the same as when the drafts were written.

## 2. Where the brief changed once checked against the code

Three items in the brief needed more precision. The documents follow the code, and each difference is stated here for review.

| Brief | What the code shows | How the documents handle it |
|---|---|---|
| Flush writes against the campaign association retained in the record | True in effect, but the write goes through the retained `event_id`, then `AttackEvent.campaign_id`. The writer never reads the record's `campaign_id`. They matched in 18 of 18 flushed records | "Mutated campaign" is defined by the actual path |
| Include `investigation_payload` with the existing campaign fields | It is an `AttackEvent` property, not a `Campaign` one, and it is replaced, not added | D12 covers it, grouped by node. The same write also changes other `AttackEvent` and `Technique` fields, so those are listed too |
| V8.5 must define the offset boundary, with decisions kept to D1 to D12 | No decision covered the offset boundary, so V8.5 had no source for its expected result | **Sub-decision D7-O added under D7.** D1 to D12 themselves are unchanged. Please confirm this placement |

## 3. Mandatory corrections

| Brief item | Result | Location |
|---|---|---|
| 2A Duplicate attribution | Retained, classified, mutated and intended campaign defined and used throughout. D5 consequences given per alternative, none selected | Contract A.1, B, C; D5 |
| 2B Association vs mutation | I4 split into I4a and I4b, plus a matrix of what each invariant applies to. States that association consistency doesn't guarantee mutation consistency | Contract A.4 |
| 2C D12 scope | All five named fields covered, plus the others the write changes. Activity and risk semantics kept separate. No risk formula introduced | D12, D12-F |
| 2D Restart modes | Modes H, L, C and D defined. B3 is labelled as mode H only | Contract E; V5 |
| 2E `closed_at` | Three alternatives, each with its effect on ACTIVE campaigns, history and consumers | D10 |
| 2F Flush failure | Six cases for both worker and shutdown paths. `logger` remains a static finding. V6 and V8 stated as not passed | Contract F.1, F.2; V6, V8 |
| 3 D3 | Sub-decision D3-C with C1 to C4. Reopen evaluation flagged as changing grouping and untested | D3-C |
| 3 D6 | Five alternatives with what flush points depend on and what determinism they can offer. Wall-clock speed-dependence stated. Determinism not implied for any option | D6, D6-R |
| 3 D9 | Five-minute flag documented, including that it doesn't stop the watermark. Four-effect policy question. No new threshold | D9-S |
| 3 D11 | Value for all five named operations under each alternative. Two-field alternative defines activity vs evidence time | D11 |
| 3 D12 | `investigation_payload` included. Lists why D5 doesn't settle D12 | D12 |
| 4 V1 | V1.3 is conditional on D6-R and D6. V1.4 uses an injected clock | V1 |
| 4 V2 | `M`, `P` and `B` defined, with the expected relationship per decision. V2.3 tests each operation | V2 |
| 4 V3 | V3.2 reports two counts. V3.5 is conditional on D5 and D6. V3.1 has explicit timestamps and window boundaries | V3 |
| 4 V4 | V4.2 clarified and made conditional on D4 | V4 |
| 4 V5 | Grouping and persisted fields asserted separately. Mode labels added. B3 explicitly no evidence for V5.5 or V5.7 | V5 |
| 4 V6 | Failure modes T, M, S defined. Collision test added. Applicability table by architecture | V6 |
| 4 V7 | New and duplicate UNKNOWN kept separate. No-fabrication criterion unconditional | V7 |
| 4 V8 | Accepted alert defined. Six-step order set against what the code does. V8.5 boundary table | V8 |
| 4 V9 | V9.1 decisions widened. V9.5 uses identical inputs. V9.6 separates defect from documented dependency. V9.3 kept | V9 |

## 4. Corrections beyond the brief

These were contradictions or errors in the drafts.

| # | Draft problem | Correction |
|---|---|---|
| 1 | The draft said invariants don't depend on any open policy, yet I3, I4, I5 and I7 conflict with listed alternatives | Each invariant now states what it depends on |
| 2 | Section D.1 proposed "one defined `last_seen`", which pre-selected D11b | Now "Open, D11" |
| 3 | Matrix rows 13 and 14a stated expected outcomes ahead of D3, D6, D10, D11 | Now point to those decisions |
| 4 | Row 15 said buffer-full attribution differs from the buffered path, without qualification | It runs only with no pending record, so its first write matches what a new record would retain. Only later repeats differ |
| 5 | V9.2 required no cross-campaign inversions, which conflicts with D5a and D5d | Conditional on D5 |
| 6 | V7.4 didn't account for D3-C4 | Conditional on D2 and D3-C |
| 7 | V9.5 expected the same `closed_at` for all paths, though the DB sweep reads a different value under D11a, and a score close follows another rule | Scoped accordingly |
| 8 | Register alternative D12a said fixing D5 would fix all four fields | Removed. D12 lists why it doesn't |
| 9 | "30 of 101" sat beside close counts of 96 and 11 with no explanation | 101 counts campaigns by last close (90 + 11). 96 and 11 count writes |

## 5. New findings, documented and not fixed

All are `SOURCE` and unexercised unless noted.

- **Queue-full alerts can be lost silently.** The id is marked seen before queueing, so the retry is dropped as a raw duplicate.
- **`restore()` on a collision** moves counters to the newer record's event and campaign, and overwrites its time and payload with older values.
- **Shutdown doesn't wait for the worker.** `stop()` joins for 2 s, the worker sleeps 7 s, and it is a daemon thread.
- **`alert_queue.join()` has no timeout.** Failed alerts are marked done without retry, and their offset has already advanced.
- **The `future_skew` flag isn't visible in live operation:** neither persisted nor logged.
- **`risk_score` has two meanings:** persisted running sum vs memory chain sum.
- **A client error after a server commit** would double counters. `HYPOTHESIS`.

## 6. Contradictions that remain, for the owner

The documents state these. They can't be resolved without a decision.

| # | Issue | Needed from the owner |
|---|---|---|
| 1 | D6a can't coexist with "replay determinism required" | D6-R, then D6 |
| 2 | D3a can't coexist with C1 or C2. D3b with D12a needs an extra rule, since `last_seen` isn't written | Decide D3, D3-C and D12 together |
| 3 | D11b with duplicates extending and a buffer kept: do decisions see pending activity? | State this with D11 |
| 4 | I5 can't hold under D5d or D6a, or I3 under D1d or D2c | Accept the invariant or the alternative |

**Claims kept with a qualification, for owner review.** They couldn't be confirmed from available evidence:
- Per-batch atomicity is assumed, not tested.
- `stop()` raising `NameError` has not been run.
- Readers of `Campaign.closed_at` and the risk consumers were found by text search only.
- Which `risk_score` each consumer receives wasn't traced.

**Outside these documents.** The drafts keep their old wording. `research/` files elsewhere weren't changed.

## 7. Consistency checks

Run by a read-only script over the four review copies and this report. Results are as of the final edit.

| Check | Result |
|---|---|
| Every D reference, sub-decision and alternative letter resolves | Pass |
| Every V reference resolves. V1 to V9 all present | Pass, 51 criteria |
| All 46 draft criterion IDs preserved | Pass. Added: V3.6, V5.9, V6.5, V6.6, V8.6 |
| Each criterion's decision tags match the dependency table | Pass. The first run found 53 mismatches, all corrected by rebuilding the table from the tags |
| No review copy cites a superseded file as current | Pass. The contract names its draft once, as superseded |
| Required figures and hash present | Pass |
| Wording review for decisions presented as approved | Pass: 4 hits, all stating non-approval, a separate-task recommendation or a generic reference to a future policy |

**Read by hand, not scripted:** agreement between matrix, invariants, register and criteria; association vs mutation; lifecycle vs evidence time; and whether each label is justified. A script can't judge these, so a reviewer should read them too.

**Evidence labels.** No `PREDICTED` or `HYPOTHESIS` statement from the drafts was upgraded. New statements are in `EVIDENCE_TRACEABILITY_FINAL.md`, N1 to N22.

## 8. Change log

| File | Change |
|---|---|
| `CAMPAIGN_ACTIVITY_CONTRACT_FINAL_DRAFT.md` | Patched copy of the draft. Terms added; invariants restructured with applicability matrix; matrix rows 3 to 7, 13, 14a, 15 reworded; C clarified; D.1 un-selected; D.3 explained; E gains four modes and a mode column; F split into F.1 to F.3; G and H updated |
| `EVIDENCE_TRACEABILITY_FINAL.md` | Patched copy. Rows M14, B7, B9, E1 amended; four gaps and rows N1 to N22 added. All other rows unchanged |
| `OPEN_POLICY_DECISIONS_FINAL.md` | Rewritten from the draft, alternatives and letters kept. Terms and dependency table added; D3-C, D6-R, D7-O, D9-S, D12-F added; D5, D6, D10, D11, D12 restructured |
| `VALIDATION_CRITERIA_FINAL.md` | Rewritten from the draft, all IDs kept. Dependency column and conditional tables added; five criteria added; V3.1 scenario and failure modes specified |
| `DOCUMENTATION_RECONCILIATION_REPORT.md` | New |
