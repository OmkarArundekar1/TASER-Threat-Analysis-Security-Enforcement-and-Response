# Open decisions and blockers

Planning only. Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Numbering:** `ODn` is new in this phase. `Dn` and `Nn` are the coordination register in `phase_2_6a_decisions/DECISION_REGISTER_D1_D15.md`. `CA-Dn` is the campaign-activity list in `campaign_activity_contract/OPEN_POLICY_DECISIONS_FINAL.md`.

**Each recommendation is a proposal, not an approval.**

## What's settled, and not reopened

| Item | Decision | Where it's used |
|---|---|---|
| `NOT_COMMITTED` retry | Reuses the record, through `authorize_retry` | Path A |
| `FOUND_NOT_COMMITTED` retry | New linked intent | Path B |
| Later epochs | Need `authorized_by` and `reason` | Path B |
| Attempt numbers | Assigned by the ledger | Step 6 |
| Marker fields and rules | Your amended D6 | `MARKER_AND_RECOVERY_DESIGN.md` |
| `TransactionTimedOutClientConfiguration` at COMMIT | Stays `AMBIGUOUS` | Classification |

## Decisions needed before implementation starts

Only three block the first work package, and each has a safe recommended answer.

| ID | Decision | Recommendation | If you choose otherwise | Blocks |
|---|---|---|---|---|
| **D5** | Who authorizes a retry | **A person only,** for now | A policy needs a validated classifier | WP-C1 |
| **D14** | No identity available | **Refused by the coordinator, counted.** The caller takes the existing unprotected path during rollout | Refusing outright could lose detection data | WP-C1 |
| **D15** | Stores unavailable | **Hold, don't discard,** using the backlog. Details in OD-4 | [F] A counted loss breaks your Phase 2.3 rule | WP-C1, with fakes |

## Decisions needed before the throughput work

| ID | Decision | Options | Recommendation | Evidence |
|---|---|---|---|---|
| **OD-1** | Approve **Approach A,** with the offset saved **after the durable spool** (a fourth option for CA-D7-O) | A; B; keep today's memory queue | **A** | [F] Closes the 1,000-alert crash gap. **It reverses D10's "not for the pilot".** Needs your approval |
| **OD-2** | Tolerated detection delay, and the drain-rate target | 30 min to page, and 50 a second, are placeholders | **Decide after B1** | [U] `process_alert` is unmeasured |
| **OD-3** | Alert interrupted mid-processing | a. Held for a person. b. Re-run automatically. c. Treated as lost, as today | **a, until a route is fully coordinated.** Then b for that route | [F] Most writers aren't idempotent. Touches CA-D7, CA-D7-O |
| **OD-4** | During an outage, does the whole worker hold, or only coordinated routes? | Whole worker; coordinated only | **Whole worker.** Today's unmigrated writers swallow failures, so holding prevents loss | **Ingestion to the graph stops for the outage,** and detection is delayed |
| **OD-5** | Rotation-aware reader, with a new sidecar state file | Add; leave the gaps | **Add.** `offset.dat` keeps its format | [F] 110 rotations in nine months. Changes a 2.6C decision |
| **OD-9** | Per-alert INFO logging in the reader | Lower to DEBUG; keep | **Lower,** with counters | [F] About a third of reader time. Changes what operators see |
| **OD-10** | Location of the backlog, journal and ledger | One explicit local directory | **Yours to name.** Local, not `/mnt/c` | [U] The production host's disks aren't known |
| **OD-11** | Same source key, different content | Hold; process both; process first only | **Hold, with an alarm** | [F] 0 in 681,259. It's a visible hold of a valid, reused identity |

## Decisions needed before the marker work

| ID | Decision | Recommendation | Evidence |
|---|---|---|---|
| **OD-8** | Compare-and-set on `investigation_payload`. Adds `payload_event_time`, and changes which payload a late alert leaves | **Approve, after X2.** Replaces N3's "accept during the pilot" | Interacts with CA-D8 and CA-D12-F5 |
| **OD-6** | Keep backlog keys as long as markers | **Keep.** Revisit with B6's growth figures | Layer 1 of D13's treatment depends on it |
| **N1** | `COMMITTED_NO_EFFECT` | **`COMPLETED`, with an error log** | [F] As recommended in 2.6A |
| **N2** | Re-delivery across operations | **Replaced by this design's layers** rather than accepted | [H] Tests still to come |

## Decisions that can wait

| ID | Decision | Waits for | Recommendation |
|---|---|---|---|
| **OD-7** | Automatic `FOUND_COMMITTED` from a matching marker, under a named policy | Real volume of uncertain requests | Not yet |
| **OD-12** | Additive ledger transition `NOT_COMPLETED` to `ABANDONED` | WP-R1 | **Add it.** [F] A conflict otherwise has no terminal state. Changes a completed component |
| **OD-13** | When the tolerant ID policy is implemented | Before any live run | Implement with WP-T2. [F] Accepted in 2.6E. Listener code still has the strict pattern |
| **D7** | Derived ids | W06 | Deferred |
| **D8** | Batch identity | After W06 | Deferred. W08 stays unprotected, and that's a known double-count risk |
| **D9** | Sweeps ledgered? | | No: helper only |
| **D11** | Which tiers | Benchmark | **Decide with B2.** It may be what makes OD-2 achievable |
| **D12** | Retention | Growth data | Keep everything |
| **CA-D1 to CA-D12** | Campaign activity | Their own process | **This design decides none.** Each writer keeps today's semantics, with the exception OD-8 flags |

## Blockers

| # | Blocker | Blocks | What clears it | Status |
|---|---|---|---|---|
| **BL-1** | **`process_alert` latency has never been measured,** so no drain rate can be promised | OD-2, final capacity numbers | **B1** on a disposable graph | **Open.** The design proceeds on placeholders, and they're labelled as such |
| **BL-2** | **Real arrival rate is unknown.** 11,566 is event time | Final sizing | **X3,** and it needs read authorization | **Open.** The backlog tolerates uncertainty, by design |
| **BL-3** | **A lock wait under `db.lock.acquisition.timeout = 0s` isn't proven to end at the transaction timeout** | Worker capacity, and Phase 2.8 | **X1** | **Open.** If it fails, the design needs another limit |
| **BL-4** | **The CAS's evaluation order is untested** | OD-8, W07 and W08 templates | **X2** | **Open.** Alternative below |
| **BL-5** | **Marker behaviour at production scale is untested.** 2.6D used a disposable server | Any production constraint | **X4,** with a copy you authorize | **Open** |
| **BL-6** | **Backend read paths with markers present have never run** | Rollout | X4 | **Open.** [F] Static scans found nothing that reaches them |
| **BL-7** | **Coordinated writes may not meet any useful drain rate.** [H] An illustrative 20 to 40 a second | Choice of tiers, and possibly B | B2 | **Open.** Alternatives listed in the throughput document |

**D13's cross-operation problem isn't a blocker for coordinator work,** because layers 1 and 2 close the mechanism in design. It does stay unproven until T-16 runs. **Stale payload is a blocker only for W07U,** through BL-4.

### If an experiment fails

| Fails | Alternative | Cost |
|---|---|---|
| **X1:** lock waits outlast the transaction timeout | A client-side watchdog that closes the session; or AC-9 metadata and an operator kill. **Changing the server setting needs your separate approval,** and isn't assumed | The worker can stall longer |
| **X2:** the CAS doesn't behave | Read the time first in one `CALL` subquery; or keep N3's acceptance during the pilot, with a warning on retry | Slower, or accepted risk |
| **X4:** constraint creation harms a live writer | A maintenance window | Downtime you'd schedule |
| **B2:** far below target | D11 narrows the tiers; or Approach B for W07 and W08, after D8 | Coverage, or complexity |
| **X7:** spool admission can't keep up | Larger batches. The file still holds the rest, so **this means delay, not loss,** subject to rotation | Time |

## Experiments

**None was run in this phase.** Each needs authorization when it's reached.

| ID | Question | Environment | Touches production? |
|---|---|---|---|
| **X1** | With a long-held lock from a legacy-style writer, or the flush thread, does a marked write end at its 8 s transaction timeout? And how is it classified? | Disposable Neo4j 5.26.27 | No |
| **X2** | Does `WITH ... AS newer` use the value from before the `SET`? On equal times? With a null? | Disposable | No |
| **X3** | Real arrival rate: sample the size of `alerts.json` at intervals | Production file, **read only** | **Reads only,** with your authorization |
| **X4** | The rehearsal in `MARKER_AND_RECOVERY_DESIGN.md` | **A disposable copy of production's data** | Taking the copy needs authorization |
| **X5** | B1 to B6 | Disposable | No |
| **X6** | Per-second counts for every archive day | Archive, read only | **Reads only,** with your authorization |
| **X7** | Spool admission rate, and fsync cost per batch | Local, no Neo4j | No |
| **X8** | Cost of a source-prefix marker lookup at size | Disposable | No |
| **X9** | When and how rotated files appear, and read permissions | Production, **read only** | **Reads only,** with your authorization |

## Assumptions

| # | Assumption | If wrong |
|---|---|---|
| A1 | Event-time order is near arrival order | [F] 0 late alerts in the sample. If wrong, CAS matters more |
| A2 | One day's sample shape stands in for the model | X6 corrects it |
| A3 | Key plus hash takes about 150 bytes a row | B6 corrects it |
| A4 | The rotated file's path follows the archive naming from 2.6E | X9 corrects it. [F] Seen in the archive listing |
| A5 | The driver and server stay at 6.2.0 and 5.26.27 | Startup check refuses marker mode until re-validated |
| A6 | One listener, one reader, one manager | [F] One value over nine months. A second needs a new design for the backlog |
