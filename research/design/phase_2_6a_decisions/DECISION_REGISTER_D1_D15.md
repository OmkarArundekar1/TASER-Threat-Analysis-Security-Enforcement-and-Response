# Decision register, D1 to D15

Phase 2.6A. **Every decision here is open. Each "recommended decision" is my recommendation, not an approval.** Phase 2.5's text is in `research/design/phase_2_5_coordination/PHASE_2_5_SCOPE_AND_FINDINGS.md`, which is unchanged.

| ID | Current proposal | New evidence | Recommended decision | Dependencies | Consequence of rejection | Approval required |
|---|---|---|---|---|---|---|
| **D1** Source identity | `alert["id"]`, with a content-hash fallback | `id` is event seconds plus file offset, with no manager part. Unique in the sample, with one manager, no restart, no rotation. A hash can't separate equal occurrences | **Changed:** `wazuh:<manager>/<id>`, versioned, no hash fallback. Adopt only after the production checks in `D1_SOURCE_IDENTITY_DECISION.md` | Production-shaped data. D14 for missing IDs | No writer can migrate | **Yes:** the format and dropping the fallback. Also a copy of production-shaped data |
| **D2** `alert_id` collision | Investigate, then include `id` | **Confirmed offline:** 70 of 9,015 refused with distinct IDs, 1 of them a modelled new event, all before any write | **Fix, as a separate change first.** Contract in `D2_ALERT_DEDUPLICATION_INVESTIGATION.md` | D1's identity. Independent of the coordinator | The listener keeps silently dropping distinct alerts, and the coordinator can't see them | **Yes:** a listener change that shifts live counts |
| **D3** Correlation | Journal fields, AC-1 and AC-3, plus AC-2 | Contracts written. Four rules added: required attempts, rise-only, store-enforced epoch chain, identical intent. 7,552 real payloads pass the serializer | **Approve AC-1, AC-2, AC-3** as in `D3_JOURNAL_LEDGER_API_CONTRACT.md` | None. It can start before D1 is settled, because it treats the key as an opaque string | No coordinator on these two stores | **Yes:** changes Phase 2.3 code, tests and schema |
| **D4** Retry after `RESOLVED` | New intent, linked by key | Holds in seven crash windows. Two state pairs were missing. A person's wrong "not committed" can't be detected without a marker | **Approve a new intent,** with the additions in `D4_RETRY_AFTER_RESOLUTION.md` | D3. D6 for evidence | Reverses your Phase 2.3 rule on `RESOLVED` | **Yes** |
| **D5** Who authorizes a retry | A person first, then a named policy | At the sample's rate W07U makes 11,197 requests a day. An outage could queue many | **Keep: a person only during the pilot.** Revisit with real counts | D3, D4 | With policy now: an unvalidated classifier could double-count | **Yes** |
| **D6** Marker | Marker plus constraint for Tier 1, resolved by hand | A separate statement gives false evidence, so it shares the `MATCH`. No `attempt` property. Absence isn't proof. About 11,197 markers a day for W07U. **Unproven on a real server** | **Approve in principle. Test on a disposable server,** then decide on production. Mandatory for the pilot. Details in `D6_APPLIED_WRITE_MARKER_DESIGN.md` | D1. An authorized test server. Production edition check | Uncertain additive writes can only be abandoned, and retry can't be validated | **Yes, twice:** the test, then the production schema |
| **D7** Derived ids | Derive from the key | W07U doesn't need it. W06 is the natural second writer and does. The key now includes the manager, which affects the derivation | **Defer until W06 is planned.** Not needed for the pilot | D1 | W01, W05, W06, W16 can't migrate | **Yes,** later: consumers see ids |
| **D8** Batch identity | Batch id persisted at snapshot | 1,201 modelled attributed duplicates feed this path, so **it carries the reproduced double count at volume.** No new identity evidence | **Unchanged, deferred.** It should be next after W06. Your ambiguous-flush decision applies | D1, D3, D4, D6 | The reproduced double count stays unprotected | **Yes,** before W08 |
| **D9** Sweeps ledgered? | No: helper only | Nothing new | **Unchanged:** helper only | None | A request every 5 s, for no gain | **Yes:** leaves two writers outside the contract |
| **D10** Crashed alert resumes? | No, for Phase 2.6 | **Corrected:** a crash can also re-deliver, since the offset is saved after the read loop. So repeats and losses both occur. Re-delivery across operations bypasses keys | **Unchanged for the pilot.** Add the corrected facts. Offset work needs its own design | D2 | Queue losses persist. **The coordinator doesn't fix them** | **Yes:** to accept that gap |
| **D11** Which tiers | Decide after measuring, starting from Tiers 1 and 2 | W07U's peak is 377 a second, and the cost per request is still unmeasured | **Unchanged,** and the W07U benchmark is the first measurement | Benchmark | Possibly unaffordable durability on Tier 3 | **Yes** |
| **D12** Retention | Keep, then archive. Never delete | Sized now: a W07U payload is 384 bytes at the median, and 11,197 requests a day. Markers add a graph-side question, kept separate in D6 | **Unchanged for stores. Keep everything during the pilot,** and measure growth | D6, pilot data | Deleting ledger rows makes old requests runnable again | **Yes** |
| **D13** First writer | W07 on the unattributed path | Stops before the Campaign, verified along the call path. Payloads deterministic in 7,552 of 7,552. **New:** busiest path; can't catch re-delivery across operations; retry may overwrite a newer payload | **Keep W07U,** with those three conditions. Details in `D13_FIRST_WRITER_SELECTION.md` | D1, D3, D4, D6, D14, D15 | Alternatives need D7 or D8 first | **Yes** |
| **D14** No key available | Unprotected fallback during rollout, then refuse | **Now more important:** with no hash fallback, any alert without valid identity reaches this. 0 such alerts in the real sample, but all 8 synthetic B3 alerts | **Keep:** unprotected with a counted warning during the pilot, then refuse. Fixtures declare their own kind | D1 | Refusing at once could lose detection data. Falling back forever leaves a silent gap | **Yes** |
| **D15** Stores unavailable | Stop, don't acknowledge | **A gap in the proposal:** once dequeued, an alert has no retry path, so simply stopping would discard its update, which your Phase 2.3 rule forbids. But a full queue delays rather than loses (`wazuh_listener.py` L274-280), so holding the alert is possible | **Keep the rule, and make it concrete:** the worker holds the alert and pauses until the stores return, so the queue fills and the reader stops advancing. A logged, counted loss would be the alternative, and it breaks your rule. Pause behaviour is untested, and a long pause could outlast log rotation | D3, D13 | An unprotected fallback silently removes protection | **Yes:** ingestion stops for as long as the stores are down |

## New items raised by this phase

These don't have D numbers yet. Each needs a ruling or an owner.

| Ref | Item | Recommendation | Approval |
|---|---|---|---|
| N1 | A committed write that found no target (`COMMITTED_NO_EFFECT`) | Record `COMPLETED`, and log an error | Yes, with D6 |
| N2 | Re-delivery across operations (a W06 creator returning as a W07U duplicate) | Accept during the pilot. Closed by migrating W06 | Yes |
| N3 | A retry overwriting a newer `investigation_payload` | Accept during the pilot | Yes |
| N4 | Replay harness bypasses the listener's collision, so offline and live paths differ | Note it in any comparison. Frozen evidence is unaffected | Your call on how to report it |
| N5 | `offset.dat` is written after the read loop, and an empty file skips to the end | Separate listener hardening, with D10 | Yes, if wanted |
| N6 | AC-9: transaction metadata on the helper | Not for the pilot | Later |

## Approvals needed before any implementation

| To begin | You need to approve |
|---|---|
| Journal API work | D3, D4 |
| Listener collision fix | D1's identity rule, D2 |
| Disposable-server marker tests | D6 test scope, and a server start |
| Production-shape validation | Providing the data |
| Coordinator | D3, D4, D5, D14, D15 |
| W07U migration | All of the above, plus D1, D6 production schema, D13, N1 to N3 |
