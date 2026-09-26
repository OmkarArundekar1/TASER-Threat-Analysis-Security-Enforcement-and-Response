# Campaign Correlation / Operation Matching

Status before this session: module_status.md rated this **C** — "code
present, imported and called in `process_alert()`... no dedicated test
file found; live nodes exist but matching quality not evaluated."

This documents the matching contract as it actually exists in
`campaign_correlation_engine.py` / `operation_manager.py` /
`operation_feature_engine.py` / `operation_decision_engine.py`, and
what this session verified and fixed. **No accuracy claim is made** —
this is behavioral/engineering verification, not a validation of how
well the matcher identifies real operations.
*Planned:* build a labeled ground-truth operation-matching dataset so a
real matching-quality metric can be computed, alongside the planned
attribution-accuracy benchmark.

## What an "Operation" is

An **Operation** groups multiple **Campaigns** believed to belong to
the same actor over time — typically one attacker hitting several
victims. This is a separate, higher-level concept from a Campaign
itself (one attacker–victim pair's activity window, owned by
`campaign_manager.py` / `CAMPAIGN_TIMEOUT`, out of scope for this
document).

## The matching contract

`OperationFeatureEngine.extract_features()` computes 7 similarity
features between the campaign being processed and each candidate
operation; `OperationDecisionEngine.evaluate()` combines them with
fixed weights into a `score`, and attaches iff
`score >= ATTACH_THRESHOLD (0.70)`:

| Feature | Weight | How it's computed |
|---|---|---|
| `attacker_similarity` | 0.20 | exact match against the operation's "sticky" `primary_attacker` (set once, from the first campaign attached) |
| `victim_similarity` | 0.15 | membership in the operation's accumulated `victims` set |
| `technique_similarity` | 0.30 | Jaccard overlap of technique sets (highest weight — technique overlap dominates) |
| `temporal_similarity` | 0.20 | campaign's `first_seen` vs. operation's `last_seen`: ≤1h→1.0, ≤24h→0.8, ≤7d→0.5, else 0.0 |
| `chain_similarity` | 0.10 | LCS(campaign chain, operation chain) / max(len(campaign), len(operation)) |
| `prediction_similarity` | 0.05 | best next-technique-distribution overlap (computed but excluded from `IMPLEMENTED_FEATURES`) |
| `graph_similarity` | 0.00 | stub, always 0.0 |

**`confidence` is not a match-quality score.** It's a constant —
`len(IMPLEMENTED_FEATURES) / total_features * 100` (currently 5/7 =
71.4%, `config.py`) — a data-completeness percentage that does not vary
with how good any specific match is. `score` is the actual match
strength; `confidence` says nothing about a particular decision.
Verified: `test_confidence_is_a_constant_data_completeness_ratio_not_a_match_score`.

**`temporal_similarity`'s hour/day bands are independent of
`OPERATION_TIMEOUT`** (120 seconds, `config.py`), which governs a
completely separate mechanism: ACTIVE→INACTIVE *expiry*
(`operation_manager.expire_active_operations()`, run by the listener's
maintenance worker every 5s). In practice, an operation found via
`get_active_operations()` will almost always be well within the ≤1h
temporal band (since it can't have gone 120s without an update without
expiring first); the 24h/7-day bands mainly matter when matching
against `get_recent_inactive_operations()` (reopening a recently-closed
operation).

## Campaign separation (a provable safety property)

With the current weights, **attacker_similarity == victim_similarity
== 0 can never reach ATTACH_THRESHOLD**, even with every other feature
at its ceiling: `0.20 + 0.15 + 0.30 + 0.10 + 0.05 = 0.65 > sum without
identity = 0.65 < 0.70`. This is what actually prevents unrelated
attacker/victim pairs from merging into one operation — proven directly
in `test_attacker_and_victim_mismatch_can_never_reach_attach_threshold`,
which will fail if a future change to `weights` or `ATTACH_THRESHOLD`
breaks this invariant.

## Missing-data behavior (verified, not assumed)

Every feature extractor has an explicit empty/None guard — none raise:
empty attacker/victim strings degrade to 0.0 similarity; empty
technique sets on both sides return 0.0 (not a `0/0` error); `None`
timestamps return `temporal_similarity = 0.0`; a candidate operation
that can no longer be loaded (`build_operation_context` → `None`, e.g.
deleted between listing and lookup) is skipped, not fatal;
`correlate(None)` returns a controlled empty result.

## Fix made this session: nondeterministic candidate ordering

`neo4j_client.get_active_operations()` had no `ORDER BY`, unlike its
sibling `get_recent_inactive_operations()` (`ORDER BY o.last_seen
DESC`). Since `correlate()` only replaces its best candidate on a
**strict** `>` (never on a tie), the candidate order returned by Neo4j
determined which operation wins an exact score tie — and that order
was not guaranteed stable. Fixed by adding the same `ORDER BY
o.last_seen DESC` to `get_active_operations()`. Effect: ties now
deterministically favor the most-recently-active operation. Verified
structurally in `test_get_active_operations_query_orders_results_deterministically`
(a live, multi-node Neo4j instance would be needed to observe the raw
non-determinism directly, which this environment doesn't have —
*Planned:* re-run this observation once a live, multi-node Neo4j
instance is available) and
behaviorally in `test_correlate_breaks_exact_score_ties_by_first_candidate_in_list`
(documents the Python-side tie-break policy that ordering feeds into).

## Neo4j write shape (verified without live Neo4j)

`attach_campaign_to_operation` / `update_operation_activity`'s actual
Cypher parameters were verified against a fake `driver.session()` that
captures the query and params rather than executing them —
`test_attach_campaign_to_operation_sends_correct_parameters` /
`test_update_operation_activity_sends_correct_parameters`. The
`MERGE (o)-[r:HAS_CAMPAIGN]->(c)` relationship creation is idempotent
by construction (Cypher `MERGE` semantics), and `ON CREATE SET
o.campaign_count = coalesce(...) + 1` only increments on first
creation of that specific edge — repeated correlation of the same
campaign against the same operation does not inflate `campaign_count`.

## Known limitation (not a bug)

`operation_schema.py`'s `OPERATION_DEFAULTS` dict is imported into
`neo4j_client.py` but never actually referenced —
`create_operation_db` hardcodes its own defaults inline instead. Not a
behavioral bug (both produce the same values today), but a drift risk:
if `OPERATION_DEFAULTS` is edited expecting it to take effect, it
silently won't. Left as-is this session (out of scope — a genuinely
unrelated dead-import cleanup, not a correlation/attribution behavioral
issue), noted here for visibility.
*Planned:* have `create_operation_db` reference `OPERATION_DEFAULTS`
directly (or remove the unused import) in a future cleanup pass.

## Tests

`backend/tests/test_campaign_correlation.py` — 27 tests: same-operation
matching, different-operation rejection, temporal boundary bands (all
documented thresholds, parametrized), missing-data handling, a vanished
candidate, `None` input, idempotent repeated correlation, Neo4j write
parameter verification, the campaign-separation invariant, the
confidence-vs-score distinction, a realistic scenario using real MITRE
technique IDs, and the ordering-determinism fix.

Run: `cd backend && python -m pytest tests/test_campaign_correlation.py -q`

Integration with the wider system (an operation match feeding into the
investigation loop's evidence/confidence state) is covered by
`tests/test_operation_attribution_integration.py`.
