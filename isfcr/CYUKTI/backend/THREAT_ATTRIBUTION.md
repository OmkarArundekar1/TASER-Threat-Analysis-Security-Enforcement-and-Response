# Threat Attribution

Status before this session: module_status.md rated this **C** — "code
present; `attribution_collector.py` has 2 tests as part of the
evidence-collector suite... no attribution-accuracy dataset exists...
NOT MEASURED."

This documents the attribution contract as it actually exists in
`threat_attribution_engine.py` / `attribution_context.py` /
`attribution_similarity.py`, and what this session verified. **No
accuracy claim is made** — this is behavioral/engineering verification
(does the implementation do what its own algorithm says it does), not
a validation of whether its attributions are correct.

## Two independent attribution mechanisms exist — this document covers one

CYUKTI has two separate things both called "attribution":

1. **`ThreatAttributionEngine.attribute()`** (this document) —
   **campaign-to-campaign similarity**. Candidates are *past CYUKTI
   campaigns*; `campaign.campaign_id` is used directly as the `actor`
   identifier. "Attribution" here means "this looks like a previously
   seen campaign," not "this matches a named threat-actor group."
   Wired into the evidence-aware investigation loop
   (`InvestigationAction.ATTRIBUTION_MATCH`) and MISP event generation.
   Never persisted to Neo4j.

2. **`neo4j_client.update_actor_attribution()`** /
   `dashboard_api.py`'s `/api/attribution/actors/<campaign_id>`
   — matches a campaign's techniques against real
   `(:ThreatActor)-[:USES]->(:Technique)` graph nodes (e.g. imported
   STIX threat-actor data). This one **does** persist (`MERGE
   (c)-[r:RESEMBLES]->(ta)`, gated at `confidence >= 50.0`). The two
   Cypher queries in `update_actor_attribution` and
   `attribution_actors` implement the same shared/total-technique-count
   formula independently (a real, minor duplication — not fixed this
   session; both are internally consistent, and unifying them is a
   Neo4j-side refactor outside this session's scope).

These are **deliberately not unified** here — they serve different
purposes (real-time per-alert graph annotation vs. on-demand
campaign-similarity for investigations/MISP) and unifying them would be
a real architectural change, not a stabilization fix. Mechanism 2 could
not be behaviorally tested in this environment (its logic is Cypher
`WHERE`/`MERGE` clauses that only execute against a live Neo4j
instance); this is a genuine environment limitation, not a deferred
validation choice.

## The attribution contract (mechanism 1)

**Evidence pool**: `attribution_context.load_historical_campaigns()`
returns only campaigns with `status IN ['INACTIVE','ARCHIVED']` — the
currently-active campaign under investigation can never attribute
against itself or another still-open campaign. Query is deterministically
ordered (`ORDER BY c.campaign_id, e.first_seen`); each candidate's
`techniques` list is deduplicated preserving first-occurrence order.

**Scoring**, per candidate:

```
coverage   = |observed ∩ historical| / |observed|      (asymmetric — denominator is the CURRENT campaign)
precision  = |observed ∩ historical| / |historical|
chain_sim  = LCS(current_chain, historical.techniques) / len(current_chain)
total_score = round((coverage*0.50 + precision*0.20 + chain_sim*0.30) * 100, 2)
```

Note: this `chain_similarity` formula (divide by `len(current_chain)`)
is **not the same** as `operation_feature_engine.py`'s
`chain_similarity` (divide by `max(m, n)`) — the two engines were
written independently for different purposes and their chain scores
are not comparable.

**Insufficient evidence**: a candidate with `similarity == 0` is
**dropped entirely**, not returned as a manufactured "0% confidence"
entry. The engine's insufficient-evidence signal is a shorter (or
empty) `actors` list — never a fabricated confident guess. Verified:
`test_no_historical_campaigns_returns_empty_actors_not_fabricated`,
`test_zero_technique_overlap_is_excluded_not_a_low_confidence_candidate`,
`test_current_campaign_with_no_techniques_yields_no_attribution`.

**Multiple candidates**: up to `TOP_K = 5`, sorted descending by
`total_score`. Ties preserve input order (Python's `sort()` is stable),
which composes with `load_historical_campaigns()`'s own deterministic
`ORDER BY` into a fully deterministic overall ranking. Verified:
`test_more_than_top_k_candidates_truncates_deterministically`,
`test_tied_candidates_preserve_deterministic_input_order`.

**Conflicting signals**: there's no explicit "conflict resolution"
step in this engine — competing candidates are simply ranked by the
weighted formula above (verified against hand-computed expected scores
in `test_competing_candidates_are_ranked_by_the_real_weighted_score_not_arbitrarily`,
proving a high-precision small match outranks a larger, noisier one).
Evidence-level conflict detection (disagreeing sources on the *same*
underlying fact) is a separate, existing mechanism in
`investigation/confidence.py` (`CONFLICT_PENALTY`, compounds per
unresolved conflict) — this engine doesn't need its own copy of that,
since its output becomes ordinary ranked Evidence for that mechanism to
operate on.

## Evidence provenance (no second provenance system)

`evidence/collectors/attribution_collector.py` wraps **every** ranked
candidate (not just the top one) as `Evidence`
(`EvidenceSource.ATTRIBUTION`), with `relevance` decayed by rank and
`relationships=[actor.actor]`. This reuses the existing
`Evidence.derived_from` / `ActionMeta.depends_on` mechanism unchanged
— `investigation/loop.py`'s `_tag_derived_from` already tags
`ATTRIBUTION_MATCH` evidence with the `evidence_id`s of any
`CAMPAIGN_HISTORY` evidence already in the store (declared via
`ActionMeta.depends_on = {CAMPAIGN_HISTORY}`, since attribution's
historical-campaign data is a real, code-verified subset of what
campaign-history evidence already covers). This session added
`test_attribution_evidence_derives_from_campaign_history_evidence_with_real_candidates`,
which proves this using the REAL engine's multi-candidate output rather
than the single-item hand-built fixture the original Phase 22 test
used. No new provenance mechanism was introduced.

## Persistence

Confirmed structurally
(`test_attribute_engine_never_writes_to_neo4j`): `attribute()` never
references `driver.session`, `MERGE`, `CREATE`, or `.run(` — it is
read-only and ephemeral by design. Its result is consumed in-process
only (see Propagation below).

## Propagation

`dashboard_api.py` does **not** expose `ThreatAttributionEngine`
directly — `/api/attribution/actors/<campaign_id>` queries mechanism 2
(the ThreatActor-node one) instead. This engine's real downstream
consumers, both verified this session with real objects:

1. **MISP event generation** (`misp_event_generator.py`) — the
   top-ranked candidate's actor/score/coverage/precision/chain-similarity
   and rationale text become MISP attributes and tags
   (`actor:<id>`, `attribution:<score>`). Verified:
   `test_attribution_result_propagates_into_misp_event_generation`.
2. **The evidence-aware investigation trace**
   (`/api/investigate/<campaign_id>`) — every candidate becomes
   Evidence, reaches `InvestigationRecord.to_dict()`, and is
   JSON-serializable end to end. Verified:
   `test_attribution_result_propagates_into_investigation_trace_via_evidence`.

## Tests

`backend/tests/test_threat_attribution.py` — 15 tests covering the
positive path, insufficient evidence, conflicting/competing candidates,
top-k truncation and tie-breaking, campaign-history wiring (including a
structural check that the query excludes ACTIVE campaigns), chain-order
sensitivity, persistence (absence), and both propagation paths.

Run: `cd backend && python -m pytest tests/test_threat_attribution.py -q`

The full realistic scenario (operation correlation → attribution →
investigation state → MISP propagation, all real engines, only Neo4j
mocked) is `tests/test_operation_attribution_integration.py`.
