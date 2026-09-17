# Multi-RAG

Status before this session: `rag/retriever.py`'s own docstring already
declared the design intent — "This is the 'Multi-RAG' retrieval
mechanism... the same class can index CTI/MISP event descriptions or
historical-campaign narratives" — but exactly **one** concrete source
existed (`rag/mitre_retriever.py`, over the MITRE ATT&CK STIX corpus).
The generic engine was real, tested, and integrated; the "multi" part
of Multi-RAG was not. This session added a second real source.

## What was added

`rag/campaign_retriever.py` — `CampaignNarrativeRetriever`, a second
concrete instance of the existing `rag.retriever.SemanticRetriever`
(unchanged), indexed over real historical campaign records
(`attribution_context.py`'s `HistoricalCampaign` — the same real data
`threat_attribution_engine.py` and
`evidence/collectors/campaign_history_collector.py` already consume).
Document text is a deterministic formatting of real fields (attacker,
victim, technique IDs) — nothing fabricated, same pattern
`mitre_retriever.py` already uses for real STIX fields.

**Why this and not a placeholder source**: it reuses 100% existing,
already-tested infrastructure (`SemanticRetriever`), needs zero
model/research decisions (same TF-IDF mechanism, just a second real
corpus), and answers a genuinely different question than every
existing retrieval path over the same data — see "Not a duplicate"
below.

## Not a duplicate of existing campaign-history retrieval

`InvestigationAction.CAMPAIGN_HISTORY` (existing, unchanged) returns
**every** historical campaign sharing at least one technique with the
current one — exhaustive, structured, set-overlap. `CAMPAIGN_NARRATIVE_SEARCH`
(new) answers "which past campaigns best match this free-text
description of observed behavior" — semantic, ranked, free-text. This
is exactly the same complementary relationship `mitre_retriever.py`
already has with `mitre_mapper.py`'s exact keyword lookup (stated in
that module's own docstring) — not a new pattern, the second instance
of an existing one.

## Integration

Added as a ninth entry in the evidence-aware investigation action menu
(`investigation/actions.py`: `CAMPAIGN_NARRATIVE_SEARCH`), following
the exact precedent `MITRE_SEMANTIC_SEARCH` already set for pairing a
semantic-search action with an exhaustive one: it shares
`EvidenceSource.CAMPAIGN_HISTORY` with the `CAMPAIGN_HISTORY` action,
so the existing same-source novelty discount already prevents
double-crediting redundant retrieval — no new mechanism, no
`depends_on` declaration needed (matches the reasoning already
documented for `MITRE_SEMANTIC_SEARCH` in that file).

Wired into `investigation/loop.py`'s `default_action_executor`: query
text is built from the current campaign's own `attack_chain` (falling
back to `techniques`, then the current technique id) — the same
query-construction pattern `MITRE_SEMANTIC_SEARCH` already uses.
Degrades to an empty evidence list (not a crash) when no historical
campaign with a resolved technique exists yet to search — a real,
expected state early in a deployment's life.

**On the investigation action menu growing from 8 to 9 actions**: this
does not modify `next_best_evidence.py`'s scoring formula, weights, or
any other action's metadata — it adds one more candidate to the
existing competition. It does **not** touch, contradict, or "fix" the
frozen Phase 21/22 findings (`review/phase21_real_investigation_validation.md`,
`review/phase22_nbe_sensitivity_validation.md`), which describe the
8-action menu's specific, historical, already-recorded behavior on 3
specific real campaigns — those documents and their JSON data are
unmodified. A future re-run of `scripts/run_real_investigations.py`
would now reflect 9 actions; that re-run was **not** performed this
session (per this phase's explicit scope: no NBE ablations, no
re-validation), and no claim is made here about how the new action
affects ranking adaptivity, novelty, or any other Phase 21/22 metric.

## Verification

Behavioral: `tests/test_rag.py` (4 new tests — controlled-error on
empty corpus, exclusion of campaigns with no resolved techniques,
ranking, full Evidence contract) and `tests/test_default_wiring.py`
(2 new tests — real wiring, graceful degradation).

Live: run directly against this environment's real Neo4j instance
(`CampaignNarrativeRetriever().query("T1595 T1110 credential access
reconnaissance")`) returned 3 real, ranked historical campaigns with
real relevance scores. A full real investigation
(`run_investigation` against real campaign `CAMP_7331E223`, real
`default_action_executor`, `max_steps=9`) naturally selected all 9
actions including `campaign_narrative_search`, in a sensible NBE
order (last, since `CAMPAIGN_HISTORY` — sharing its EvidenceSource —
had already run), with zero errors.

## Tests

```bash
cd backend
python -m pytest tests/test_rag.py tests/test_default_wiring.py -q
python -m pytest tests/ -q   # full suite, 345 tests
```
