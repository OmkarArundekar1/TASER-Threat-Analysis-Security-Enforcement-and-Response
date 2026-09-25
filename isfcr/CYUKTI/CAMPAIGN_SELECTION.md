# CYUKTI Best Campaign Selection

## Why this exists

GNN topology retrieval (`rag/gnn_topology_retriever.py`) already returns a top-K ranked list of structurally-similar historical campaigns. Two things that list alone doesn't give an analyst: (1) other real signals besides topology (technique overlap, timing, identity), each kept separate rather than blended by GNN, and (2) a single, internally-decided "best match" with a plain-language explanation, while still letting the analyst inspect every alternative that was considered.

**Critical rule enforced**: GNN topology similarity is never the sole basis for a security decision. It is one of five explicit, independently-weighted, independently-reported signals.

## The five signals (`campaign_selection.py`)

| Signal | Source | Formula |
|---|---|---|
| `topology_similarity` | `ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns` (existing, unchanged) | Cosine similarity of GNN embeddings. `None` (not 0) when GNN is disabled/unavailable. |
| `technique_similarity` | Real `CampaignContext.techniques` sets (both campaigns) | Jaccard overlap — shared reused from `soar.matcher._technique_jaccard`, not duplicated. |
| `temporal_similarity` | Real `first_seen` timestamps | Linear decay over a fixed, documented 1-week scale: same instant → 1.0, ≥7 days apart → 0.0. |
| `attacker_similarity` | Real `attacker_ip` | 1.0 exact match, 0.5 same /24 subnet, 0.0 otherwise — a simple, explainable graded identity metric, not a learned similarity. |
| `host_similarity` | Real `victim_ip` | Same formula as attacker_similarity. |

A candidate missing a signal (e.g. GNN off) has that signal excluded from its composite score, with the remaining weights renormalized to sum to 1 — never penalized with a fabricated zero.

**Historical playbook success rate is reported per candidate but deliberately excluded from the composite similarity score.** "Is this the same/similar campaign" (identity/structure) and "did our past response to it work" (outcome quality) are different questions; folding them together would hide which one is actually driving a given selection.

## Composite score and confidence

Fixed, documented weights: topology 0.30, technique 0.25, temporal 0.20, attacker 0.15, host 0.10 (renormalized over whichever signals are actually available for that candidate).

`BestCampaignSelector.select()` ranks all candidates by composite score, picks the top one, and reports **every candidate**, not just the winner, as `ranked_candidates`/`alternatives`. Confidence is a fixed function of the score gap between the top two candidates (not a fabricated probability): gap ≥ 0.20 → `HIGH`, ≥ 0.08 → `MEDIUM`, else `LOW`; a single candidate is reported `HIGH` (nothing to be uncertain against).

## Explanation

Generated deterministically (no LLM call) from the selected candidate's own real signal values: names the strongest-contributing signal and, when more than one signal is available, the weakest one too — e.g. *"CAMP_A ranked highest (composite score 0.87) because its topology similarity (94%) matched the current campaign most closely, despite comparatively weaker attacker similarity (30%)."* Never fabricates a factor that wasn't actually computed.

## Candidate discovery (`GET /api/campaign-selection/<campaign_id>`)

Candidates are found via a real, **always-available** Cypher query — campaigns sharing at least one MITRE technique or the same attacker IP with the current campaign — capped at 10. This is deliberately **not** GNN-gated: GNN topology is one of five signals scored per candidate, not the source of the candidate list itself, so campaign selection still works with useful signal (technique/temporal/identity) even when `GNN_ENABLED=false`.

## Dashboard

`CampaignSelectionPanel.tsx`, a new sub-tab ("Selection") in the Intelligence Workspace: shows the confidence badge, the plain-language explanation, and every ranked candidate with its full five-signal breakdown and historical response success rate, the selected one visually marked.

## Selection tree visualization (added this phase)

`IncidentView.tsx`'s "Campaign Selection" section now renders the literal `CURRENT INCIDENT → candidates → SELECTED` relationship as a visual tree (top 4 ranked candidates as sibling nodes below the current incident, connecting lines, the selected one visually marked with an award icon and its full five-signal breakdown shown beneath), not just a ranked list. `CampaignSelectionPanel.tsx` (the earlier, more compact in-workspace version) is unchanged and still available as its own Intelligence Workspace tab for quick reference without opening the full Incident View.

Live-verified against real data (Neo4j running, 100 real campaigns): `GET /api/incidents/CAMP_4FDA1A87/overview` correctly ranked 10 real historical candidates, selected the top one, and produced a real, non-fabricated explanation string from actual signal values.

## Composed into the Incident View aggregate endpoint

`GET /api/incidents/<id>/overview` (`INCIDENT_VIEW.md`) calls the same `_build_campaign_selection()` helper the standalone `/api/campaign-selection/<id>` route uses — one implementation, not duplicated.

## Tests

`test_campaign_selection.py` (18 — pure signal functions, composite scoring, ranking, confidence, explanation generation, and the real-data integration path), plus the dashboard route tests in `test_dashboard_api_threat_and_selection_routes.py`.
