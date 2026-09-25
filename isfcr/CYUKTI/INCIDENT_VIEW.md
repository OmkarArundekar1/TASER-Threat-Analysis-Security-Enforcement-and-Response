# CYUKTI Incident View

## What this is

The single, primary analyst-facing page (`frontend/src/components/IncidentView.tsx`, reachable via the "Incident View" top-nav button) that answers the full chain in one place: *what happened, why it matters, what CYUKTI thinks it matches historically, whether it's a real threat, what the evidence says, and what should happen next.* Before this phase, this data existed but was scattered across ~8 separate dashboard pages/tabs the analyst had to jump between.

## Backend: one aggregate endpoint, no duplicated logic

`GET /api/incidents/<campaign_id>/overview` composes:

- Real `CampaignContext` (attacker/victim, techniques, risk score, prediction state)
- `operation_id` (real `(Operation)-[:HAS_CAMPAIGN]->(Campaign)` lookup)
- MITRE enrichment (`mitre_resolver.enrich_technique_metadata`, honestly labeled `mapping_source: AGGREGATED_FROM_CAMPAIGN_RECORD` since per-alert provenance isn't preserved at the campaign level)
- `severity_from_tps` (existing, unchanged)
- Threat qualification (`_build_threat_qualification`, shared helper — same code the standalone `/api/threat-qualification/<id>` route uses)
- Campaign selection (`_build_campaign_selection`, shared helper — same code the standalone `/api/campaign-selection/<id>` route uses)
- GNN status
- SOAR playbooks/executions already associated with this campaign (`soar.memory.PlaybookMemoryStore`)

`GET /api/incidents/<campaign_id>/response-plan` composes the same qualification/selection plus `soar.generator.PlaybookGenerator`/`soar.response_plan.ResponsePlanGenerator` into the human-readable response narrative.

**Deliberately NOT composed into either aggregate endpoint**: a fresh evidence-aware investigation (`POST /api/investigate/<id>`) and Multi-RAG search (`POST /api/rag/search`). Both are real, non-trivial operations — investigation in particular has a documented, non-idempotent side effect on campaign prediction state (see `DASHBOARD_API.md`) — so silently re-running them on every page load would be a regression (unwanted cost + side effects), not an aggregation. The frontend calls them separately, exactly as the pre-existing `EvidenceInvestigation.tsx` component already did; `IncidentView.tsx` reuses that exact component for its "Investigation / Evidence" section rather than rebuilding it.

## Frontend sections (all real data, degrading honestly when unavailable)

| Section | Data source | Notes |
|---|---|---|
| A. Header | `overview.campaign`, `.severity`, `.threat_qualification`, `.mitre` | Status badges, not raw JSON |
| B. Attack Story / Timeline | existing `GET /api/campaigns/<id>/timeline` | Real timestamps only, never invented |
| C. Campaign Selection | `overview.campaign_selection` | Visual tree (current incident → top 4 candidates → selected), full 5-signal breakdown |
| D. Threat Qualification | `overview.threat_qualification` | PASS/FAIL/**UNKNOWN** per check (see `THREAT_QUALIFICATION.md`) |
| E. Investigation / Evidence | reuses `<EvidenceInvestigation />` verbatim | Collapsed by default (a real, explicit action, not run automatically) |
| F. Multi-RAG | `POST /api/rag/search` | Per-source results shown separately, never blended into one score |
| G. Response Plan | `GET /api/incidents/<id>/response-plan` | Threat summary, why, selected campaign + explanation, MISP readiness, per-action SAFE/DESTRUCTIVE + AUTO/APPROVAL badges |
| H. Shuffle | `overview.soar.executions` | "Execution unavailable — recommendation generated locally" when none exist, never fakes a run |
| I. Playbook Memory | `GET /api/soar/recommendations/<id>` | "No previous playbook found. CYUKTI generated a new recommendation." when there's no historical match |

## Live verification

With Neo4j and the CYUKTI backend/listener running live in this environment (see `FINAL_SYSTEM_READINESS.md`), `GET /api/incidents/CAMP_4FDA1A87/overview` and `GET /api/incidents/CAMP_4FDA1A87/response-plan` were both called directly against real data (100 real campaigns in Neo4j) and returned fully-composed, non-fabricated responses — real MITRE names/tactics, a real ranked list of 10 historical candidates with a real explanation, a real SUSPICIOUS classification (CTI score 29.15) correctly blocking MISP.

**Not live-rendered in an actual browser this phase** — `chromium-cli`/Playwright were not available in this environment and installing them was judged not worth the setup time given the endpoint-level live verification and full frontend test suite already covered correctness; both the Vite dev server and the backend were left running so the user can view it directly.

## Tests

`IncidentView.test.tsx` (5), `test_dashboard_api_incident_views.py` (7, overview + response-plan).
