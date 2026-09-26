# Frontend Testing

Status before this session: no test framework, no test files — the
React frontend had zero automated coverage while the backend/API layer
had 336 behaviorally-verified tests. This documents what this session
built. **No claim is made about UI/UX quality, visual correctness, or
performance** — this is behavioral/contract verification only.

## Stack

- **Vitest 3.2.7** — the natural fit for this Vite 5 project (shares
  `vite.config.ts`, no separate bundler/config to maintain).
- **React Testing Library 16.3.3** (React-19-compatible) +
  `@testing-library/jest-dom` for DOM matchers + `@testing-library/user-event`
  (installed; most interactions here use `fireEvent`, which is
  sufficient for the click/type/keydown patterns actually exercised).
- **jsdom** as the DOM environment.
- **@vitest/coverage-v8** (pinned to `3.2.7`, matching vitest's own
  version — installing the version npm resolved by default, `5.0.1`,
  silently mismatched vitest's major version and crashed at runtime
  under this environment's Node 18 with `ERR_UNKNOWN_BUILTIN_MODULE`;
  fixed by installing the matching version explicitly).

No new UI framework, state-management library, or browser automation
tool was introduced. `react-force-graph-2d` (the existing graph
library) is mocked at the module boundary in tests — see "Graph
component" below.

## Environment note: how to actually run npm here

This repository lives on a WSL filesystem accessed by this session's
Windows shell via a UNC path (`\\wsl.localhost\...`). Windows' `npm`
cannot install here: package postinstall scripts (`esbuild`'s in
particular) spawn `cmd.exe`, which refuses a UNC working directory
("UNC paths are not supported. Defaulting to Windows directory."),
and `npm`'s own `.bin` symlink handling over this filesystem produces
`EPERM`/`EISDIR` errors independent of that. **Fix**: run `npm`/`node`
natively inside WSL instead (`wsl.exe -- bash -lc "cd
/home/.../frontend && npm install"`) — genuine Linux node/npm on a
genuine Linux path, no UNC/cmd.exe involved. All commands below assume
this; adjust if running from an actual WSL/Linux shell directly (drop
the `wsl.exe -- bash -lc "cd ... && "` wrapper).

## Running tests

```bash
cd frontend
npm test                # vitest run -- one-shot, CI-style
npm run test:watch      # vitest -- watch mode
npm run test:coverage   # vitest run --coverage
```

## Typecheck / build

```bash
npx tsc -b        # project-wide typecheck (also run automatically by `npm run build`)
npm run build     # tsc -b && vite build -- the real production build
```

Both were run this session and pass cleanly against every change made
(including the new test files, which are covered by the same
`tsconfig.app.json` `noUnusedLocals`/`noUnusedParameters` strictness
as production code).

## Test architecture

Two tiers, per the mission for this phase:

1. **API service layer** (`src/services/api.test.ts`) — the real
   exported `api.*` functions, with only `fetch` (the HTTP boundary)
   mocked. Verifies method/URL/body construction and, critically, that
   HTTP failures (400/404/500/503, malformed JSON bodies, network
   rejection) become real thrown `Error`s — never a silently-resolved
   success.
2. **Component tests** — the real component, rendered with React
   Testing Library, asserting on visible output (text, roles, test
   ids on the one necessarily-mocked graph library) rather than
   internal state or mock call counts alone. Two sub-patterns:
   - Most components (`EvidenceInvestigation`, `ThreatActorAttribution`,
     `ThreatCorrelation`, `CampaignIntelligence`, `AttackGraph`,
     `PredictionPanel`) mock `useDashboard()` (the context) and the
     `api` module (one level above raw `fetch`) with realistic fixture
     data, since these components don't own that state themselves.
   - `src/context/DashboardContext.test.tsx` is the **integration**
     tier the mission specifically asked for: the REAL
     `DashboardProvider`, the REAL `api.ts`, with only `fetch` mocked
     — proving `refreshAll()`'s actual `Promise.allSettled` resilience
     logic (partial-failure vs. total-failure vs. success) end to end,
     not just as read from the source.

Realistic fixtures live in `src/test/fixtures.ts`, built from the
actual backend response shapes this project's backend sessions
verified (`../backend/DASHBOARD_API.md`, `../backend/ML_NBE_INTEGRATION.md`,
`../backend/THREAT_ATTRIBUTION.md`) — not invented. Reused across every
test file that needs an investigation/prediction/campaign/attribution
response.

## Defects found and fixed this session

**1. `EvidenceInvestigation.tsx` silently dropped real backend fields.**
Reading the component against the real `InvestigationResult`/
`SeverityPrediction` contracts (confirmed via `investigation/loop.py`'s
`InvestigationRecord.to_dict()` and `ml/train_xgboost.py`'s `predict()`)
showed it never rendered `why_selected` (the NBE decision rationale),
`candidate_hypotheses` (the full competing-hypothesis distribution),
`derived_from` (evidence provenance), `top_k`, or `model_metadata` —
despite `types/index.ts` already declaring them (fixed in the prior
backend/API session) and the real API already returning them. This is
real decision-transparency data CYUKTI's adaptive investigation loop
computes specifically to be inspectable, invisible to any user of the
UI. **Fixed**: `EvidenceCard` now shows a `derived_from` provenance
note when present; each step now shows its `why_selected` rationale and
`candidate_hypotheses` distribution; the severity prediction card now
shows `top_k` and `model_metadata.feature_schema_version`. Additive
only — no existing element removed, no layout restructured. Regression
coverage: `EvidenceInvestigation.test.tsx`.

**2. `@vitest/coverage-v8` version mismatch** — see "Stack" above.
Not a CYUKTI application defect, an environment/tooling one, fixed the
same session it was found.

No component crashed on empty/error/malformed data during testing;
every empty/loading/error state already present in the source behaved
as its own code says it should.

## Known dead code found (documented, not wired in — a layout decision out of this session's scope)

- **`PredictionPanel.tsx`** is never imported by `App.tsx` or any
  other component — `predictions` state is fetched
  (`DashboardContext.refreshAll`) but nothing in the actual render
  tree displays it via this panel. Tested here in isolation anyway
  (`PredictionPanel.test.tsx`) since it's real, non-trivial,
  reachable-if-mounted logic (a confidence gauge, an empty state) —
  not wired into `App.tsx`'s layout this session, since doing so is a
  layout/product decision, not a test-infrastructure fix.
  *Planned:* wire `PredictionPanel` into `App.tsx`'s layout in a future
  product/design pass, once its placement is decided.
- **`hooks/useWebSocket.ts`** is never imported anywhere in `src/`.
  Not tested (no consumer to test it through, and adding one would
  again be a feature decision, not infrastructure).
  *Planned:* add a test once a real consumer wires this hook in.

## What is covered

- API service layer: `health`, `overview`, `events`, `campaigns`,
  `investigate`, `predictSeverity`, `predict`, `query` — success,
  parameter encoding, and every documented failure mode.
- `DashboardContext`: loading → success, total failure, **partial**
  failure (the real `Promise.allSettled` resilience contract), and a
  real re-fetch triggered by `selectCampaign`.
- Components: `EvidenceInvestigation` (investigation run, severity
  prediction, MITRE semantic search — success/error/empty for all
  three), `ThreatActorAttribution`, `ThreatCorrelation`,
  `CampaignIntelligence` (list + detail + selection + timeline),
  `AttackGraph` (loading/empty/populated + node/link counts through
  the merge logic), `PredictionPanel`, `PanelWrapper` (collapse,
  fullscreen, Escape).

## What is intentionally not covered

- **Visual/pixel rendering of the force-directed graph.**
  `react-force-graph-2d` is mocked at the module boundary — its canvas
  physics engine is neither meaningfully testable under jsdom nor
  something this phase's mission asked for ("do NOT write brittle
  tests based on ... internal implementation details"). What IS tested
  is `AttackGraph.tsx`'s own logic: state branching and the
  node/link merge-and-dedupe it performs before handing data to the
  library.
- **`AttackPathAnalytics`, `AttackerIntelligence`, `IntelligenceWorkspace`
  (the tab-switcher shell), `LiveEventsFeed`, `MitreAttackChain`,
  `PathExplorer`, `QueryConsole`, `RecommendationEngine`,
  `RiskPropagation`, `SecurityOverview`, `TopNavBar`** were not tested
  this session — proportioning effort to the mission's explicitly
  named highest-value areas (campaign, investigation, prediction,
  attribution, graph, loading/error/empty states, the API service
  layer) rather than exhaustively covering all 17 components in one
  pass. `RiskPropagation`'s and `QueryConsole`'s backend endpoints
  (`/api/risk/propagation`, `/api/query`) are already behaviorally
  verified at the API layer in `backend/tests/test_dashboard_api_routes.py`.
- **Real browser rendering / visual regression.** No Playwright,
  Cypress, or similar was introduced (none existed before, and the
  mission was explicit not to add one "unless genuinely necessary").

## Live integration verification

**HTTP-BOUNDARY VERIFIED; LIVE BROWSER EXECUTION UNAVAILABLE.**

This environment has no browser automation tooling, so the React app
was never actually rendered in a real browser against a live backend
this session — that claim is not made. What WAS verified for real:
the actual `dashboard_api.py` Flask server was started against this
environment's live Neo4j instance, and hit directly via `curl` using
the exact request shapes `api.ts` constructs (`GET /api/overview?time_range=24h`,
`GET /api/campaigns`, `GET /api/graph?view_mode=campaign`,
`GET /api/correlation/campaigns/<real_id>`,
`GET /api/attribution/actors/<real_id>` — the query this session's
prior backend work fixed, confirmed returning a clean `200` against
real data rather than the syntax error it previously raised — and
`POST /api/investigate/<real_id>`, which returned a real investigation
result with real MITRE evidence and real production-model
probabilities). Every response matched the shape
`src/test/fixtures.ts` and `src/types/index.ts` declare. This
establishes the HTTP contract end to end for real; it does not
establish that the React rendering pipeline consumes it correctly in
an actual browser DOM, which is what the component/integration test
suite above verifies instead (in jsdom, not a browser).

## Remaining limitations

- Real browser end-to-end testing was not performed (see above) —
  genuinely unavailable in this environment without introducing new
  tooling, which was out of scope.
  *Planned:* introduce a browser automation tool (e.g. Playwright) for
  real end-to-end coverage once it is judged genuinely necessary.
- `@types/react` (`^18.2.14`) and `@types/react-dom` (`^18.2.3`) are
  pinned to React 18 types despite the project using React 19 —
  pre-existing (not introduced this session), not touched, since it
  did not cause any actual typecheck or test failure. Worth revisiting
  if a future React-19-specific API is needed.
  *Planned:* upgrade these type packages to their React-19-compatible
  versions once a React-19-specific API is actually needed.
- `npm audit` reports pre-existing transitive-dependency
  vulnerabilities (unrelated to this session's additions) — a
  dependency-hygiene task, out of scope for test-infrastructure work.
  *Planned:* address these in a dedicated dependency-hygiene pass.
