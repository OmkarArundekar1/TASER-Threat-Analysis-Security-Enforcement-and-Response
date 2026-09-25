# CYUKTI — Recommended Figures (from existing data only)

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date.

Each figure below uses only numbers already established in `quantitative_results.md`. None require new data collection.

## Figure 1 — MITRE resolution provenance distribution
- **Data source**: `mitre_coverage_report.py` output, 2026-09-25 snapshot (post rule-fix/reboot; was the 2026-08-31 snapshot below).
- **Chart type**: stacked/horizontal bar, single bar split into segments.
- **Categories (current)**: NATIVE_WAZUH (26), REVIEWED_RULE_MAPPING (0), DETERMINISTIC_INFERENCE (0), UNKNOWN (94), out of 120 total. (Earlier snapshot, 2026-08-31, 391 total: NATIVE_WAZUH 39, UNKNOWN 352 — kept as a secondary bar/footnote to show the split is traffic-dependent, not fixed.)
- **X-axis**: count (or %). **Y-axis**: N/A (single bar) or "alert" if using a simple 2-bar chart.
- **Demonstrates**: the real, measured attribution-coverage split.
- **Must NOT imply**: that 90% UNKNOWN is a failure rate or that CYUKTI "only detects 10% of attacks" — UNKNOWN is an attribution outcome, not a detection outcome, and must be labeled as such directly on the figure.

## Figure 2 — Graph-integrity repair (before/after)
- **Data source**: `campaign_reconstruction.py` repair record + live `check_integrity()` result.
- **Chart type**: simple 2-bar before/after.
- **Categories**: "Before repair" (44 orphaned events), "Current" (0 orphaned events).
- **X-axis**: state (before/after). **Y-axis**: orphaned AttackEvent count.
- **Demonstrates**: a real, durable data-integrity fix.
- **Must NOT imply**: that orphaning can never recur — this is a point-in-time repair result plus a currently-passing integrity check, not a permanent guarantee.

## Figure 3 — Current graph scale
- **Data source**: live Neo4j query, 2026-09-25 (was 2026-09-12).
- **Chart type**: simple bar chart, 3 bars.
- **Categories**: Campaigns (111), AttackEvents (212), Technique nodes (858).
- **X-axis**: node type. **Y-axis**: count.
- **Demonstrates**: real accumulated scale, and that the ATT&CK knowledge base (858) dwarfs the observed campaign data (111) — an honest visual of "we have a large reference corpus, modest real attack data."
- **Must NOT imply**: that 858 techniques were all observed in real attacks — the vast majority are reference metadata from the vendored STIX corpus, not campaign-derived.

## Figure 4 — Dataset severity distribution (limitation figure, not a strength figure)
- **Data source**: `campaign_dataset.csv`, 60 rows.
- **Chart type**: bar chart, 4 bars.
- **Categories**: Low (53), Medium (3), Critical (4), High (0).
- **X-axis**: severity class. **Y-axis**: campaign count.
- **Demonstrates**: severe class imbalance and a structurally empty High class — this figure should be framed as evidence *for* the `NOT_READY_FOR_CALIBRATION` verdict, not as a dataset-quality showcase.
- **Must NOT imply**: that this distribution reflects real-world attack severity base rates — it reflects a small, non-representative lab dataset from 3 attacker identities.

## Figure 5 — XGBoost confusion matrix
- **Data source**: `evaluate_model.py` run, 2026-08-31, n=60.
- **Chart type**: 3x3 heatmap/confusion matrix.
- **Categories (both axes)**: Critical, Low, Medium (in that order, matching the evaluation output).
- **Values**: `[[1,2,1],[0,51,2],[0,0,3]]`.
- **Demonstrates**: where the model's in-sample errors concentrate — specifically, that Critical is the weak class (only 1/4 correctly classified) despite strong aggregate accuracy.
- **Must NOT imply**: generalization performance — the title/caption must state "in-sample, n=60, no held-out data" directly on the figure, not only in surrounding text.

## Figure 6 — NEXT_TECHNIQUE evaluable predictions (recommend a plain statement, not a chart)
- **Data source**: Phase 18 dataset rebuild.
- **Recommendation**: **do not chart this.** n=12 is too small to support a meaningful visual (a pie/bar chart of 4 vs 8 would look like it's presenting a real distribution when it's a single small offline experiment). Present as plain text: "4/12 correct (33.3%) — labeled `INSUFFICIENT_FOR_SUPERVISED_ML`."
- **Demonstrates (as text, not a chart)**: the pipeline can be evaluated at all, and the project correctly self-identifies the result as statistically insufficient.
- **Must NOT imply**: that a chart of 12 data points represents a validated accuracy rate.

## Figure 7 (optional) — Test suite composition
- **Data source**: `pytest -q -v` breakdown by file, 2026-09-14 re-run (historical — the file-by-file counts below have not been re-tallied since; the current total is 689 backend across 65 files, up from the ~150 these per-file counts summed to).
- **Chart type**: horizontal bar, one bar per test file, sorted descending.
- **Categories**: test file names (investigation=25, evidence=13, mitre_resolver=12, gnn=11, next_technique_pipeline=9, etc.) — if redrawing this figure for the paper, re-run `pytest -q -v` for a current per-file breakdown rather than reusing these historical counts as-is.
- **X-axis**: test count. **Y-axis**: module/test file.
- **Demonstrates**: test coverage is spread across the real breadth of the system, not concentrated in one trivial module.
- **Must NOT imply**: that test count is a proxy for correctness or coverage completeness — it demonstrates tested behavior, not absence of undiscovered defects (see Slide 5's defensible answer in `review_slides.md`).
