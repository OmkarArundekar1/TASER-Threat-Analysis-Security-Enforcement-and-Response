# CYUKTI Review Pack

Generated 2026-08-31 as a repository-level audit and capstone review evidence pack. Every quantitative claim in this pack traces to a fresh test run, a live Neo4j/Wazuh query, or a specific repository file — see each document's inline source citations. No metric was invented; where evidence didn't exist, it's marked `NOT MEASURED / NOT AVAILABLE`.

## How to use this pack

**Preparing the architecture section of your review** → `01_architecture.md`, `02_architecture_detailed.md`, `cyukti_architecture.mmd`, `cyukti_detailed_architecture.mmd`. Diagram 1 (`cyukti_architecture.mmd`) for the high-level system slide; Diagram 2 (`cyukti_detailed_architecture.mmd`) for the detailed data-flow slide — this second one is the strongest single diagram for demonstrating the core contribution (see `10_limitations_and_future_work.md`'s final verdict).

**Preparing results/metrics slides** → `04_results_and_metrics.md` (test suite, Neo4j stats, dataset stats, ML in-sample evaluation, explicit `NOT MEASURED` list) and `05_phase20_results.md` (MITRE resolution coverage, live UNKNOWN-path verification, the disclosed coverage-number discrepancy, T1548.003, the 100500/100501 status).

**Preparing the module-completeness section** → `03_module_status.md` — every module classified A (complete+validated) through F (blocked), with evidence for each.

**Preparing the presentation itself** → `07_review_slides.md` (12-slide structure with bullets, metrics, and one line to say per slide) and `09_presenter_script.md` (30-second / 2-minute / 5-minute versions).

**Preparing for viva/reviewer questions** → `08_reviewer_questions.md` (20 likely questions with defensible answers) and `06_research_contribution.md` (what is and isn't a defensible novelty claim, plus the detection-paradox explanation for the hardest conceptual question).

**Preparing the "what's left" / honesty section** → `10_limitations_and_future_work.md` — includes the two live, disclosed, unfixed issues found during this audit, the dataset's own `NOT_READY_FOR_CALIBRATION` verdict, an explicit "what must NOT be claimed" list, and the final maturity classification (**Research Prototype**) with justification.

## File index

| File | Purpose |
|---|---|
| `01_architecture.md` | High-level system architecture, plain-text + component table |
| `02_architecture_detailed.md` | Full annotated data-flow, step by step |
| `03_module_status.md` | Module-by-module A-F completeness table |
| `04_results_and_metrics.md` | Every extractable metric, with source and caveats |
| `05_phase20_results.md` | Phase 20 MITRE resolution deep-dive, live-verified |
| `06_research_contribution.md` | Contribution strength assessment + detection-paradox explanation |
| `07_review_slides.md` | 12-slide suggested deck |
| `08_reviewer_questions.md` | 20 Q&A pairs |
| `09_presenter_script.md` | 30s/2min/5min scripts |
| `10_limitations_and_future_work.md` | Known issues, what not to claim, final verdict |
| `cyukti_architecture.mmd` | Mermaid: high-level architecture |
| `cyukti_detailed_architecture.mmd` | Mermaid: detailed data-flow architecture |

## Ground rules this pack followed

No invented numbers. No estimates presented as measurements. Implementation status is distinguished from tested status, from live-validated status, from benchmarked status — these are not synonyms and are labeled separately throughout. Failed/incomplete/unresolved items are disclosed, not hidden. The codebase, Neo4j, datasets, and live services were not modified while producing this pack.
