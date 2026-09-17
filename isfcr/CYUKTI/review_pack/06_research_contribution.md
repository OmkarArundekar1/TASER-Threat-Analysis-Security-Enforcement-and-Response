# CYUKTI — Research Contribution Assessment

## Contribution table

| Contribution | Implemented? | Evidence | Novelty Claim Strength |
|---|---|---|---|
| 1. Evidence-aware investigation | Yes | `investigation/` module: `confidence.py` separates evidence_reliability/coverage/model_confidence/model_uncertainty/investigation_confidence; 25 tests | **Moderate** — the specific separation of these 5 signals (rather than one conflated scalar) is a real, deliberate design fix documented as correcting an earlier bug (investigations falsely reaching confidence=1.0 after one evidence item); the general concept of evidence-weighted confidence is not new to security research |
| 2. Attack-chain modelling | Yes, but data-limited | `chain_updater.py`, live `NEXT_TECHNIQUE` graph | **Weak** — only 3 real learned edges exist; the mechanism is sound but has not been demonstrated at a scale that supports a novelty claim |
| 3. Campaign reconstruction | Yes | `campaign_reconstruction.py`, repaired 44 real orphaned events into 27 campaigns, idempotent, 10 tests | **Moderate** — a concrete, evidence-only (non-guessing) repair strategy for a real data-integrity defect found in production; the underlying idea (reconstruct from child evidence, never guess) is a defensible methodological contribution for this specific pipeline, not a general novel algorithm |
| 4. Next-technique prediction | Implemented, explicitly not viable yet | Full corrected label pipeline (Phase 18); own audit concluded `INSUFFICIENT_FOR_SUPERVISED_ML` | **Do not claim novelty** — the system correctly identifies its own data insufficiency rather than claiming a working predictor; this self-assessment rigor is worth presenting, the prediction accuracy itself is not |
| 5. Threat attribution | Implemented, unvalidated | `threat_attribution_engine.py`, evidence-collector tests only | **Weak** — no ground-truth attribution dataset exists to support an accuracy claim |
| 6. MISP enrichment | Implemented, unconfirmed live | `misp_sync.py`, `cti_publisher.py` | **Weak as a result claim** — the integration exists and initializes; whether it successfully publishes has not been confirmed this session |
| 7. Multi-source evidence/RAG | Yes | `evidence/orchestrator.py` (6 real collector types), `rag/mitre_retriever.py` (TF-IDF over real, vendored ATT&CK STIX corpus) | **Moderate** — genuinely multi-source (MITRE, graph, detection, CTI, campaign history, attribution), grounded in real STIX data rather than a toy corpus; TF-IDF retrieval itself is not novel |
| 8. Provenance-aware MITRE resolution | Yes | `mitre_resolver.py`, full precedence hierarchy, live-validated (Phase 20) | **Strong** — the explicit 4-tier precedence (native > reviewed > deterministic-inference > unknown) with mandatory provenance/confidence/reason on every result, validated against a real ATT&CK knowledge base, and specifically engineered to never let a lower-confidence tier override a higher one, is a concrete, demonstrable, correctly-implemented design — this is the strongest single piece of engineering produced in this project and the most defensible novelty claim, specifically *for this system's own problem* (Wazuh's incomplete native MITRE coverage) |
| 9. Explicit UNKNOWN state | Yes | Live-verified: 5 real UNKNOWN AttackEvents with zero fabricated technique IDs | **Strong** (paired with #10) — treating "no defensible mapping" as a valid, first-class, evidence-preserving outcome rather than silent discard is the core design decision and is fully implemented and live-demonstrated |
| 10. Prevention of attribution contamination | Yes | Live-verified: 0 fabricated `attack_id`, 0 unexpected `Technique` nodes, `NEXT_TECHNIQUE` count unchanged (3 before/after), structural control-flow proof that UNKNOWN can never reach `append_technique`/`chain_updater`/`predict_next`/MISP | **Strong** — this is verifiable both by static code inspection (control-flow proof) and by live graph evidence, not merely asserted |
| 11. Evidence-aware adaptive investigation | Yes | `next_best_evidence.py` selects the highest-value unexplored action; `stopping.py` requires confidence AND low uncertainty AND no conflicts before stopping | **Moderate** — a real, tested decision policy that explicitly refuses to manufacture confidence from absent evidence (verified via `test_uncertain_model_probabilities_do_not_falsely_report_near_certainty` and related tests); the specific policy is implemented correctly, but "adaptive evidence gathering" as a general concept is a known pattern in active-learning/investigation literature |
| 12. Detection paradox / detection-attribution distinction | Yes, conceptually and structurally | The entire Phase 20 architecture *is* this distinction, implemented | **Strong as a framing device, not as an algorithm** — see dedicated section below; this is the most presentable conceptual contribution, but it is a design principle correctly applied, not a novel algorithm |

## The "Detection Paradox" — technical explanation

**The paradox in one sentence**: a security pipeline that requires ATT&CK attribution as a precondition for ingestion will systematically discard the majority of its own telemetry, because most real alerts — even from a well-configured Wazuh deployment — do not carry a native technique mapping.

**Separating six distinct concepts that get conflated in naive pipelines**:

| Concept | What it actually answers | Where CYUKTI implements it |
|---|---|---|
| Detection | "Did a sensor observe something?" | Wazuh's rule engine — happens regardless of MITRE tagging |
| MITRE attribution | "Can we defensibly say *which* technique this was?" | `mitre_resolver.py` — a separate, explicit resolution step |
| Investigation | "What does this mean in context of a campaign?" | `investigation/` loop, evidence collectors |
| Evidence confidence | "How much do we trust what we know?" | `investigation/confidence.py` — reliability × coverage, or model confidence × (1-uncertainty) |
| Prediction | "What is likely to happen next?" | `prediction_engine.py`, gated by `NEXT_TECHNIQUE` learned counts |
| Response | "What should be done?" | `recommendation_engine.py`, MISP publication |

**Why "no MITRE mapping" ≠ "irrelevant"**: a systemd service crash, an Apparmor denial, or a disk-usage warning is *detected* — Wazuh fired a real rule on real system behavior. The absence of a technique tag reflects a gap in Wazuh's own out-of-the-box ATT&CK coverage for that specific rule, not a judgment that the event doesn't matter. Discarding it destroys evidence that might later matter (e.g., as context for a campaign, or for a human reviewing what actually happened on the host).

**Why fabricating a technique is worse than preserving UNKNOWN**: a fabricated attribution doesn't just fail to help — it actively corrupts every downstream consumer that trusts `attack_id` as ground truth: the `NEXT_TECHNIQUE` learned-transition graph, the campaign's `last_technique` (used to compute the *real* final-technique ground truth for the ML dataset per Phase 18), TPS/risk scoring, and any future attribution or prediction training data. A single mislabeled event silently becomes indistinguishable from a genuine observation. An `UNKNOWN` event, by contrast, is inert with respect to all of those — it changes nothing it shouldn't, and is honestly queryable later ("how much of our telemetry could we not attribute, and to what rules").

**Before Phase 20**: `mapped alert` → ingested, OR `unmapped alert` → discarded entirely (invisible to CYUKTI).

**After Phase 20**: `resolved evidence` (native/reviewed/inferred, with provenance) → full pipeline, OR `preserved unresolved evidence` (UNKNOWN, tps=0, no chain/prediction/MISP participation) → still a real, queryable `AttackEvent`, still attached to the right campaign, never contaminating attack-chain learning.

**30-second version for a faculty reviewer**: "Most real Wazuh alerts don't come with a MITRE ATT&CK tag out of the box. The naive approach is to only process alerts that do — which throws away most of your telemetry. Our approach instead resolves MITRE attribution as a separate, provenance-tracked step with an explicit UNKNOWN outcome, so we never guess a technique we can't defend, but we also never lose the underlying evidence. We verified live that unattributed events get zero contamination into the attack-chain learning or risk scoring — they're preserved, not faked."
