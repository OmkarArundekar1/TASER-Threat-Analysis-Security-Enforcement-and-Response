# CYUKTI — Presenter Script (30s / 2min / 5min)

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date.

## 30 seconds — elevator pitch

"CYUKTI ingests real Wazuh security telemetry into a Neo4j attack-campaign graph and tries to attribute each alert to a MITRE ATT&CK technique. Most real alerts don't come with a native ATT&CK tag — so instead of discarding them, we built a provenance-tracked resolver with four precedence levels, ending in an explicit UNKNOWN state, so we never fabricate an attribution but never lose the evidence either. We verified live that UNKNOWN events can't contaminate the attack-chain learning or risk scoring."

## 2 minutes — architecture + contribution + results

"CYUKTI's pipeline runs: Wazuh agent → Wazuh manager → our listener → alert parsing → a MITRE resolver → Neo4j. The resolver is the core contribution: it checks, in order, whether Wazuh's own rule already carries a native ATT&CK mapping, then a manually-reviewed rule registry we maintain, then deterministic structural inference — and only falls back to an explicit UNKNOWN state if none of those can defensibly produce a technique. Every result carries its provenance and an ordinal confidence, never a fabricated probability.

Downstream, resolved events flow through attack-chain learning, a prediction engine, campaign correlation, and an evidence-aware investigation loop that separates evidence reliability, evidence coverage, model confidence, and model uncertainty as distinct signals — so the system can't manufacture confidence just because evidence is missing.

We validated this live, not just in tests: right now, of 120 real alerts processed (the current snapshot, taken after a recent rule fix and manager restart), 21.7% resolve to a native ATT&CK technique and 78.3% are correctly preserved as UNKNOWN — and we directly inspected five of those UNKNOWN events in Neo4j to confirm zero fabricated technique IDs, zero contamination of the learned attack-chain graph, and zero TPS contribution. Our own dataset-validity audit also found the accumulated real dataset — 60 campaigns — isn't yet large or diverse enough for severity calibration or next-technique prediction, and we report that honestly rather than overstating results on 60 rows."

## 5 minutes — full narrative

**Problem.** Real-world SOC telemetry from tools like Wazuh mostly arrives without a MITRE ATT&CK tag. A pipeline that requires attribution before ingestion discards most of its own evidence. A pipeline that fabricates attribution to avoid that corrupts everything downstream that trusts the technique label as ground truth — attack-chain learning, risk scoring, and any ML trained on it.

**Architecture.** CYUKTI ingests Wazuh alerts through a Python listener into a Neo4j knowledge graph. The central objects are Campaign, AttackEvent, and Technique, linked by real relationships (HAS_EVENT, LAUNCHED, TARGETS, MATCHES, NEXT_TECHNIQUE). The ATT&CK knowledge base itself is a vendored, versioned STIX snapshot (Enterprise v19.1, 858 real techniques) imported directly into Neo4j — no runtime internet dependency, and every proposed technique is validated against it (rejecting revoked or deprecated IDs).

**Implementation.** The centerpiece is `mitre_resolver.py`: a four-tier precedence — native Wazuh mapping, a manually-reviewed rule registry (currently empty by design, populated only when independently justifiable), deterministic structural inference (currently zero live rules — infrastructure only), and UNKNOWN. Resolved events get the full pipeline: attack-chain learning, next-technique prediction, campaign correlation, threat attribution, MISP publication. UNKNOWN events are preserved as evidence — attached to the right campaign, full raw alert retained — but structurally barred from touching any of those, verified by both code inspection and live data.

**Metrics.** 798 of 798 tests pass (689 backend across 65 files, plus 109 frontend across 17 files, including a new 83-test SOAR/playbook layer). Live: 111 campaigns, 212 attack events (2,539 total nodes, 20,804 total relationships), 858 real techniques, only 3 learned attack-chain transitions. Coverage today: 21.7% native, 78.3% UNKNOWN, out of 120 real alerts — and we show, not just claim, that none of that leaked into technique learning. Our XGBoost severity model reaches 0.917 accuracy, but only in-sample on the 60 rows it was trained on — we don't have a held-out split yet, and we say so explicitly. End-to-end latency is now measured too: p50 35.5ms, p95 66.9ms on a real HTTP round trip.

**Research contribution.** Not the individual algorithms — graph databases, TF-IDF, gradient boosting are all standard. The contribution is the system-level design: separating detection from attribution from investigation from prediction, with mandatory provenance on every attribution decision and a structurally-enforced explicit-UNKNOWN state that cannot contaminate learning. We can prove this holds, live, not just assert it.

**Limitations.** The accumulated dataset — 60 real campaigns from 3 attacker identities — is not diverse enough for severity calibration or supervised next-technique prediction, and our own validity-gate analysis says so explicitly. There's a known, disclosed, non-blocking bug in a maintenance thread we found yesterday and have deliberately not yet fixed. A duplicate Wazuh local-rule-ID configuration issue remains formally unresolved pending root access.

**Future work.** Deliberate dataset expansion across more attacker/victim diversity — not just more volume — before revisiting calibration; a real held-out ML evaluation; fixing the two disclosed open issues; and only then reconsidering the modules we've explicitly marked as not-yet-viable."
