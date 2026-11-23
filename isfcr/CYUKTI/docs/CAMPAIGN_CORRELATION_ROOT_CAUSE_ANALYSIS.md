# Campaign Correlation Root Cause Analysis

**Investigation Date**: 2025-01-19  
**Status**: IN PROGRESS  
**Severity**: CRITICAL

---

## Problem Statement

CYUKTI is severely fragmenting single attack sessions into many separate campaigns:

| Session | Expected Campaigns | Actual Campaigns | Fragmentation Factor |
|---------|-------------------|------------------|----------------------|
| SCN-SESSION-BOUNDARY-001 | 1 | 4 | 4× |
| SCN-SESSION-BOUNDARY-002 | 1 | 25 | 25× |
| SCN-NMAP-001 | 1 | 17 | 17× |
| **Total** | **3** | **46** | **15.3× average** |

**Measured Metrics**:
- Pairwise Precision: 0.0
- Pairwise Recall: 0.0  
- Pairwise F1: 0.0
- Adjusted Rand Index: 0.0
- Cluster Purity: 1.0 (misleading under severe fragmentation)

---

## Root Cause Investigation

### Critical Bug #1: lateral_movement_similarity Not Used in Scoring

**Location**: `backend/campaign_decision_engine.py`

**Issue**: The `lateral_movement_similarity` feature was being calculated in `campaign_feature_engine.py` but NOT included in the decision engine's breakdown dictionary.

**Code Evidence**:

`campaign_feature_engine.py` line 318:
```python
return CampaignFeatures(
    transition_similarity=transition_sim,
    temporal_similarity=self.temporal_similarity(...),
    attacker_similarity=self.attacker_similarity(...),
    duplicate_similarity=self.duplicate_similarity(...),
    graph_similarity=self.graph_similarity(...),
    runtime_similarity=self.runtime_similarity(...),
    lateral_movement_similarity=lateral_sim  # <-- CALCULATED
)
```

`campaign_decision_engine.py` line 10 (BEFORE FIX):
```python
breakdown = {
    "transition_similarity": features.transition_similarity,
    "temporal_similarity": features.temporal_similarity,
    "attacker_similarity": features.attacker_similarity,
    "duplicate_similarity": features.duplicate_similarity,
    "graph_similarity": features.graph_similarity,
    "runtime_similarity": features.runtime_similarity
    # <-- lateral_movement_similarity MISSING!
}
```

`config.py` CAMPAIGN_WEIGHTS:
```python
CAMPAIGN_WEIGHTS = {
    "transition_similarity": 0.35,
    "temporal_similarity": 0.25,
    "attacker_similarity": 0.15,
    "lateral_movement_similarity": 0.15,  # <-- WEIGHT DEFINED
    "runtime_similarity": 0.05,
    "graph_similarity": 0.03,
    "duplicate_similarity": 0.02,
}
```

**Impact**: 
- The lateral_movement_similarity feature has a 0.15 weight (3rd highest weight)
- It was never being used in scoring decisions
- Multi-hop attack continuity detection was completely non-functional
- Campaigns were being split when attacker IP changed (even during lateral movement)

**Fix Applied**: Added `lateral_movement_similarity` to the breakdown dictionary in `campaign_decision_engine.py`

**Status**: ✅ FIXED

---

### Hypothesis #2: Premature Campaign Closure

**Active Campaign Threshold**: `ACTIVE_CONTINUE_THRESHOLD = 0.30`

**Theory**: Even with the lateral_movement bug fixed, campaigns may be closing prematurely if the correlation score drops below 0.30 when:
1. Temporal gap increases (but still within reasonable session boundaries)
2. Transition is unseen (despite 0.1 neutral default)
3. Multiple features unavailable simultaneously

**Evidence Needed**:
- [ ] Actual correlation scores at closure for the 46 fragmented campaigns
- [ ] Temporal gaps between successive events in fragmented campaigns
- [ ] Feature availability patterns during fragmentation
- [ ] Distribution of closure reasons (score vs. timeout expiry)

**Investigation Status**: PENDING (requires instrumentation)

---

### Hypothesis #3: Campaign Timeout Too Aggressive

**Current Timeout**: `CAMPAIGN_TIMEOUT` (value needs to be checked in config.py)

**Theory**: If the timeout is too short, campaigns may be expiring between bursts of related activity, causing the next event to create a new campaign instead of continuing.

**Evidence Needed**:
- [ ] Value of CAMPAIGN_TIMEOUT in config.py
- [ ] Temporal gap distribution between events in same real session
- [ ] Comparison of temporal gaps vs. CAMPAIGN_TIMEOUT

**Investigation Status**: PENDING

---

### Hypothesis #4: Fast Reopen Criteria Too Strict

**Fast Reopen Requirements** (from `campaign_manager.py` line ~486):
```python
if (
    first_technique == current_technique and
    temporal >= 0.8 and
    context.attacker_ip == attacker_ip and
    context.victim_ip == victim_ip
):
    # Fast reopen
```

**Theory**: Fast reopen only triggers if the FIRST technique of the closed campaign matches the CURRENT technique. For diverse attack chains, this is too restrictive.

**Example**:
- Campaign has chain: [T1595, T1210, T1059]
- Campaign closes
- Next event is T1055 (different from T1595)
- Fast reopen doesn't trigger even though it's clearly related

**Evidence Needed**:
- [ ] How many closed campaigns had first_technique != current_technique but should have reopened?
- [ ] Temporal proximity of rejected reopen candidates

**Investigation Status**: PENDING

---

### Hypothesis #5: General Reopen Scoring Too Conservative

After fast reopen check fails, the system evaluates ALL inactive campaigns for general reopening.

**Reopen Threshold**: `REOPEN_THRESHOLD = 0.70` (much higher than continue threshold of 0.30)

**Theory**: The 0.70 reopen threshold is too high. A campaign that closed at 0.29 should be a strong reopen candidate at 0.60, but current logic requires 0.70.

**Evidence Needed**:
- [ ] Distribution of reopen scores for inactive campaigns that should have been reopened
- [ ] Comparison of continue scores vs. reopen scores for same feature patterns

**Investigation Status**: PENDING

---

## Campaign Lifecycle Flow

```
NEW EVENT arrives
    ↓
Is there an ACTIVE campaign for (attacker, victim)?
    ↓
   YES → Evaluate correlation features
         Score ≥ 0.30? → CONTINUE (append to existing)
         Score < 0.30? → CLOSE, then check reopen...
    ↓
   NO → Check for INACTIVE campaigns
        ↓
       YES → Fast reopen check:
             - first_technique == current_technique?
             - temporal_similarity ≥ 0.8?
             - Same attacker/victim?
             → REOPEN if all true
             
             Otherwise: Score all inactive campaigns
             Best score ≥ 0.70? → REOPEN best candidate
             Best score < 0.70? → CREATE NEW
        ↓
       NO → CREATE NEW campaign
```

---

## Feature Weight Distribution

| Feature | Weight | Purpose |
|---------|--------|---------|
| transition_similarity | 0.35 | What comes next (prediction + chain) |
| temporal_similarity | 0.25 | Time proximity |
| attacker_similarity | 0.15 | Same attacker IP |
| **lateral_movement_similarity** | **0.15** | **Multi-hop continuity (WAS BROKEN)** |
| runtime_similarity | 0.05 | Event patterns |
| graph_similarity | 0.03 | Graph topology |
| duplicate_similarity | 0.02 | Duplicate detection |

---

## Transition Similarity Handling

**Unseen Transition Default**: 0.1 (neutral)

**Rationale**: "Never seen this transition" means "no information", not "evidence against"

**Code** (`campaign_feature_engine.py` line 73):
```python
UNSEEN_TRANSITION_DEFAULT = 0.1
```

This is correct - unseen transitions should not force fragmentation.

---

## Next Steps

### Immediate Actions

1. ✅ **Fix lateral_movement_similarity bug** - COMPLETED
2. ⬜ **Check CAMPAIGN_TIMEOUT value**
3. ⬜ **Instrument campaign_manager with detailed logging**:
   - Log correlation scores at every decision point
   - Log closure reasons (score vs. timeout)
   - Log reopen evaluation results
   - Log feature values at closure/reopen
4. ⬜ **Run evaluation with instrumentation** to collect decision traces
5. ⬜ **Analyze decision traces** to identify fragmentation patterns

### Diagnostic Tests Needed

1. **Same Session, Delayed Events**:
   - Events from same session separated by increasing temporal gaps
   - Expected: Continue until CAMPAIGN_TIMEOUT, then reopen if score ≥ 0.70
   
2. **Same Session, Diverse Techniques**:
   - Events from same session with varied technique sequences
   - Expected: Transition similarity should not block continuation
   
3. **Same Attacker/Victim, Different Sessions**:
   - Two distinct sessions with same (attacker, victim) pair
   - Expected: Should NOT merge different sessions
   
4. **Multi-Hop Lateral Movement**:
   - External → Internal → Internal attack chain
   - Expected: lateral_movement_similarity should detect continuity

5. **Campaign Timeout Edge Cases**:
   - Event arrives exactly at timeout boundary
   - Event arrives just after timeout
   - Expected: Deterministic closure and reopen behavior

### Potential Fixes (Priority Order)

1. ✅ **lateral_movement_similarity bug** - FIXED
2. ⬜ **Lower REOPEN_THRESHOLD** from 0.70 to 0.50 (investigate first)
3. ⬜ **Improve fast reopen criteria** - check last N techniques, not just first
4. ⬜ **Add session boundary hints** - if Wazuh provides session identifiers
5. ⬜ **Adjust CAMPAIGN_TIMEOUT** if too aggressive
6. ⬜ **Tune feature weights** based on decision trace analysis

---

## Testing Strategy

### Unit Tests (Targeted)

- [ ] Test lateral_movement_similarity is included in score calculation
- [ ] Test unseen transition returns 0.1
- [ ] Test temporal decay function
- [ ] Test campaign expiry at exact timeout
- [ ] Test fast reopen criteria
- [ ] Test general reopen scoring

### Integration Tests (Campaign Lifecycle)

- [ ] Single session, continuous events → 1 campaign
- [ ] Single session, delayed events → 1 campaign (if within timeout)
- [ ] Single session, diverse techniques → 1 campaign
- [ ] Different sessions, same attacker/victim → 2+ campaigns
- [ ] Multi-hop lateral movement → 1 campaign
- [ ] Campaign timeout → close then reopen on next event
- [ ] Replay determinism → same input → same campaigns

### Evaluation Tests (Real Data)

- [ ] Re-run campaign_correlation_eval.py with lateral_movement fix
- [ ] Measure improvement in fragmentation
- [ ] Verify no over-merging introduced
- [ ] Check edge cases: SCN-SESSION-BOUNDARY-002 (25 → ?)

---

## Success Criteria

### Minimum Acceptable

- Pairwise F1 ≥ 0.75 (up from 0.0)
- Pairwise Precision ≥ 0.90 (avoid false merges)
- Pairwise Recall ≥ 0.70 (reduce fragmentation)
- Average fragmentation factor ≤ 2.0 (down from 15.3)

### Ideal Target

- Pairwise F1 ≥ 0.85
- Pairwise Precision ≥ 0.95
- Pairwise Recall ≥ 0.80
- Average fragmentation factor ≤ 1.5

### Safety Constraints

- Different sessions with same (attacker, victim) must remain separate
- No false merges (precision is more important than recall)
- Deterministic replay behavior maintained
- All existing tests continue passing

---

## Open Questions

1. What is the actual value of `CAMPAIGN_TIMEOUT`?
2. What is the typical temporal gap distribution in real attack sessions?
3. Are there any session identifiers in Wazuh alerts we could leverage?
4. How does lateral_movement_tracker.is_likely_same_campaign() actually work?
5. What is the distribution of techniques per session in the real data?
6. How often do real sessions have the same first technique?

---

## Document History

- 2025-01-19 15:30 UTC: Initial root cause analysis created
- 2025-01-19 15:45 UTC: Critical bug #1 (lateral_movement_similarity) identified and fixed

---

**Next Update**: After instrumentation and decision trace collection
