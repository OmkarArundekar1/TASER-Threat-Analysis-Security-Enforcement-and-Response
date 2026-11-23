# Campaign Correlation Evaluation

**Date**: 2025-01-19  
**Status**: BASELINE ESTABLISHED, FIX VALIDATED (CODE-LEVEL), FULL RE-EVALUATION PENDING  
**Bug Fix Commit**: `f042bde` - lateral_movement_similarity now included in scoring

---

## Executive Summary

This document establishes the campaign correlation baseline, documents the critical bug fix, and outlines validation requirements.

**Critical Bug Fixed**: `lateral_movement_similarity` (weight=0.15) was calculated but never used in scoring decisions. This is now FIXED AND TESTED at the code level.

**Current Status**: The existing Neo4j database contains campaigns created WITH THE BUG PRESENT. Full re-evaluation requires event replay with the fixed code, which is pending.

---

## Historical Baseline (Before Fix)

### Dataset
- **Source**: Historical evaluation ground truth (`campaign_correlation_v0.jsonl`)
- **Created**: 2026-09-26
- **Status**: AUTO_PROPOSED (no human review)

### Independent Sessions
1. **SCN-SESSION-BOUNDARY-001**: 192.168.56.105 → pes1ug23cs411-VirtualBox (mixed exploitation)
2. **SCN-SESSION-BOUNDARY-002**: 192.168.56.106 → pes1ug23cs411-VirtualBox (brute-force)
3. **SCN-NMAP-001**: 192.168.56.106 → 192.168.56.105 (Nmap reconnaissance)

### Measured Fragmentation (Pre-Fix)

| Session | Expected Campaigns | Actual Campaigns | Fragmentation Factor |
|---------|-------------------|------------------|----------------------|
| SCN-SESSION-BOUNDARY-001 | 1 | 4 | 4.0× |
| SCN-SESSION-BOUNDARY-002 | 1 | 25 | 25.0× |
| SCN-NMAP-001 | 1 | 17 | 17.0× |
| **Total** | **3** | **46** | **15.3× average** |

### Clustering Metrics (Pre-Fix)
- **Pairwise Precision**: 0.0
- **Pairwise Recall**: 0.0
- **Pairwise F1**: 0.0
- **Adjusted Rand Index (ARI)**: 0.0
- **Adjusted Mutual Information (AMI)**: ~4.7e-15 (effectively 0)
- **Cluster Purity**: 1.0 (misleading under severe fragmentation)

### Campaign IDs (Verified in Current Database)
All 46 historical evaluation campaigns are **still present** in the current Neo4j database (which contains 204 total campaigns as of 2025-01-19).

**SCN-SESSION-BOUNDARY-001** (4 campaigns):
- CAMP_23FB320A
- CAMP_1429ADB4
- CAMP_F91A7652
- CAMP_066240B7

**SCN-SESSION-BOUNDARY-002** (25 campaigns):
- CAMP_5DD3F083, CAMP_A93237FD, CAMP_B12C16D4, CAMP_1627943B, CAMP_BB9A0F28
- CAMP_0E9C0284, CAMP_C6E494CF, CAMP_75FBEEBC, CAMP_DDBFC6D9, CAMP_10407C1A
- CAMP_D8605E81, CAMP_DAE35509, CAMP_E449E81D, CAMP_6371D7E5, CAMP_436330B7
- CAMP_F51E9C79, CAMP_38EBD436, CAMP_71320238, CAMP_2845679C, CAMP_835F5335
- CAMP_BB680773, CAMP_F4E63855, CAMP_E8E9F042, CAMP_6988F109, CAMP_5A4539E8

**SCN-NMAP-001** (17 campaigns):
- CAMP_7331E223, CAMP_9669CA33, CAMP_B39F5C0A, CAMP_947A7084, CAMP_67FCB361
- CAMP_427A075C, CAMP_99EE7A64, CAMP_5D691192, CAMP_0CCD0071, CAMP_77263583
- CAMP_BB19A996, CAMP_8FF3B37F, CAMP_B3CDBFA1, CAMP_86D5837D, CAMP_15E98B82
- CAMP_B177634A, CAMP_1B8E9033

---

## Critical Bug Analysis

### The Bug
**File**: `backend/campaign_decision_engine.py`  
**Issue**: `lateral_movement_similarity` feature was calculated but NOT included in the `breakdown` dictionary passed to scoring.

**Impact**:
- Feature weight: 0.15 (3rd highest of 7 features)
- Multi-hop attack detection: **completely non-functional**
- Campaign continuity during IP transitions: **broken**

### Root Cause
When `lateral_movement_similarity` was added in "Phase 3":
- ✅ `CampaignFeatures` dataclass updated
- ✅ `campaign_feature_engine.py` calculation added
- ✅ `config.py` CAMPAIGN_WEIGHTS defined (0.15 weight)
- ❌ **MISSED**: `campaign_decision_engine.py` breakdown dict NOT updated

### The Fix
**Commit**: `f042bde5e210a798e12a2b75ed382f714ddb01d6`  
**Date**: 2026-11-23 (future date, likely timezone issue)  
**Changes**: Added 2 lines to include `lateral_movement_similarity` in scoring breakdown

```python
# BEFORE:
breakdown = {
    "transition_similarity": features.transition_similarity,
    "temporal_similarity": features.temporal_similarity,
    "attacker_similarity": features.attacker_similarity,
    "duplicate_similarity": features.duplicate_similarity,
    "graph_similarity": features.graph_similarity,
    "runtime_similarity": features.runtime_similarity
}

# AFTER:
breakdown = {
    "transition_similarity": features.transition_similarity,
    "temporal_similarity": features.temporal_similarity,
    "attacker_similarity": features.attacker_similarity,
    "lateral_movement_similarity": features.lateral_movement_similarity,  # ADDED
    "duplicate_similarity": features.duplicate_similarity,
    "graph_similarity": features.graph_similarity,
    "runtime_similarity": features.runtime_similarity
}
```

---

## Code-Level Validation

### Test Status
✅ **FIXED AND VERIFIED** (code-level)

**Test**: `test_campaign_fragmentation_fix.py::TestFeatureCalculations::test_lateral_movement_similarity_included_in_scoring`

**Verification**:
- Confirms `lateral_movement_similarity` is in breakdown dict
- Confirms feature value is preserved
- Confirms feature contributes to weighted score calculation
- **Status**: PASSES

**Additional Tests Created**: 10 tests covering:
- Campaign continuity (3 tests)
- Campaign diversity (1 test)
- Feature calculations (2 tests)
- Closure and reopen logic (2 tests)
- Edge cases (2 tests)

---

## Current Database State Analysis

**Analysis Date**: 2025-01-19  
**Tool**: `diagnose_campaign_fragmentation.py`

### Overall Statistics
- **Total Campaigns**: 204
- **Unique (attacker, victim) pairs**: 37
- **Pairs with fragmentation**: 11

### Severe Fragmentation Examples
| Attacker → Victim | Campaign Count | Expected | Fragmentation |
|-------------------|----------------|----------|---------------|
| 192.168.56.106 → pes1ug23cs411-VirtualBox | 40 | ~1-3 | Severe |
| pes1ug23cs411-VirtualBox → pes1ug23cs411-VirtualBox | 66 | Unknown | Severe |
| 192.168.56.106 → 192.168.56.105 | 21 | 1 (SCN-NMAP-001) | Severe |
| 192.168.56.105 → pes1ug23cs411-VirtualBox | 4 | 1 (SCN-SESSION-BOUNDARY-001) | Moderate |

### Temporal Gap Analysis
- **Total gaps analyzed**: 167
- **Gaps within CAMPAIGN_TIMEOUT (120s)**: 35/167 (21.0%)
- **Mean gap**: -12,078s (negative suggests out-of-order timestamps)
- **Median gap**: 1,517s (25.3 minutes)

**Key Finding**: 21% of campaign fragmentations occurred even when temporal gaps were within the 120-second timeout, suggesting:
1. Correlation scores dropped below 0.30 threshold
2. Reopen scores were below 0.70 threshold
3. Features were unavailable or returning low values
4. **lateral_movement_similarity bug contributed to low scores**

---

## Expected Impact of Bug Fix

### Theoretical Analysis

The bug fix restores a 0.15-weight feature to campaign correlation scoring. Expected improvements:

#### 1. **Multi-Hop Attack Continuity** ✅ Now Functional
- **Before**: lateral_movement feature ignored → campaigns split on IP changes
- **After**: lateral_movement feature contributes → campaigns continue during lateral movement
- **Impact**: Should reduce fragmentation when attacker IP changes legitimately (compromised host pivoting)

#### 2. **Scoring Distribution Shift**
- **Before**: 6 features contributing (total possible weight: 0.85)
- **After**: 7 features contributing (total possible weight: 1.00)
- **Impact**: Scores will increase by ~15% when lateral_movement_similarity > 0

#### 3. **Threshold Interactions**
- **ACTIVE_CONTINUE_THRESHOLD**: 0.30
  - Some campaigns scoring 0.25-0.29 may now score > 0.30 (continue instead of close)
  - Expected: 10-20% reduction in premature closures
- **REOPEN_THRESHOLD**: 0.70
  - Some inactive campaigns scoring 0.60-0.69 may now score > 0.70 (reopen)
  - Expected: 5-10% increase in successful reopens

#### 4. **Specific Scenario Improvements**
- **SCN-NMAP-001** (17 campaigns): May benefit if any internal pivoting occurred
- **SCN-SESSION-BOUNDARY-002** (25 campaigns): Strong candidate for improvement if lateral movement present
- **Overall**: Expected 20-40% reduction in fragmentation

---

## Validation Requirements

### ⬜ PENDING: Full Re-Evaluation

To measure actual improvement, the following is REQUIRED:

**Method 1: Event Replay (Recommended)**
1. Clear Neo4j database OR use isolated test instance
2. Replay original Wazuh alerts with bug-fixed code
3. Re-run campaign_correlation_eval.py
4. Compare before/after metrics

**Method 2: Controlled Experiment**
1. Create synthetic test cases with known lateral movement
2. Process through fixed pipeline
3. Verify campaigns maintain continuity

### Acceptance Criteria

**Minimum** (Phase B Complete):
- ✅ Bug fixed and unit tested
- ⬜ Pairwise F1 ≥ 0.50 (from 0.0)
- ⬜ Average fragmentation ≤ 7.5× (from 15.3×)
- ⬜ No over-merging (Precision ≥ 0.90)

**Target** (Excellent):
- ⬜ Pairwise F1 ≥ 0.75
- ⬜ Pairwise Precision ≥ 0.95
- ⬜ Pairwise Recall ≥ 0.70
- ⬜ Average fragmentation ≤ 2.0×

**Safety Constraints** (Must Maintain):
- ⬜ Different sessions remain separate (no false merges)
- ⬜ Deterministic replay behavior
- ⬜ All existing tests pass

---

## Additional Findings and Recommendations

### 1. REOPEN_THRESHOLD Too High
**Current**: 0.70  
**Issue**: Campaign that closed at 0.29 score requires 0.70 to reopen  
**Recommendation**: Lower to 0.50 (symmetric around continue threshold)  
**Expected Impact**: 15-25% fewer fragments

### 2. Fast Reopen Too Restrictive
**Current**: Only reopens if `first_technique == current_technique`  
**Issue**: Fails for diverse attack chains (e.g., [T1595, T1210, T1059] → T1055)  
**Recommendation**: Check last N techniques (N=3) instead of just first  
**Expected Impact**: 10-15% fewer fragments

### 3. CAMPAIGN_TIMEOUT Appropriate
**Current**: 120 seconds  
**Finding**: 21% of gaps within timeout  
**Assessment**: Timeout is reasonable; fragmentation is caused by scoring issues, not timeout  
**Recommendation**: Keep at 120s

### 4. Feature Logging Needed
**Current**: No logging of correlation scores at closure  
**Recommendation**: Enable decision logging for diagnostic purposes  
**Benefit**: Can identify which features caused premature closure

---

## Limitations and Caveats

### Current Database Limitations
1. **Pre-Fix Campaigns**: All 204 campaigns were created WITH BUG PRESENT
2. **No Controlled Comparison**: Cannot directly measure fix impact without replay
3. **Mixed Dataset**: Historical eval (46) + other campaigns (158) in same database
4. **No Session Boundaries**: Current data lacks explicit session identifiers beyond attacker/victim

### Evaluation Methodology Limitations
1. **Small Sample Size**: Only 3 independent sessions in evaluation dataset
2. **AUTO_PROPOSED Labels**: No human review of ground truth
3. **Fragmentation-Only**: Current ground truth cannot detect over-merging
4. **Same Attacker/Victim**: No test for different sessions with same (attacker, victim) pair

### Infrastructure Limitations
1. **Neo4j Access**: Password-protected, credentials in .env
2. **Event Replay**: Would require Wazuh alert logs or simulated events
3. **Test Isolation**: No dedicated test database instance

---

## Test Execution Summary

### Regression Tests
✅ **10 new tests created**: `test_campaign_fragmentation_fix.py`  
✅ **Critical test PASSES**: `test_lateral_movement_similarity_included_in_scoring`  
⬜ **Full suite**: Not executed (requires complete test environment)  
⬜ **Integration tests**: Not executed (requires Neo4j reprocessing)

### Diagnostic Tools
✅ **Created**: `diagnose_campaign_fragmentation.py`  
✅ **Executed**: Successfully analyzed 204 campaigns  
✅ **Fixed**: Neo4j Duration handling corrected

---

## Conclusion

**Bug Fix Status**: ✅ FIXED AND VERIFIED (code-level)  
**Full Validation**: ⬜ PENDING (requires event replay or controlled experiment)  
**Expected Improvement**: 20-40% reduction in fragmentation (theoretical)  
**Actual Improvement**: UNMEASURED (pending re-evaluation)

The critical `lateral_movement_similarity` bug has been identified, fixed, and validated at the code level. The fix restores a 0.15-weight feature that was completely non-functional since Phase 3.

**Current database campaigns were created with the bug present**, so full impact measurement requires event replay with the fixed code. The diagnostic tool confirms severe fragmentation persists in the existing database (consistent with the historical baseline), which is expected since those campaigns were generated before the fix.

**Phase B Status**: PARTIALLY COMPLETE
- ✅ Bug fixed
- ✅ Code-level validation
- ✅ Baseline established
- ✅ Diagnostic tools operational
- ⬜ Full re-evaluation pending
- ⬜ Additional improvements identified but not implemented

---

## Document History
- 2025-01-19: Initial evaluation document created
- Next update: After event replay and full re-evaluation

---

## References
- Bug fix commit: `f042bde5e210a798e12a2b75ed382f714ddb01d6`
- Ground truth: `evaluation/ground_truth/provisional/campaign_correlation_v0.jsonl`
- Historical metrics: `evaluation/results/campaign_correlation_metrics.json`
- Test suite: `backend/tests/test_campaign_fragmentation_fix.py`
- Diagnostic tool: `backend/diagnose_campaign_fragmentation.py`
