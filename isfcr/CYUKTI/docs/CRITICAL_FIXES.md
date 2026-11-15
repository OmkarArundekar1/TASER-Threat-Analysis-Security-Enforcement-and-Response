# CYUKTI Critical Fixes - Implementation Guide

## Overview

This document describes the critical fixes implemented to address the major issues identified in the capstone feedback. These fixes target the root causes of campaign fragmentation, labeling starvation, and incomplete integration.

## Problem Summary

The original system had severe issues that undermined its core promise:

1. **Campaign Fragmentation**: 3 real attacks became 40+ campaigns
2. **Labeling Starvation**: 82% of alerts were UNKNOWN, starving ML components
3. **Never Recording Unseen Transitions**: System couldn't learn new patterns
4. **Misleading Metrics**: High purity masked severe splitting
5. **Incomplete Integration**: MISP and Shuffle not properly publishing

## Fixes Implemented

### 1. Fix Campaign Splitting (Critical Priority)

**Problem**: Unseen transitions scored 0.0, causing campaigns to split immediately when attackers used new techniques.

**Root Cause**: The system treated "never seen this transition" as "evidence against", when it actually means "no information".

**Fix**: `campaign_feature_engine.py` - chain_similarity()
```python
# OLD: Returned 0.0 for unseen transitions
if total == 0:
    return 0.0

# NEW: Return small default value instead
UNSEEN_TRANSITION_DEFAULT = 0.1
if total == 0:
    return UNSEEN_TRANSITION_DEFAULT
```

**Impact**:
- Campaigns no longer split on first unseen technique
- Expected reduction: 15 campaigns/attack → 1-2 campaigns/attack
- Allows system to learn instead of staying blind

### 2. Record Transitions Even When Splitting

**Problem**: When campaigns split due to low score, the transition causing the split was never recorded. System never learned from its mistakes.

**Fix**: `transition_learner.py` (NEW MODULE)

Key features:
- Tracks all attacker activity in memory cache
- Records transitions from same attacker IP within time window
- Works independently of campaign decisions
- Marks transitions with `learned_from_splits = true` flag

Usage:
```python
from transition_learner import learner

# After processing each alert
learner.record_attacker_activity(
    attacker_ip=attacker_ip,
    technique=technique,
    event_time=event_time
)
```

**Impact**:
- System learns from all attacker behavior, not just patterns it already knows
- Transitions that cause splits are now recorded for future learning
- Knowledge base grows even when initial grouping fails

### 3. Merge Duplicate Features

**Problem**: Prediction and chain similarity both measure "what comes next" but had separate weights (0.35 + 0.30 = 0.65 total). One missing pattern dominated decision.

**Fix**: `config.py` and `campaign_feature_engine.py`

Merged into single feature:
```python
# OLD WEIGHTS
CAMPAIGN_WEIGHTS = {
    "prediction_similarity": 0.35,
    "chain_similarity": 0.30,
    ...
}

# NEW WEIGHTS
CAMPAIGN_WEIGHTS = {
    "transition_similarity": 0.40,  # Combined
    "temporal_similarity": 0.30,
    "attacker_similarity": 0.20,
    ...
}
```

Feature extraction:
```python
# Average when both available
if pred_sim is not None and chain_sim is not None:
    transition_sim = (pred_sim + chain_sim) / 2
```

**Impact**:
- One missing pattern no longer kills the score
- More robust to incomplete knowledge
- Better balance between features

### 4. Address Labeling Starvation

**Problem**: 82% of alerts labeled UNKNOWN because Wazuh doesn't provide ATT&CK labels for most rules.

**Fix**: `wazuh_rule_mapper.py` (NEW MODULE)

Provides explicit Wazuh rule → ATT&CK technique mappings:
```python
RULE_MAPPINGS = {
    5551: ("T1110", "SSH Brute Force"),
    5710: ("T1078", "Valid Accounts - SSH"),
    2501: ("T1548", "Abuse Elevation Control"),
    # ... 50+ mappings
}
```

Integration:
```python
from wazuh_rule_mapper import get_technique_for_alert

technique = get_technique_for_alert(alert)
if technique:
    # Use mapped technique
else:
    # Fall back to Wazuh-provided label
```

**What Security Team Must Do**:
1. Review actual Wazuh rules firing in your environment
2. Map each rule to appropriate ATT&CK technique(s)
3. Add mappings to `RULE_MAPPINGS` dict
4. Test against known attacks
5. Maintain as rules are added/updated

**Impact**:
- Expected: UNKNOWN alerts reduced from 82% to <30%
- ML components get usable data
- Next-step prediction becomes meaningful
- Stage scoring accurately reflects attack progression

### 5. Add Inverse Purity Metric

**Problem**: System reported only purity (each campaign = one attack?), hiding the real issue of splitting.

**Fix**: `evaluation_metrics.py` - Added:
- `inverse_purity()`: Is each real attack kept in one campaign?
- `campaigns_per_attack()`: How many predicted campaigns per real attack?

Usage:
```python
from evaluation_metrics import inverse_purity, campaigns_per_attack

# After evaluation
purity = cluster_purity(true_labels, predicted_labels)
inv_purity = inverse_purity(true_labels, predicted_labels)
fragmentation = campaigns_per_attack(true_labels, predicted_labels)

print(f"Purity: {purity:.2%}")  # High even when splitting
print(f"Inverse Purity: {inv_purity:.2%}")  # Shows splitting
print(f"Campaigns/Attack: {fragmentation['mean']:.1f}")  # Target: ~1.0
```

**Impact**:
- Splitting is now immediately visible in metrics
- Balanced view: both over-merging and over-splitting
- Aligns evaluation with actual system goal

### 6. Enhanced MISP Publishing

**Problem**: MISP integration existed but wasn't publishing properly structured events.

**Fix**: `misp_event_publisher.py` (NEW MODULE)

Features:
- Extracts full campaign intelligence from Neo4j
- Builds properly structured MISP events with:
  - Campaign metadata and timeline
  - ATT&CK technique tags
  - IOCs (IPs)
  - Risk scores and severity
  - Attack chain sequence
- Handles both creation and updates

Usage:
```python
from misp_event_publisher import create_publisher_from_config

publisher = create_publisher_from_config()

# Publish single campaign
result = publisher.publish_campaign(campaign_id)

# Publish entire operation
result = publisher.publish_operation(operation_id)
```

**Impact**:
- MISP events properly structured and tagged
- Campaign intelligence shared with other tools
- Threat intelligence integration complete

### 7. Enhanced Shuffle Publishing

**Problem**: Shuffle integration incomplete, not publishing actionable workflows.

**Fix**: `shuffle_workflow_publisher.py` (NEW MODULE)

Features:
- Builds Shuffle workflow payloads with campaign data
- Includes suggested actions based on techniques
- Priority and approval requirements based on severity
- Context for automated decision making

Usage:
```python
from shuffle_workflow_publisher import create_publisher_from_config

publisher = create_publisher_from_config()

# Trigger incident response workflow
result = publisher.publish_campaign(
    campaign_id,
    workflow_type="incident_response"
)

# Check execution status
if result["success"]:
    print(f"Workflow triggered: {result['execution_id']}")
```

**Impact**:
- Automated response workflows triggered with proper context
- Human approval required for high-risk actions
- Integration with SOAR platform complete

## Implementation Steps

### Immediate (Fix campaign splitting - highest priority)

1. **Update campaign_feature_engine.py**: Apply unseen transition default
2. **Update config.py**: Merge feature weights
3. **Update campaign_decision_engine.py**: Use merged feature
4. **Update campaign_manager.py**: Update feature display names
5. **Test**: Run on existing data, verify campaigns/attack drops

### Short Term (Enable learning)

1. **Deploy transition_learner.py**: Add to backend
2. **Integrate in realtime_socgraph.py**: Call after each alert
3. **Add cleanup task**: Periodically clean stale cache
4. **Monitor**: Check transitions marked with `learned_from_splits`

### Medium Term (Fix labeling)

1. **Review Wazuh rules**: Identify rules firing in your environment
2. **Map to ATT&CK**: Use `wazuh_rule_mapper.py` template
3. **Test mappings**: Validate against known attacks
4. **Integrate**: Add mapper to alert processing pipeline
5. **Measure**: Track UNKNOWN percentage, target <30%

### Integration (Complete publishing)

1. **Configure MISP**: Set MISP_URL and MISP_API_KEY in .env
2. **Test MISP publisher**: Publish test campaign
3. **Configure Shuffle**: Set SHUFFLE_WEBHOOK in .env
4. **Test Shuffle publisher**: Trigger test workflow
5. **Automate**: Add to operation closure workflow

## Testing and Validation

### Campaign Fragmentation Test

```python
# Before fix
campaigns_before = count_campaigns_for_attack(attack_id)
# Expected: 15+ campaigns

# After fix
campaigns_after = count_campaigns_for_attack(attack_id)
# Target: 1-2 campaigns
```

### Labeling Coverage Test

```python
# Before mapping
unknown_count_before = count_unknown_alerts()
# Expected: ~82%

# After mapping
unknown_count_after = count_unknown_alerts()
# Target: <30%
```

### Integration Test

```python
# MISP
misp_publisher = create_misp_publisher_from_config()
assert misp_publisher.health_check()
result = misp_publisher.publish_campaign("test_campaign")
assert result["success"]

# Shuffle
shuffle_publisher = create_shuffle_publisher_from_config()
result = shuffle_publisher.publish_campaign("test_campaign")
assert result["success"]
```

### Metrics Validation

```python
from evaluation_metrics import inverse_purity, campaigns_per_attack

# Run evaluation
purity = cluster_purity(true_labels, pred_labels)
inv_purity = inverse_purity(true_labels, pred_labels)
cpa = campaigns_per_attack(true_labels, pred_labels)

# Check targets
assert purity > 0.80  # Each campaign is one attack
assert inv_purity > 0.80  # Each attack is one campaign
assert cpa["mean"] < 2.0  # Low fragmentation
```

## Configuration

### Required Environment Variables

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# MISP
MISP_URL=https://misp.yourdomain.com
MISP_API_KEY=your_api_key
VERIFY_MISP_SSL=false

# Shuffle
SHUFFLE_WEBHOOK=https://shuffle.yourdomain.com/api/v1/hooks/webhook_id
SHUFFLE_BASE_URL=https://shuffle.yourdomain.com
SHUFFLE_API_KEY=your_api_key
```

### Campaign Weight Tuning (Optional)

The weights in `config.py` can be tuned using labeled data:

```python
CAMPAIGN_WEIGHTS = {
    "transition_similarity": 0.40,  # Adjust based on validation
    "temporal_similarity": 0.30,
    "attacker_similarity": 0.20,
    "runtime_similarity": 0.05,
    "graph_similarity": 0.03,
    "duplicate_similarity": 0.02,
}
```

Process:
1. Collect labeled dataset (AIT or own attacks)
2. Try different weight combinations
3. Measure inverse_purity and campaigns_per_attack
4. Choose weights that minimize fragmentation while maintaining purity

## Expected Outcomes

After implementing all fixes:

| Metric | Before | Target | Impact |
|--------|--------|--------|--------|
| Campaigns per attack | 15 | 1-2 | System groups properly |
| UNKNOWN alerts | 82% | <30% | ML components work |
| Inverse purity | Low | >80% | Attacks stay together |
| Transitions learned | 3 | 100+ | System learns continuously |
| MISP events | Incomplete | Complete | Intelligence sharing works |
| Shuffle workflows | None | Automated | Response is triggered |

## Maintenance

### Regular Tasks

1. **Weekly**: Review and update Wazuh rule mappings
2. **Monthly**: Tune campaign weights if needed
3. **Quarterly**: Validate against new attack datasets

### Monitoring

```python
# Check system health
from transition_learner import learner
from wazuh_rule_mapper import mapper

# Transition learning
stats = learner.get_cache_stats()
print(f"Active attackers tracked: {stats['active_count']}")

# Label coverage
coverage = mapper.get_coverage_stats()
print(f"Mapped rules: {coverage['explicit_mappings']}")

# Campaign fragmentation
avg_campaigns = get_avg_campaigns_per_attack()
print(f"Campaigns/attack: {avg_campaigns:.1f}")
```

## Future Work

Beyond these fixes, consider:

1. **Cross-IP campaign tracking**: Let campaigns follow lateral movement
2. **More sophisticated weight learning**: Use gradient descent on labeled data
3. **Expanded Wazuh mappings**: Community-driven rule mapping database
4. **Advanced Shuffle workflows**: Multi-stage response orchestration
5. **Real-time metrics dashboard**: Monitor fragmentation live

## Support

For issues or questions:
1. Check logs: `backend/logs/`
2. Review Neo4j queries: Verify transitions being recorded
3. Validate configurations: Ensure all env vars set
4. Test components individually before integration

## References

- Original feedback document: `feedback.pdf`
- ATT&CK Enterprise: https://attack.mitre.org/
- Wazuh rules: `/var/ossec/ruleset/rules/`
- MISP API: https://www.misp-project.org/openapi/
- Shuffle docs: https://shuffler.io/docs
