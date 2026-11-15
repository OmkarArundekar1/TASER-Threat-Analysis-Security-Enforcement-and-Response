# CYUKTI Critical Fixes - Deployment Checklist

## Overview

This checklist guides you through deploying the critical fixes to address campaign fragmentation, labeling starvation, and incomplete integration.

**Expected Impact**:
- Campaigns per attack: 15 → 1-2
- UNKNOWN alerts: 82% → <30%
- System learns continuously instead of staying blind
- Full MISP and Shuffle integration

## Pre-Deployment

### 1. Backup

- [ ] Backup Neo4j database
  ```bash
  neo4j-admin backup --from=bolt://localhost:7687 --backup-dir=/backup/neo4j
  ```

- [ ] Backup configuration files
  ```bash
  cp backend/config.py backend/config.py.backup
  cp backend/.env backend/.env.backup
  ```

- [ ] Git commit current state
  ```bash
  git add -A
  git commit -m "Pre-deployment backup"
  git tag pre-critical-fixes
  ```

### 2. Review Changes

- [ ] Read `CRITICAL_FIXES_README.md` completely
- [ ] Understand each fix and its impact
- [ ] Review modified files:
  - `campaign_feature_engine.py` (unseen transitions)
  - `config.py` (merged weights)
  - `campaign_decision_engine.py` (merged features)
  - `campaign_manager.py` (display updates)
  - `evaluation_metrics.py` (inverse purity)
  - NEW: `transition_learner.py`
  - NEW: `wazuh_rule_mapper.py`
  - NEW: `misp_event_publisher.py`
  - NEW: `shuffle_workflow_publisher.py`

### 3. Dependencies

- [ ] Verify Python version (3.8+)
- [ ] Check Neo4j is running
- [ ] Verify all imports work:
  ```bash
  python -c "from campaign_feature_engine import engine"
  python -c "from transition_learner import learner"
  python -c "from wazuh_rule_mapper import mapper"
  ```

## Phase 1: Core Fixes (Critical - Must Deploy First)

### Step 1.1: Update Campaign Scoring

- [ ] Apply changes to `campaign_feature_engine.py`
  - Unseen transition default (0.1 instead of 0.0)
  - Merged transition feature

- [ ] Apply changes to `config.py`
  - Updated CAMPAIGN_WEIGHTS with merged features

- [ ] Apply changes to `campaign_decision_engine.py`
  - Use transition_similarity instead of separate features

- [ ] Apply changes to `campaign_manager.py`
  - Update feature display names

- [ ] Test scoring logic:
  ```bash
  cd backend
  python -c "from campaign_decision_engine import engine; print('OK')"
  ```

### Step 1.2: Verify Campaign Behavior

- [ ] Restart CYUKTI backend
  ```bash
  # Stop existing process
  pkill -f realtime_socgraph.py
  
  # Start with logging
  python realtime_socgraph.py 2>&1 | tee deployment_test.log
  ```

- [ ] Monitor campaign creation
  - Watch for "Campaign Features" output
  - Verify transition_similarity appears (not prediction/chain separately)
  - Check scores are reasonable (not all 0)

- [ ] Send test alerts (if test environment available)
  - Simulate multi-step attack
  - Verify campaigns don't split on unseen transitions
  - Check campaign count stays low (target: 1-2 per attack)

## Phase 2: Transition Learning

### Step 2.1: Deploy Transition Learner

- [ ] Add `transition_learner.py` to backend/
  - Already created, verify it's in place

- [ ] Test module independently:
  ```bash
  cd backend
  python -c "from transition_learner import learner; print('OK')"
  ```

### Step 2.2: Integrate with Alert Processing

- [ ] Open `realtime_socgraph.py`

- [ ] Add import at top:
  ```python
  from transition_learner import learner as transition_learner
  ```

- [ ] Find campaign creation (around line 490):
  ```python
  transition = campaign_manager.get_or_create_campaign_with_transition(...)
  campaign_id = transition.campaign_id
  ```

- [ ] Add immediately after:
  ```python
  # Record for cross-campaign learning
  if mitre_id != "UNKNOWN":
      transition_learner.record_attacker_activity(
          attacker_ip=attacker_ip,
          technique=mitre_id,
          event_time=event_time.value
      )
  ```

- [ ] Restart backend and verify learning:
  ```bash
  # Watch logs for "[LEARNER]" messages
  tail -f deployment_test.log | grep LEARNER
  ```

### Step 2.3: Add Cleanup Task (Optional but Recommended)

- [ ] Add background cleanup thread to `realtime_socgraph.py`:
  ```python
  import threading
  import time
  
  def learner_cleanup():
      while True:
          time.sleep(300)  # 5 minutes
          transition_learner.cleanup_cache()
  
  cleanup_thread = threading.Thread(target=learner_cleanup, daemon=True)
  cleanup_thread.start()
  ```

## Phase 3: Wazuh Rule Mapping

### Step 3.1: Initial Setup

- [ ] Add `wazuh_rule_mapper.py` to backend/
  - Already created with starter mappings

- [ ] Test mapper:
  ```bash
  cd backend
  python -c "from wazuh_rule_mapper import mapper; print(mapper.get_technique(5551))"
  # Should print: T1110
  ```

### Step 3.2: Map Your Rules

- [ ] Identify rules firing in your environment:
  ```bash
  # Check Wazuh alerts for common rule IDs
  grep '"id":' /var/ossec/logs/alerts/alerts.json | sort | uniq -c | sort -rn | head -20
  ```

- [ ] For each common rule ID:
  1. Check what it detects: `/var/ossec/ruleset/rules/*.xml`
  2. Find matching ATT&CK technique: https://attack.mitre.org/
  3. Add to `RULE_MAPPINGS` in `wazuh_rule_mapper.py`

- [ ] Document your mappings:
  ```python
  # Example:
  RULE_MAPPINGS = {
      # ... existing mappings ...
      
      # Custom mappings for your environment
      12345: ("T1234", "Description of what this detects"),
      # ...
  }
  ```

### Step 3.3: Integrate Mapper (Choose One)

**Option A: Update MITRE Resolver** (Recommended - proper integration)
- [ ] Modify `mitre_resolver.py` to use mapper as fallback
- [ ] Test with UNKNOWN alerts

**Option B: Log Only** (Quick validation)
- [ ] Add to `realtime_socgraph.py` after resolve_mitre():
  ```python
  if resolution.provenance == PROVENANCE_UNKNOWN:
      mapped = get_technique_for_alert(alert)
      if mapped:
          logger.info(f"[MAPPER] Rule {rule_id} would map to {mapped}")
  ```

### Step 3.4: Monitor Coverage

- [ ] Track UNKNOWN percentage:
  ```cypher
  // In Neo4j Browser
  MATCH (e:AttackEvent)
  WITH count(e) AS total,
       sum(CASE WHEN e.mitre_id = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown
  RETURN unknown, total, (100.0 * unknown / total) AS percent_unknown
  ```

- [ ] Target: <30% UNKNOWN
- [ ] If still high, add more rule mappings

## Phase 4: Evaluation Metrics

### Step 4.1: Update Metrics

- [ ] Changes to `evaluation_metrics.py` already applied
  - Added `inverse_purity()`
  - Added `campaigns_per_attack()`

- [ ] Test metrics:
  ```bash
  cd backend
  python -c "from evaluation_metrics import inverse_purity; print('OK')"
  ```

### Step 4.2: Update Evaluation Scripts

- [ ] Find existing evaluation scripts in `experiments/` or `evaluation/`

- [ ] Add inverse purity to reports:
  ```python
  from evaluation_metrics import inverse_purity, campaigns_per_attack
  
  purity = cluster_purity(true_labels, pred_labels)
  inv_purity = inverse_purity(true_labels, pred_labels)
  cpa = campaigns_per_attack(true_labels, pred_labels)
  
  print(f"Purity: {purity:.2%}")
  print(f"Inverse Purity: {inv_purity:.2%}")
  print(f"Campaigns/Attack: {cpa['mean']:.1f}")
  ```

- [ ] Run evaluation and verify:
  - Inverse purity shows splitting issue (before fix)
  - After fixes, both purity and inverse purity should be high

## Phase 5: MISP Integration

### Step 5.1: Configure MISP

- [ ] Add to `.env`:
  ```bash
  MISP_URL=https://your-misp-instance.com
  MISP_API_KEY=your_api_key_here
  VERIFY_MISP_SSL=false
  ```

- [ ] Test connection:
  ```bash
  cd backend
  python -c "from misp_event_publisher import create_publisher_from_config; p = create_publisher_from_config(); print(p.health_check())"
  ```

### Step 5.2: Test Publishing

- [ ] Publish test campaign:
  ```python
  from misp_event_publisher import create_publisher_from_config
  
  publisher = create_publisher_from_config()
  result = publisher.publish_campaign("test_campaign_id")
  print(result)
  ```

- [ ] Verify in MISP web interface:
  - Event created with proper tags
  - ATT&CK techniques tagged
  - IOCs present (IPs)
  - Risk score included

### Step 5.3: Automate Publishing

- [ ] Add to operation closure workflow:
  ```python
  # When operation closes
  from misp_event_publisher import create_publisher_from_config
  
  publisher = create_publisher_from_config()
  publisher.publish_operation(operation_id)
  ```

## Phase 6: Shuffle Integration

### Step 6.1: Configure Shuffle

- [ ] Add to `.env`:
  ```bash
  SHUFFLE_WEBHOOK=https://your-shuffle.com/api/v1/hooks/webhook_id
  SHUFFLE_BASE_URL=https://your-shuffle.com
  SHUFFLE_API_KEY=your_api_key_here
  ```

- [ ] Test connection:
  ```bash
  cd backend
  python -c "from shuffle_workflow_publisher import create_publisher_from_config; p = create_publisher_from_config(); print('OK')"
  ```

### Step 6.2: Test Workflow Trigger

- [ ] Publish test campaign:
  ```python
  from shuffle_workflow_publisher import create_publisher_from_config
  
  publisher = create_publisher_from_config()
  result = publisher.publish_campaign("test_campaign_id")
  print(result)
  ```

- [ ] Verify in Shuffle:
  - Workflow triggered
  - Payload includes campaign data
  - Suggested actions present

### Step 6.3: Configure Response Workflows

- [ ] In Shuffle, create workflows for:
  - Incident response (high/critical severity)
  - Investigation (medium severity)
  - Forensics collection (all severities)

- [ ] Set approval requirements:
  - Block/isolate actions: require human approval
  - Forensics collection: auto-approve
  - Notification: auto-approve

## Phase 7: Testing

### Step 7.1: Run Test Suite

- [ ] Execute comprehensive tests:
  ```bash
  cd backend
  python test_critical_fixes.py
  ```

- [ ] All tests should pass
- [ ] Fix any failures before proceeding

### Step 7.2: Integration Testing

- [ ] Simulate multi-step attack:
  ```bash
  # If you have test environment with Kali + Ubuntu
  # Run scripted attack and monitor
  ```

- [ ] Verify:
  - Campaigns stay together (1-2 per attack, not 15+)
  - Techniques properly labeled (not UNKNOWN)
  - Transitions learned
  - MISP events created
  - Shuffle workflows triggered

### Step 7.3: Performance Testing

- [ ] Monitor system resources
- [ ] Check Neo4j query performance
- [ ] Verify no memory leaks (learner cache)

## Phase 8: Production Deployment

### Step 8.1: Final Checks

- [ ] All tests pass
- [ ] Integration tests successful
- [ ] Backups verified
- [ ] Documentation updated

### Step 8.2: Deploy to Production

- [ ] Stop production backend
  ```bash
  systemctl stop cyukti-backend
  ```

- [ ] Deploy new code
  ```bash
  git pull origin main
  # or copy files
  ```

- [ ] Update configuration
  ```bash
  cp .env.production .env
  ```

- [ ] Start backend
  ```bash
  systemctl start cyukti-backend
  ```

### Step 8.3: Monitor Deployment

- [ ] Watch logs for errors:
  ```bash
  tail -f /var/log/cyukti/backend.log
  ```

- [ ] Monitor metrics:
  - Campaign fragmentation
  - UNKNOWN percentage
  - Transition learning rate
  - MISP/Shuffle publishing success

### Step 8.4: Rollback Plan (If Needed)

If issues occur:

- [ ] Stop backend
  ```bash
  systemctl stop cyukti-backend
  ```

- [ ] Restore backup:
  ```bash
  git checkout pre-critical-fixes
  cp config.py.backup config.py
  cp .env.backup .env
  ```

- [ ] Restore Neo4j:
  ```bash
  neo4j-admin restore --from=/backup/neo4j
  ```

- [ ] Start backend
  ```bash
  systemctl start cyukti-backend
  ```

## Post-Deployment

### Monitor for 24 Hours

- [ ] Campaign fragmentation metric
  - Query Neo4j for campaigns per hour
  - Compare to pre-fix baseline
  - Target: significant reduction

- [ ] UNKNOWN alert percentage
  - Track hourly
  - Should decrease as mapper covers more rules
  - Target: <30%

- [ ] Transition learning
  - Count new transitions learned
  - Should see steady growth
  - Check `learned_from_splits` flag

- [ ] Integration health
  - MISP events created successfully
  - Shuffle workflows triggered
  - No authentication failures

### Week 1 Tasks

- [ ] Review campaign quality
  - Sample campaigns manually
  - Verify they represent real attacks
  - Check for under-splitting or over-merging

- [ ] Tune weights if needed
  - Run evaluation on known attacks
  - Adjust CAMPAIGN_WEIGHTS
  - Test and re-deploy

- [ ] Expand rule mappings
  - Identify remaining UNKNOWN rules
  - Add mappings incrementally
  - Monitor coverage improvement

- [ ] Document anomalies
  - Any unexpected behavior
  - Performance issues
  - Integration problems

### Ongoing Maintenance

- [ ] Weekly: Review and update Wazuh mappings
- [ ] Monthly: Validate campaign grouping quality
- [ ] Quarterly: Re-evaluate weights on new data
- [ ] Continuous: Monitor all metrics

## Success Criteria

Deployment is successful when:

- ✓ Campaigns/attack reduced from 15 to 1-2
- ✓ UNKNOWN alerts below 30%
- ✓ Transitions being learned continuously
- ✓ Inverse purity >80%
- ✓ MISP events publishing successfully
- ✓ Shuffle workflows triggering successfully
- ✓ No critical errors in logs
- ✓ System performance acceptable

## Support

If you encounter issues:

1. Check logs: `backend/logs/` or systemd journal
2. Run test suite: `python test_critical_fixes.py`
3. Verify Neo4j: Check transitions in database
4. Review configurations: All env vars set correctly?
5. Check integration: MISP/Shuffle reachable?

## References

- `CRITICAL_FIXES_README.md` - Detailed fix documentation
- `test_critical_fixes.py` - Automated test suite
- `integration_patch.py` - Integration examples
- `feedback.pdf` - Original issues identified
