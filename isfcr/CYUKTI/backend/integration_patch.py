"""
integration_patch.py
====================
Integration patch for adding transition learner to realtime_socgraph.py

This script shows exactly where and how to integrate the transition_learner
into the main alert processing pipeline.

INTEGRATION POINT:
After line ~490 where transition is captured, add transition learner call.
This ensures all attacker behavior is recorded for learning, regardless of
whether campaigns split or continue.
"""

# ============================================================================
# ADD TO TOP OF realtime_socgraph.py (with other imports)
# ============================================================================

"""
from transition_learner import learner as transition_learner
from wazuh_rule_mapper import get_technique_for_alert
"""

# ============================================================================
# IN process_alert() FUNCTION
# ============================================================================

# LOCATION 1: After technique resolution (around line 350)
# Add Wazuh rule mapping as fallback for UNKNOWN techniques
"""
        # Existing code:
        resolution = resolve_mitre(alert)
        
        # ADD THIS:
        # If Wazuh didn't provide ATT&CK label, try our mapping
        if resolution.provenance == PROVENANCE_UNKNOWN:
            mapped_technique = get_technique_for_alert(alert)
            if mapped_technique:
                logger.info(
                    f"[MAPPER] Mapped rule {rule_id} -> {mapped_technique} "
                    f"(was UNKNOWN)"
                )
                # Re-resolve with mapped technique
                # Note: This is a simplified example - actual integration
                # would need to construct proper resolution object
                # For now, continue with UNKNOWN but log the mapping
                logger.info(
                    f"[MAPPER] Would map to {mapped_technique} - "
                    f"full integration requires resolver updates"
                )
"""

# LOCATION 2: After campaign creation (around line 490)
# Record all attacker activity for learning
"""
        # Existing code:
        transition = campaign_manager.get_or_create_campaign_with_transition(
            attacker_ip,
            victim_ip,
            mitre_id,
            event_time=event_time.value
        )
        campaign_id = transition.campaign_id
        
        # ADD THIS IMMEDIATELY AFTER:
        # Record attacker activity for cross-campaign transition learning
        # This runs regardless of whether campaign continued or split
        if mitre_id != "UNKNOWN":
            transition_learner.record_attacker_activity(
                attacker_ip=attacker_ip,
                technique=mitre_id,
                event_time=event_time.value
            )
            logger.debug(
                f"[LEARNER] Recorded {attacker_ip} -> {mitre_id} for learning"
            )
"""

# ============================================================================
# ADD NEW BACKGROUND TASK (optional but recommended)
# ============================================================================

"""
import threading
import time

def transition_learner_cleanup_task():
    \"\"\"
    Background task to periodically clean up stale attacker cache entries.
    Run this in a separate thread.
    \"\"\"
    while True:
        try:
            time.sleep(300)  # 5 minutes
            transition_learner.cleanup_cache()
            logger.info("[LEARNER] Cleaned up stale attacker cache entries")
        except Exception as e:
            logger.exception(f"[LEARNER] Cleanup task error: {e}")

# Start cleanup thread (add this to main() or at module level)
cleanup_thread = threading.Thread(
    target=transition_learner_cleanup_task,
    daemon=True
)
cleanup_thread.start()
logger.info("[LEARNER] Started transition learner cleanup task")
"""

# ============================================================================
# COMPLETE INTEGRATION EXAMPLE
# ============================================================================

def integration_example():
    """
    This function shows the complete integration in context.
    Copy the relevant parts to realtime_socgraph.py.
    """
    
    # Top of file imports:
    from transition_learner import learner as transition_learner
    from wazuh_rule_mapper import get_technique_for_alert
    
    # Inside process_alert(), after technique resolution:
    def process_alert_snippet_1(alert, rule_id, resolution):
        # Try mapper for UNKNOWN techniques
        if resolution.provenance == "UNKNOWN":
            mapped_technique = get_technique_for_alert(alert)
            if mapped_technique:
                print(f"[MAPPER] Rule {rule_id} -> {mapped_technique}")
                # Use mapped technique (implementation depends on resolver design)
    
    # Inside process_alert(), after campaign creation:
    def process_alert_snippet_2(attacker_ip, mitre_id, event_time, transition):
        campaign_id = transition.campaign_id
        
        # Record for learning (new code)
        if mitre_id != "UNKNOWN":
            learned = transition_learner.record_attacker_activity(
                attacker_ip=attacker_ip,
                technique=mitre_id,
                event_time=event_time.value
            )
            if learned:
                print(f"[LEARNER] Learned new transition from {attacker_ip}")
        
        # Continue with existing code...
        # create_attack_event()
        # update_attack_chain()
        # etc.


# ============================================================================
# TESTING INTEGRATION
# ============================================================================

def test_integration():
    """
    Test that integration is working correctly.
    Run this after integrating into realtime_socgraph.py.
    """
    from transition_learner import learner
    from datetime import datetime, timezone
    
    # Simulate attack sequence
    attacker = "192.168.1.100"
    
    # Alert 1: Reconnaissance
    learner.record_attacker_activity(
        attacker_ip=attacker,
        technique="T1595",
        event_time=datetime.now(timezone.utc)
    )
    
    import time
    time.sleep(1)
    
    # Alert 2: Credential Access (should learn T1595 -> T1110)
    learned = learner.record_attacker_activity(
        attacker_ip=attacker,
        technique="T1110",
        event_time=datetime.now(timezone.utc)
    )
    
    assert learned, "Should have learned transition T1595 -> T1110"
    
    # Verify in Neo4j
    from neo4j_client import driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (:Technique {attack_id: 'T1595'})
                  -[r:NEXT_TECHNIQUE]->
                  (:Technique {attack_id: 'T1110'})
            RETURN r.count AS count, r.learned_from_splits AS learned_from_splits
            """
        ).single()
        
        assert result is not None, "Transition should exist in Neo4j"
        print(f"✓ Transition learned: count={result['count']}")
        print(f"✓ Learned from splits: {result['learned_from_splits']}")
    
    print("\n✓ All integration tests passed!")


if __name__ == "__main__":
    print(__doc__)
    print("\nThis file documents the integration points.")
    print("Copy the relevant code sections to realtime_socgraph.py")
    print("\nTo test after integration, run:")
    print("  python integration_patch.py")
