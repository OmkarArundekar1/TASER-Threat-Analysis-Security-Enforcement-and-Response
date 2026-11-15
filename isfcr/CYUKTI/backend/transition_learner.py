"""
transition_learner.py
=====================
Additional transition learning layer that records technique sequences
from the same attacker IP even when campaigns split due to low scores.

This addresses the critical issue: when a campaign splits because the
system hasn't seen a particular transition, that transition is never
recorded, so the system never learns it. This module ensures we learn
from all attacker behavior, not just behavior that matches existing patterns.

Policy:
- Record transitions when alerts come from the same attacker IP
- Only record if time gap is small (within LEARNING_TIME_WINDOW)
- Skip if already recorded by main chain updater
- Never override campaign decisions - this is pure learning
"""

from datetime import datetime, timezone, timedelta
from neo4j_client import driver
from config import CAMPAIGN_TIMEOUT

# Maximum time window for learning cross-campaign transitions
LEARNING_TIME_WINDOW = CAMPAIGN_TIMEOUT * 2  # 240 seconds by default


class TransitionLearner:
    """
    Learns technique transitions across campaign boundaries.
    Maintains in-memory cache of recent attacker activity for learning.
    """
    
    def __init__(self):
        # Cache: attacker_ip -> (last_technique, last_seen)
        self.attacker_cache = {}
    
    def record_attacker_activity(
        self,
        attacker_ip: str,
        technique: str,
        event_time: datetime = None
    ):
        """
        Record technique from attacker and learn transition if applicable.
        
        Args:
            attacker_ip: Source IP address
            technique: Current ATT&CK technique ID
            event_time: Event timestamp (defaults to now)
        
        Returns:
            True if a transition was learned, False otherwise
        """
        if event_time is None:
            event_time = datetime.now(timezone.utc)
        
        # Check for previous activity from this attacker
        if attacker_ip not in self.attacker_cache:
            # First time seeing this attacker, just cache it
            self.attacker_cache[attacker_ip] = (technique, event_time)
            return False
        
        previous_technique, previous_time = self.attacker_cache[attacker_ip]
        
        # Update cache with current activity
        self.attacker_cache[attacker_ip] = (technique, event_time)
        
        # Don't learn if same technique
        if previous_technique == technique:
            print(f"[LEARNER] Skipping self-transition: {technique}")
            return False
        
        # Check time window
        time_gap = (event_time - previous_time).total_seconds()
        if time_gap > LEARNING_TIME_WINDOW:
            print(
                f"[LEARNER] Time gap too large: {time_gap:.1f}s > "
                f"{LEARNING_TIME_WINDOW}s"
            )
            return False
        
        # Learn this transition
        return self._learn_transition(
            previous_technique,
            technique,
            attacker_ip,
            time_gap
        )
    
    def _learn_transition(
        self,
        previous: str,
        current: str,
        attacker_ip: str,
        time_gap: float
    ) -> bool:
        """
        Record a transition in Neo4j.
        Uses same NEXT_TECHNIQUE structure as chain_updater.py.
        """
        try:
            with driver.session() as session:
                # Check if transition already exists
                existing = session.run(
                    """
                    MATCH (a:Technique {attack_id: $src})
                          -[r:NEXT_TECHNIQUE]->
                          (b:Technique {attack_id: $dst})
                    RETURN r.count AS count
                    """,
                    src=previous,
                    dst=current
                ).single()
                
                if existing and existing["count"] > 0:
                    # Already learned by main chain updater, skip
                    print(
                        f"[LEARNER] Transition already known: "
                        f"{previous} -> {current}"
                    )
                    return False
                
                print(
                    f"\n[LEARNER] Learning new transition from attacker behavior"
                )
                print(f"  Attacker : {attacker_ip}")
                print(f"  Transition: {previous} -> {current}")
                print(f"  Time gap : {time_gap:.1f}s")
                
                # Record the transition
                session.run(
                    """
                    MATCH (a:Technique {attack_id: $src})
                    MATCH (b:Technique {attack_id: $dst})
                    
                    MERGE (a)-[r:NEXT_TECHNIQUE]->(b)
                    
                    ON CREATE SET
                        r.count = 1,
                        r.created_at = datetime(),
                        r.last_seen = datetime(),
                        r.confidence = 1.0,
                        r.learned_from_splits = true
                    
                    ON MATCH SET
                        r.count = r.count + 1,
                        r.last_seen = datetime(),
                        r.learned_from_splits = coalesce(r.learned_from_splits, false)
                    """,
                    src=previous,
                    dst=current
                )
                
                # Recalculate confidences for all transitions from source
                session.run(
                    """
                    MATCH (a:Technique {attack_id: $src})-[r:NEXT_TECHNIQUE]->()
                    
                    WITH collect(r) AS rels,
                         sum(r.count) AS total
                    
                    UNWIND rels AS rel
                    
                    SET rel.confidence =
                        CASE
                            WHEN total = 0 THEN 0.0
                            ELSE toFloat(rel.count) / total
                        END
                    """,
                    src=previous
                )
                
                print(f"[LEARNER] Successfully learned transition\n")
                return True
                
        except Exception as e:
            print(f"[LEARNER ERROR] Failed to learn transition: {e}")
            return False
    
    def cleanup_cache(self, older_than_seconds: float = None):
        """
        Remove stale entries from attacker cache.
        
        Args:
            older_than_seconds: Remove entries older than this (default: LEARNING_TIME_WINDOW)
        """
        if older_than_seconds is None:
            older_than_seconds = LEARNING_TIME_WINDOW
        
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
        
        stale_attackers = [
            ip for ip, (_, last_seen) in self.attacker_cache.items()
            if last_seen < cutoff
        ]
        
        for ip in stale_attackers:
            del self.attacker_cache[ip]
        
        if stale_attackers:
            print(
                f"[LEARNER] Cleaned up {len(stale_attackers)} stale "
                f"attacker entries"
            )


# Global instance
learner = TransitionLearner()
