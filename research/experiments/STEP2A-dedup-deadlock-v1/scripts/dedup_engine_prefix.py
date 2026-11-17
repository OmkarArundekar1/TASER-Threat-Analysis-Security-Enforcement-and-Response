import hashlib
import time
from threading import Lock, Thread
from datetime import datetime, timezone
from config import (
    ENABLE_DUPLICATE_BUFFER,
    DUPLICATE_FLUSH_INTERVAL,
    MAX_PENDING_DUPLICATES,
    MAX_BATCH_SIZE,
    DEDUP_WINDOW,
    DEDUP_CLEANUP_INTERVAL
)
from neo4j_client import (
    find_recent_duplicate,
    batch_update_duplicates,
)

class DuplicateBuffer:
    def __init__(self):
        self.buffer = {}
        self.lock = Lock()

    def add_duplicate(
        self,
        fingerprint,
        event_id,
        campaign_id,
        tps,
        payload,
        timestamp=None,
    ):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        with self.lock:
            if (
                fingerprint not in self.buffer
                and len(self.buffer) >= MAX_PENDING_DUPLICATES
            ):
                return False

            if fingerprint not in self.buffer:
                self.buffer[fingerprint] = {
                    "fingerprint": fingerprint,
                    "event_id": event_id,
                    "campaign_id": campaign_id,
                    "occurrences": 1,
                    "total_tps": tps,
                    "latest_payload": payload,
                    "last_seen": timestamp,
                }

            else:
                entry = self.buffer[fingerprint]
                entry["occurrences"] += 1
                entry["total_tps"] += tps
                entry["latest_payload"] = payload
                entry["last_seen"] = timestamp

        return True

    def snapshot(self):
        with self.lock:
            data = [entry.copy() for entry in self.buffer.values()]
            self.buffer.clear()
        return data
    def size(self):
        with self.lock:
            return len(self.buffer)
    def flush(self):
        return self.snapshot()
    def restore(self, pending):
        with self.lock:
            for record in pending:
                fingerprint = record["fingerprint"]
                if fingerprint in self.buffer:
                    self.buffer[fingerprint]["occurrences"] += record["occurrences"]
                    self.buffer[fingerprint]["total_tps"] += record["total_tps"]
                    self.buffer[fingerprint]["latest_payload"] = record["latest_payload"]
                    self.buffer[fingerprint]["last_seen"] = record["last_seen"]
                else:
                    self.buffer[fingerprint] = record

class DuplicateFlushWorker:
    def __init__(self):
        self.running = False
        self.thread = None
    def start(self):
        if not ENABLE_DUPLICATE_BUFFER:
            return
        if self.running:
            return
        self.running = True
        self.thread = Thread(
            target=self.run,
            daemon=True
        )
        self.thread.start()
    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)

        pending = duplicate_buffer.flush()
        if pending:
            try:
                for i in range(0, len(pending), MAX_BATCH_SIZE):
                    batch = pending[i:i + MAX_BATCH_SIZE]
                    batch_update_duplicates(batch)
                logger.info(
                    "[Dedup Buffer] Final flush completed (%d records).",
                    len(pending)
                )
            except Exception as e:
                duplicate_buffer.restore(pending)
                logger.exception(
                    "[Dedup Buffer] Final flush failed. Pending updates restored."
                )
    def run(self):
        while self.running:
            time.sleep(DUPLICATE_FLUSH_INTERVAL)
            pending = duplicate_buffer.snapshot()
            if not pending:
                continue
            try:

                for i in range(0, len(pending), MAX_BATCH_SIZE):

                    batch = pending[i:i + MAX_BATCH_SIZE]

                    batch_update_duplicates(batch)

                print(
                    f"[Dedup Buffer] Flushed {len(pending)} aggregated duplicate updates."
                )

            except Exception as e:

                duplicate_buffer.restore(pending)

                print(
                    f"[Dedup Buffer] Flush failed. "
                    f"Restored {len(pending)} pending updates."
                )

                print(e)

def _seconds(event_time):
    """Dedup works in epoch seconds. event_time (aware datetime, from
    event_time.py) is the alert's own timestamp; None keeps the legacy
    processing-time behaviour for direct callers. realtime_socgraph always
    passes event time, so the dedup window is measured in event time and
    replay speed cannot change which alerts are duplicates."""
    return time.time() if event_time is None else event_time.timestamp()


class DeduplicationEngine:
    def __init__(self):
        self.cache = {}
        self.lock = Lock()
        # Seeded from the first observed (event) time, not the wall clock.
        self.last_cleanup = None

    def generate_fingerprint(
        self,
        attacker_ip,
        victim_ip,
        technique_id,
        rule_id,
        agent_id
    ):

        raw = (
            f"{attacker_ip}|"
            f"{victim_ip}|"
            f"{technique_id}|"
            f"{rule_id}|"
            f"{agent_id}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()
    def reload_cache(
        self,
        fingerprint,
        event_id,
        campaign_id,
        event_time=None
    ):
        now = _seconds(event_time)
        with self.lock:
            self.cache[fingerprint] = {
                "fingerprint": fingerprint,
                "first_seen": now,
                "last_seen": now,
                "occurrences": 1,
                "campaign_id": campaign_id,
                "event_id": event_id,
            }
    def is_duplicate(
        self,
        attacker_ip,
        victim_ip,
        technique_id,
        rule_id,
        agent_id,
        event_time=None
    ):
        now = _seconds(event_time)
        fingerprint = self.generate_fingerprint(
            attacker_ip,
            victim_ip,
            technique_id,
            rule_id,
            agent_id
        )
        with self.lock:
            self.cleanup(now)
            if fingerprint in self.cache:
                last_seen = self.cache[fingerprint]["last_seen"]

                # A late alert (now < last_seen) is inside the window.
                if (now - last_seen) <= DEDUP_WINDOW:
                    self.cache[fingerprint]["last_seen"] = max(last_seen, now)
                    self.cache[fingerprint]["occurrences"] += 1
                    return True, fingerprint
            db_event = find_recent_duplicate(
                fingerprint,
                reference_time=event_time
            )
            if db_event is not None:
                self.reload_cache(
                    fingerprint,
                    db_event["event_id"],
                    db_event["campaign_id"],
                    event_time=event_time
                )
                return True, fingerprint

            return False, fingerprint

    def register_event(
        self,
        fingerprint,
        campaign_id=None,
        event_id=None,
        event_time=None
    ):

        now = _seconds(event_time)
        with self.lock:
            if fingerprint not in self.cache:
                self.cache[fingerprint] = {
                    "fingerprint": fingerprint,
                    "first_seen": now,
                    "last_seen": now,
                    "occurrences": 1,
                    "campaign_id": campaign_id,
                    "event_id": event_id
                }
            else:
                self.cache[fingerprint]["last_seen"] = max(self.cache[fingerprint]["last_seen"], now)
                self.cache[fingerprint]["occurrences"] += 1
                if campaign_id is not None:
                    self.cache[fingerprint]["campaign_id"] = campaign_id
                if event_id is not None:
                    self.cache[fingerprint]["event_id"] = event_id

    def get_metadata(self, fingerprint):
        return self.cache.get(fingerprint)

    def cleanup(self, now=None):
        if now is None:
            now = time.time()
        if self.last_cleanup is None:
            self.last_cleanup = now
            return
        if (now - self.last_cleanup) < DEDUP_CLEANUP_INTERVAL:
            # also covers late alerts (now < last_cleanup): no cleanup
            return
        expired = []
        for fingerprint, data in self.cache.items():
            if (now - data["last_seen"]) > DEDUP_WINDOW:
                expired.append(fingerprint)
        for fingerprint in expired:
            del self.cache[fingerprint]

        self.last_cleanup = now

    def reset(self):
        """Clears in-memory state (replay harness: fresh run per database)."""
        with self.lock:
            self.cache.clear()
            self.last_cleanup = None
duplicate_buffer = DuplicateBuffer()
duplicate_flush_worker = DuplicateFlushWorker()
dedup_engine = DeduplicationEngine()