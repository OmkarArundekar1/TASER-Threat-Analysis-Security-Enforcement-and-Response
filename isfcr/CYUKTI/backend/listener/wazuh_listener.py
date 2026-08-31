import os
import sys
import signal
from utils import generate_alert_id
from collections import OrderedDict
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
import json
import time
import logging
from queue import Queue, Empty
from threading import Thread, Event, Lock
from config import ENABLE_DUPLICATE_BUFFER, POLL_INTERVAL
from realtime_socgraph import process_alert
from dedup_engine import (
    duplicate_flush_worker,
    duplicate_buffer,
)

from neo4j_client import batch_update_duplicates
from campaign_manager import campaign_manager
from operation_manager import operation_manager

ALERT_FILE = "/var/ossec/logs/alerts/alerts.json"
WATCH_DIR = os.path.dirname(ALERT_FILE)

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prerana_listener.log")

QUEUE_SIZE = 1000
MAX_ALERT_CACHE = 10000
ALERT_CACHE_TTL = 600     
seen_alerts = OrderedDict()
seen_alerts_lock = Lock()

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PreranaListener")

alert_queue = Queue(maxsize=QUEUE_SIZE)

stop_event = Event()

def shutdown_handler(signum, frame):
    logger.info("Shutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

def safe_json_load(line):
    try:
        return json.loads(line)
    except Exception:
        logger.warning("Invalid JSON skipped.")
        return None
        
def already_seen(alert_id):
    now = time.time()
    with seen_alerts_lock:
        expired = [
            key
            for key, ts in seen_alerts.items()
            if now - ts > ALERT_CACHE_TTL
        ]
        for key in expired:
            del seen_alerts[key]
        if alert_id in seen_alerts:
            return True
        seen_alerts[alert_id] = now
        while len(seen_alerts) > MAX_ALERT_CACHE:
            seen_alerts.popitem(last=False)
        return False
        
def queue_worker():
    logger.info("Queue worker started.")
    while not stop_event.is_set() or not alert_queue.empty():
        try:
            alert = alert_queue.get(timeout=1)
            try:
                logger.info("Processing alert from queue")
                result = process_alert(alert)
                
                if not result:
                    logger.error("process_alert() returned False")
            except Exception as e:
                logger.exception(f"process_alert failed: {e}")
            finally:
                alert_queue.task_done()
        except Empty:
            continue
    logger.info("Queue worker stopped.")

def maintenance_worker():
    logger.info("Maintenance worker started.")

    while not stop_event.is_set():
        try:
            campaign_manager.expire_active_campaigns()
            operation_manager.expire_active_operations()

        except Exception as e:
            logger.exception(
                f"Maintenance worker failed: {e}"
            )

        time.sleep(5)

    logger.info("Maintenance worker stopped.")
    
class AlertFileHandler:
    def __init__(self):
        self.offset = 0
        offset_file = os.path.join(
            os.path.dirname(__file__),
            "offset.dat"
        )
        if os.path.exists(ALERT_FILE):
            current_size = os.path.getsize(ALERT_FILE)
        else:
            current_size = 0
        if os.path.exists(offset_file):
            try:
                with open(offset_file, "r") as f:
                    data = f.read().strip()
                if data:
                    offset = int(data)
                    if offset < 0:
                        raise ValueError("Negative offset")
                    if offset > current_size:
                        logger.warning(
                            "Saved offset beyond EOF. Resetting to file end."
                        )
                        offset = current_size
                    self.offset = offset
                else:
                    self.offset = current_size
            except Exception:
                logger.warning(
                    "Invalid offset.dat. Rebuilding offset."
                )
                self.offset = current_size
        else:
            self.offset = current_size
        logger.info(f"Initial offset = {self.offset}")
        
    def read_new_lines(self):
        try:
            if not os.path.exists(ALERT_FILE):
                return
            current_size = os.path.getsize(ALERT_FILE)
            if self.offset > current_size:
                logger.warning(
                    "Offset beyond file size. Assuming log rotation."
                )
                self.offset = 0
            offset_file = os.path.join(
                os.path.dirname(__file__),
                "offset.dat"
            )
            with open(ALERT_FILE, "r") as f:
                f.seek(self.offset)
                last_good_offset = self.offset
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        logger.info(
                            "Incomplete JSON detected. Waiting for next poll."
                        )
                        break
                    line = line.strip()
                    if not line:
                        continue
                    alert = safe_json_load(line)
                    
                    if alert is None:
                        if not line.endswith("}"):
                            logger.debug("Partial JSON detected.")
                            break
                    
                        logger.warning("Skipping malformed JSON.")
                        last_good_offset = f.tell()
                        continue
                    alert["alert_id"] = generate_alert_id(alert)
                    logger.info(
                        "Alert ID: %s",
                        alert["alert_id"][:12]
                    )
                    if already_seen(alert["alert_id"]):
                        logger.info(
                            "Duplicate raw alert ignored: %s",
                            alert["alert_id"][:12]
                        )
                        last_good_offset = f.tell()
                        continue
                    try:
                        alert_queue.put_nowait(alert)
                        last_good_offset = f.tell()
                    except Exception:
                        logger.warning(
                            "Queue full. Alert dropped."
                        )
                        break

                if last_good_offset != self.offset:
                    logger.info(
                        "Offset updated: %d -> %d",
                        self.offset,
                        last_good_offset
                    )
                self.offset = last_good_offset
                with open(offset_file, "w") as fp:
                    fp.write(str(self.offset))
        except Exception as e:
            logger.exception(f"Failure: {e}")
            
def process_existing_alerts(handler):
    try:
        handler.read_new_lines()

    except Exception as e:
        logger.exception(f"Startup catch-up failed: {e}")

def start_listener():

    logger.info("=" * 60)
    logger.info("Prerana Listener initialized.")
    logger.info(f"Watching : {ALERT_FILE}")
    logger.info("=" * 60)
    handler = AlertFileHandler()
    worker = Thread(
        target=queue_worker,
        daemon=True
    )

    worker.start()
    maintenance = Thread(
        target=maintenance_worker,
        daemon=True
    )
    
    maintenance.start()
    logger.info("Polling started.")
    logger.info("Waiting for new Wazuh alerts...")
    if ENABLE_DUPLICATE_BUFFER:
        duplicate_flush_worker.start()
        logger.info("Duplicate buffer flush worker started.")
    try:
        while not stop_event.is_set():
            try:
                handler.read_new_lines()
            except Exception:
                logger.exception("Unexpected error during polling.")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Stopping listener...")
    finally:
        stop_event.set()
        alert_queue.join()
        worker.join(timeout=5)
        maintenance.join(timeout=5)
        if ENABLE_DUPLICATE_BUFFER:
            duplicate_flush_worker.stop()
        logger.info("Listener stopped.")
if __name__ == "__main__":
    start_listener()