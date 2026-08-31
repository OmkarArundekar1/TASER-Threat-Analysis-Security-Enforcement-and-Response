import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

ALERT_FILE = os.environ.get("WAZUH_ALERT_FILE", "/var/ossec/logs/alerts/alerts.json")
POLL_INTERVAL = 0.5

CAMPAIGN_TIMEOUT = 120
MAX_CAMPAIGN_CACHE = 1500

OPERATION_TIMEOUT = 120

DEDUP_WINDOW = 90            # seconds
DEDUP_CLEANUP_INTERVAL = 60

MISP_URL = os.environ.get("MISP_URL", "https://localhost:8443")
MISP_API_KEY = os.environ.get("MISP_API_KEY", "")

VERIFY_MISP_SSL = False

DUPLICATE_FLUSH_INTERVAL = 7      # seconds
MAX_PENDING_DUPLICATES = 1000
MAX_BATCH_SIZE = 200
ENABLE_DUPLICATE_BUFFER = True

SHUFFLE_WEBHOOK = os.environ.get("SHUFFLE_WEBHOOK", "")

LOG_LEVEL = "INFO"

TPS_MAP = {
    "Reconnaissance": 10,
    "Credential Access": 45,
    "Initial Access": 70,
    "Privilege Escalation": 90,
    "Lateral Movement": 110,
    "Exfiltration": 150
}

PLATFORM_WEIGHT = 0.25
MITIGATION_WEIGHT = 0.50
THREAT_ACTOR_WEIGHT = 0.15
MALWARE_WEIGHT = 0.20
TOOL_WEIGHT = 0.20
RUNTIME_EVENT_WEIGHT = 0.50
DETECTION_WEIGHT = 1.00
THREAT_INTEL_WEIGHT = 1.00

MAX_SEVERITY_SCORE = 40.0

PREDICTION_WEIGHT = 2.0
DUPLICATE_WEIGHT = 1.0
CAMPAIGN_WEIGHT = 0.75

THREAT_INTEL_RISK_WEIGHT = 1.0

MAX_DYNAMIC_RISK = 100.0

LOW_RISK_THRESHOLD = 25
MEDIUM_RISK_THRESHOLD = 50
HIGH_RISK_THRESHOLD = 75

RISK_TREND_DELTA = 5

MIN_PREDICTION_CONFIDENCE = 30
MIN_TRANSITION_OBSERVATIONS = 3

GRAPH_DENSITY_WEIGHT = 9
CHAIN_DEPTH_WEIGHT = 1
COMPLEXITY_WEIGHT = 0.10
STRUCTURAL_RISK_WEIGHT = 0.10
EVOLUTION_RATE_WEIGHT = 70

CAMPAIGN_WEIGHTS = {
    "prediction_similarity": 0.35,
    "chain_similarity": 0.30,
    "temporal_similarity": 0.20,
    "attacker_similarity": 0.10,
    "runtime_similarity": 0.03,
    "graph_similarity": 0.01,
    "duplicate_similarity": 0.01,
}

IMPLEMENTED_FEATURES = (
    "attacker_similarity",
    "victim_similarity",
    "temporal_similarity",
    "technique_similarity",
    "chain_similarity",
)