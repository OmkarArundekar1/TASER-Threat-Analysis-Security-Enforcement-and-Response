import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# NOTE: deliberately reads the same env vars as ../config.py directly,
# rather than importing that module — this package's own config.py
# would collide with backend/config.py under the bare module name
# "config" (whichever gets imported into sys.modules first under that
# name wins, and the other can never be reached), causing a circular
# self-import.
STIX_FILE = (
    BASE_DIR
    / "mitredata"
    / "attack-stix-data"
    / "enterprise-attack"
    / "enterprise-attack.json"
)

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.environ.get("NEO4J_USERNAME", "")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")