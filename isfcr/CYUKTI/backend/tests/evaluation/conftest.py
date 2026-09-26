import os
import sys

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "evaluation"))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
