"""Put the repository root on sys.path so `lib` is importable from scripts/."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
for _d in (DATA, RESULTS, LOGS):
    os.makedirs(_d, exist_ok=True)


def strip_private(rec):
    """Drop the non-JSON helper entries that run_lta attaches to its record."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}
