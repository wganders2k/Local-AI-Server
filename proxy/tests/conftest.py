import os
import sys

# The proxy modules import each other flatly (`from state import ...`) because
# that is how they are laid out in the container image.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config.py reads these at import time and `main` binds them by value, so they
# must be set before any test module imports either. Doing it per-test-module is
# order-dependent and silently leaves the real 180s timeout in place.
os.environ.setdefault("EXTERNAL_JOB_YIELD_TIMEOUT", "1")
os.environ.setdefault("EXTERNAL_JOB_YIELD_ON_TIMEOUT", "503")
os.environ.setdefault("IDLE_EVICT_ENABLED", "false")
os.environ.setdefault("GLANCES_URL", "")
