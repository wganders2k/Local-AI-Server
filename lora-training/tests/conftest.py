import os
import sys

# Same trick as proxy/tests/conftest.py — the modules under test import each
# other by bare name, so the package root has to be importable directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
