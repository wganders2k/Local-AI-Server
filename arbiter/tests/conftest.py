import os
import sys

# The modules under test import each other by bare name, as they do at runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
