"""Put the repo root on sys.path so `import multimethod` works from `pytest`
at the repo root, without needing an editable install."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
