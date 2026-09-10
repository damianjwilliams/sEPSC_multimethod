#!/usr/bin/env python
"""Open the parameter window, then run every method and review the result.

The front door. Thin wrapper, so this is runnable from the repo root:

    python launch.py
    python launch.py recording.abf     # pre-fills the dialog

Identical to `python -m multimethod launch`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multimethod.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["launch"] + sys.argv[1:]))
