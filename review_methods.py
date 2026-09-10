#!/usr/bin/env python
"""Open the review window on a finished run: which method found which event.

Thin wrapper, so this is runnable from the repo root:

    python review_methods.py recording.abf
    python review_methods.py recording.abf --only-unique

Identical to `python -m multimethod review`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multimethod.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["review"] + sys.argv[1:]))
