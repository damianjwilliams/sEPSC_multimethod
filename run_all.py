#!/usr/bin/env python
"""Run every enabled detector on a recording and write the consensus table.

Thin wrapper, so this is runnable as a plain script from the repo root:

    python run_all.py recording.abf
    python run_all.py recording.abf --gui        # parameter window first
    python run_all.py recording.abf --review     # open the review window after

Identical to `python -m multimethod run`; see --help for the options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multimethod.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run"] + sys.argv[1:]))
