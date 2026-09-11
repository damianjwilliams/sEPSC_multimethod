"""What a sigma cutoff costs and buys -- on UNTRUNCATED distributions.

The event CSVs on disk were all written while multimethod.config.json was in
effect, so every amplitude in them is already >= 11.13 pA. Any question about
where the modal event sits, or how many events a floor removes, is unanswerable
from those files. So this redetects in process with the built-in defaults
(min_amp_abs = 0) and keeps everything the detector finds.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd


# The repo root, so this runs without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimethod import detectors
from multimethod.events import measure_all
from multimethod.settings import Settings
from multimethod.trace import load, noise_sd

# Run from the repo root:  python docs/<this file>.py [FOLDER]
# FOLDER holds the .abf recordings; it defaults to DATA below.
DATA = os.environ.get("SEPSC_DATA", "C:/Users/damia/sEPSC_analysis")
if len(sys.argv) > 1:
    DATA = sys.argv[1]

METHOD = "deconvolution"
MARGIN = 1.2       # empirically the floor must clear the own-threshold by ~20%

S = Settings()     # built-in defaults: min_amp_abs is 0
assert S.params_for(METHOD).min_amp_abs == 0.0, "not the built-in defaults"
SD_MULT = S.params_for(METHOD).thresh_sd

cells = []
for abf in sorted(glob.glob(os.path.join(DATA, "*.abf"))):
    name = os.path.splitext(os.path.basename(abf))[0]
    if "reduced" in name:
        continue
    tr = load(abf, 0, S.preprocessing)
    sd = noise_sd(tr, S.measurement.direction)
    idx = detectors.get(METHOD).find(tr, S.params_for(METHOD))
    ev, _ = measure_all(tr, idx, S.measurement, METHOD)
    amps = np.abs(np.array([e.amplitude for e in ev]))
    cells.append(dict(cell=name, sigma=sd, duration_s=tr.duration_s, amps=amps))
    print(f"  redetected {name}: sigma {sd:.2f}, {len(amps)} events", flush=True)

print()
print(f"PER-CELL, BUILT-IN DEFAULTS, NOTHING TRUNCATED  (method: {METHOD}, "
      f"threshold {SD_MULT:g} sigma)")
print()
rows = []
for c in cells:
    a = c["amps"]
    hist, edges = np.histogram(a, bins=40)
    mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    rows.append({"cell": c["cell"][-4:], "sigma": c["sigma"],
                 "own_thresh_pA": SD_MULT * c["sigma"],
                 "n": len(a), "rate_hz": len(a) / c["duration_s"],
                 "mode_pA": mode, "median_pA": float(np.median(a)),
                 "frac_below_20pA": float((a < 20).mean())})
print(pd.DataFrame(rows).round(2).to_string(index=False))

print()
print("WHAT EACH EXCLUSION RULE COSTS AND BUYS")
print()
out = []
order = sorted(cells, key=lambda c: c["sigma"])
for k in range(len(order), 0, -1):
    kept = order[:k]
    sigma_max = max(c["sigma"] for c in kept)
    floor = MARGIN * SD_MULT * sigma_max
    total = sum(len(c["amps"]) for c in kept)
    surviving = sum(int((c["amps"] >= floor).sum()) for c in kept)
    modes = []
    for c in kept:
        h, e = np.histogram(c["amps"], bins=40)
        modes.append(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    out.append({
        "cells_kept": k,
        "excluded": ", ".join(c["cell"][-4:] for c in order[k:]) or "-",
        "sigma_max": sigma_max,
        "floor_needed_pA": floor,
        "events_before": total,
        "events_after": surviving,
        "retained_pct": 100 * surviving / total if total else np.nan,
        "floor_vs_mode": ("above" if floor > min(modes) else "below"),
    })
print(pd.DataFrame(out).round(2).to_string(index=False))
