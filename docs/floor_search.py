"""At what shared amplitude floor does a noise difference stop mattering?

A post-hoc floor can only discard events a detector already found. It cannot
recover events the detector never located -- so if a cell's own sigma-relative
detection threshold sits ABOVE the shared floor, the floor is not binding and
the noisy cell still loses real events. This finds where that stops being true.
"""

import os
import sys

import numpy as np
import pandas as pd


# The repo root, so this runs without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimethod import detectors
from multimethod.events import measure_all
from multimethod.settings import Settings
from multimethod.trace import (Preprocessing, Trace, bessel_lowpass, decimate_to,
                               load, noise_sd)

# Run from the repo root:  python docs/<this file>.py [FOLDER]
# FOLDER holds the .abf recordings; it defaults to DATA below.
DATA = os.environ.get("SEPSC_DATA", "C:/Users/damia/sEPSC_analysis")
if len(sys.argv) > 1:
    DATA = sys.argv[1]

ABF = os.path.join(DATA, "2026_08_17_0006.abf")
PRE = Preprocessing()
S = Settings()
raw = load(ABF, 0, Preprocessing(enabled=False))
rng = np.random.default_rng(0)


def version(add_sd):
    v = np.asarray(raw.v, dtype=float)
    if add_sd:
        v = v + rng.normal(0.0, add_sd, v.size)
    f = bessel_lowpass(v, raw.fs, PRE.cutoff_hz, PRE.order)
    dec, fs = decimate_to(f, raw.fs, PRE.target_rate_hz)
    dt = 1.0 / fs
    tr = Trace(t=np.arange(dec.size) * dt, v=dec, dt=dt, unit=raw.unit,
               abf_path=ABF, channel=0, preprocessing=PRE,
               hardware_filter_hz=raw.hardware_filter_hz)
    return tr, noise_sd(tr, S.measurement.direction)


quiet, sd_q = version(0.0)       # as recorded, sigma 1.40
noisy, sd_n = version(15.0)      # same cell at 0010's sigma, ~2.76

print(f"quiet sigma {sd_q:.2f} pA      noisy sigma {sd_n:.2f} pA "
      f"({sd_n / sd_q:.2f}x)")
print()

FLOORS = [0, 6, 8, 10, 11.13, 13, 15, 16.7, 18, 20, 22, 25]
for method, sd_mult in (("threshold", S.params_for("threshold").amplitude_sd),
                        ("deconvolution", S.params_for("deconvolution").thresh_sd)):
    det = detectors.get(method)
    amps = {}
    for tag, tr in (("quiet", quiet), ("noisy", noisy)):
        idx = det.find(tr, S.params_for(method))
        ev, _ = measure_all(tr, idx, S.measurement, method)
        amps[tag] = np.abs([e.amplitude for e in ev])
    dur = quiet.duration_s

    print(f"--- {method}  (its own threshold is {sd_mult:g} sigma) ---")
    print(f"    that threshold is {sd_mult * sd_q:5.2f} pA in the quiet version "
          f"and {sd_mult * sd_n:5.2f} pA in the noisy one")
    rows = []
    for fl in FLOORS:
        q = (amps["quiet"] >= fl).sum() / dur
        n = (amps["noisy"] >= fl).sum() / dur
        rows.append({"floor_pA": fl, "quiet_hz": q, "noisy_hz": n,
                     "error_pct": 100 * (n - q) / q if q else np.nan})
    t = pd.DataFrame(rows)
    print(t.round(2).to_string(index=False))
    ok = t[t.error_pct.abs() <= 5]
    if len(ok):
        print(f"    -> within 5%: floors from {ok.floor_pA.min():g} pA up")
    print()
