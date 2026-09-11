"""How much of a genotype difference can noise alone manufacture?

One cell, one parameter set. The only thing that changes is how much noise the
recording carries. Noise is added to the RAW trace and then passed through the
package's own 3 kHz Bessel + 10 kHz decimation, so the added noise is shaped
exactly like real recording noise rather than sitting on top of the filter.
"""

import dataclasses
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
PRE = Preprocessing()                      # 3 kHz Bessel order 8 -> 10 kHz
TARGETS = [0.0, 4.0, 8.0, 12.0, 15.0, 18.0, 22.0]  # raw white-noise SD, pA

# Built-in defaults deliberately: min_amp_abs is 0 there, so nothing but the
# detector's own sigma-relative threshold is acting. That is the normal case
# for anyone who has not calibrated.
S = Settings()
raw = load(ABF, 0, Preprocessing(enabled=False))
rng = np.random.default_rng(0)


def preprocess(v):
    f = bessel_lowpass(v, raw.fs, PRE.cutoff_hz, PRE.order)
    dec, fs = decimate_to(f, raw.fs, PRE.target_rate_hz)
    dt = 1.0 / fs
    return Trace(t=np.arange(dec.size) * dt, v=dec, dt=dt, unit=raw.unit,
                 abf_path=ABF, channel=0, preprocessing=PRE,
                 hardware_filter_hz=raw.hardware_filter_hz)


rows = []
for add_sd in TARGETS:
    v = np.asarray(raw.v, dtype=float)
    if add_sd:
        v = v + rng.normal(0.0, add_sd, v.size)
    tr = preprocess(v)
    sigma = noise_sd(tr, S.measurement.direction)
    row = {"added_raw_sd": add_sd, "sigma_pA": sigma, "limit_4sd": 4 * sigma}
    for method in ("threshold", "deconvolution"):
        det = detectors.get(method)
        idx = det.find(tr, S.params_for(method))
        events, _ = measure_all(tr, idx, S.measurement, method)
        amps = np.abs([e.amplitude for e in events])
        row[f"{method}_n"] = len(events)
        row[f"{method}_hz"] = len(events) / tr.duration_s
        row[f"{method}_median_amp"] = float(np.median(amps)) if len(amps) else np.nan
        # The same events, kept only if they clear a floor shared across the
        # whole study -- 4 sigma of the noisiest cell (0010, sigma 2.78).
        keep = amps >= 11.13
        row[f"{method}_hz_shared_floor"] = keep.sum() / tr.duration_s
        row[f"{method}_median_amp_shared"] = (float(np.median(amps[keep]))
                                              if keep.any() else np.nan)
    rows.append(row)

t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("SAME CELL, SAME PARAMETERS, ONLY THE NOISE DIFFERS")
print("(0006 natural sigma 1.40 pA; 0010, the noisiest real cell, is 2.78)")
print()
print(t.round(2).to_string(index=False))

base = t.iloc[0]
print()
for m in ("threshold", "deconvolution"):
    near = t.iloc[(t.sigma_pA - 2.78).abs().argmin()]
    assert abs(near.sigma_pA - 2.78) < 0.25, (
        f"no level reached 0010's sigma; closest was {near.sigma_pA:.2f}")
    print(f"{m}:")
    print(f"  at its own sigma 1.40 -> {base[m + '_hz']:.2f} Hz, "
          f"median |amp| {base[m + '_median_amp']:.1f} pA")
    print(f"  made as noisy as 0010 (sigma {near.sigma_pA:.2f}) -> "
          f"{near[m + '_hz']:.2f} Hz, median |amp| {near[m + '_median_amp']:.1f} pA")
    d = 100 * (near[m + "_hz"] - base[m + "_hz"]) / base[m + "_hz"]
    print(f"  apparent frequency change from noise alone: {d:+.0f}%")
    b, n = base[m + "_hz_shared_floor"], near[m + "_hz_shared_floor"]
    d2 = 100 * (n - b) / b if b else float("nan")
    print(f"  with the shared 11.13 pA floor: {b:.2f} -> {n:.2f} Hz ({d2:+.0f}%)")
    print()
