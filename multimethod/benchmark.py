"""Planted-event ground truth, and scoring every detector against it.

Real recordings have no ground truth, so a claim like "method A is better than
method B" can only be made against events whose times you already know. This
plants biexponential events at known times in noise of a known character and
scores each detector's precision, recall and F1.

Read the numbers with the standard caveat: planted biexponential events flatter
a detector whose template or kernel is biexponential, which is all of them
here. The benchmark is for catching a detector that is outright broken, and for
comparing a change against the same fixture -- not for settling which method is
best on your own data. For that, run them all and use the review window.

    python -m multimethod benchmark
"""

from __future__ import annotations

import numpy as np

from . import detectors
from .events import MeasurementParams, measure_all
from .settings import Settings
from .trace import Preprocessing, Trace

DEFAULT_FS = 10_000.0


def synth(fs: float = DEFAULT_FS, rate_hz: float = 5.0, duration_s: float = 60.0,
          amp_mean: float = -20.0, amp_cv: float = 0.35, tau_ms: float = 6.0,
          rise_ms: float = 0.6, noise_sd_pa: float = 2.0, drift_pa: float = 0.0,
          pink: bool = False, seed: int = 0):
    """A Poisson train of planted events. Returns (v, true_peak_times_s).

    Events are ADDED, so they summate -- which is the point. A fixture where
    events never overlap cannot distinguish a detector that resolves
    overlapping events from one that merges them.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    n = int(duration_s * fs)
    v = np.zeros(n)

    n_ev = rng.poisson(rate_hz * duration_s)
    onsets = np.sort(rng.uniform(0.05, duration_s - 0.2, n_ev))
    amps = np.clip(rng.normal(amp_mean, abs(amp_mean) * amp_cv, n_ev), None, -1.0)
    taus = np.clip(rng.normal(tau_ms, tau_ms * 0.25, n_ev), 1.0, None)
    rise_n = max(1, int(round(rise_ms / 1000 * fs)))

    peaks = []
    for onset, a, tau in zip(onsets, amps, taus):
        oi = int(onset * fs)
        pi = oi + rise_n
        if pi + 1 >= n:
            continue
        v[oi:pi] += a * np.linspace(0, 1, rise_n, endpoint=False)
        dec_n = min(n - pi, int(10 * tau / 1000 * fs))
        v[pi:pi + dec_n] += a * np.exp(-np.arange(dec_n) * dt * 1000.0 / tau)
        peaks.append(pi * dt)

    if drift_pa:
        v += drift_pa * np.sin(2 * np.pi * np.arange(n) / n * 3)
    if noise_sd_pa:
        if pink:
            f = np.fft.rfftfreq(n, dt)
            f[0] = f[1]
            x = np.fft.irfft(np.exp(1j * rng.uniform(0, 2 * np.pi, f.size)) / np.sqrt(f), n)
            v += x / x.std() * noise_sd_pa
        else:
            v += rng.normal(0, noise_sd_pa, n)
    return v, np.array(peaks)


def synth_trace(**kw) -> tuple[Trace, np.ndarray]:
    """`synth`, wrapped as a Trace so detectors can be run on it directly."""
    fs = kw.get("fs", DEFAULT_FS)
    v, true_s = synth(**kw)
    dt = 1.0 / fs
    trace = Trace(t=np.arange(v.size) * dt, v=v, dt=dt, unit="pA",
                  abf_path="<synthetic>", channel=0,
                  preprocessing=Preprocessing(enabled=False))
    return trace, true_s


def score(true_s, det_s, tol_ms: float = 3.0) -> dict:
    """Greedy one-to-one match within tol_ms. precision / recall / f1.

    One-to-one matters: without it, a detector that reports one event three
    times would score three true positives for one real event.
    """
    true_s = np.asarray(true_s, dtype=float)
    det_s = np.asarray(det_s, dtype=float)
    if det_s.size == 0:
        return dict(tp=0, fp=0, fn=int(true_s.size),
                    precision=0.0, recall=0.0, f1=0.0)
    tol = tol_ms / 1000.0
    used = np.zeros(det_s.size, dtype=bool)
    tp = 0
    for ts in true_s:
        d = np.abs(det_s - ts)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    fp = int((~used).sum())
    fn = int(true_s.size - tp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def score_detector(method: str, trace: Trace, true_s, settings: Settings | None = None,
                   tol_ms: float = 3.0) -> dict:
    """Run one detector on a planted trace and score it."""
    s = settings or Settings()
    det = detectors.get(method)
    det.last_diagnostics = {}
    idxs = det.find(trace, s.params_for(method))
    events, _ = measure_all(trace, idxs, s.measurement, method)
    out = score(true_s, [e.peak_time_s for e in events], tol_ms)
    out["method"] = method
    out["n_detected"] = len(events)
    out["n_located"] = int(len(idxs))
    return out


CONDITIONS = [
    ("baseline 5 Hz", {}),
    ("tau 3 ms", dict(tau_ms=3.0)),
    ("tau 25 ms", dict(tau_ms=25.0)),
    ("rate 20 Hz", dict(rate_hz=20.0)),
    ("rate 40 Hz", dict(rate_hz=40.0)),
    ("pink noise", dict(pink=True)),
    ("drift 40 pA", dict(drift_pa=40.0)),
    ("small (SNR 4)", dict(amp_mean=-8.0)),
]


def run_benchmark(methods: list | None = None, settings: Settings | None = None,
                  duration_s: float = 40.0, seed: int = 1, verbose: bool = True):
    """Score the in-repo detectors across `CONDITIONS`.

    External detectors are excluded by default: they need another interpreter,
    and a 40-second subprocess per condition is not a reasonable default.
    Pass them explicitly to include them.
    """
    s = settings or Settings()
    methods = list(methods) if methods else detectors.in_repo_names()
    rows = []

    if verbose:
        head = "".join(f"{m[:11]:>13}" for m in methods)
        print(f"{'condition':>16}{head}")
        print("-" * (16 + 13 * len(methods)))

    for label, kw in CONDITIONS:
        trace, true_s = synth_trace(duration_s=duration_s, seed=seed, **kw)
        line = {}
        for m in methods:
            line[m] = score_detector(m, trace, true_s, s)
        rows.append((label, len(true_s), line))
        if verbose:
            cells = "".join(f"{line[m]['f1']:13.3f}" for m in methods)
            print(f"{label:>16}{cells}")

    if verbose:
        print("-" * (16 + 13 * len(methods)))
        means = "".join(
            f"{np.mean([r[2][m]['f1'] for r in rows]):13.3f}" for m in methods)
        print(f"{'mean F1':>16}{means}")
        prec = "".join(
            f"{np.mean([r[2][m]['precision'] for r in rows]):13.3f}" for m in methods)
        rec = "".join(
            f"{np.mean([r[2][m]['recall'] for r in rows]):13.3f}" for m in methods)
        print(f"{'mean precision':>16}{prec}")
        print(f"{'mean recall':>16}{rec}")
    return rows
