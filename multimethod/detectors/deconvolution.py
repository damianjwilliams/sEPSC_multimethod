"""Deconvolution detection (Pernia-Andrade et al. 2012, Biophys J 103:1429).

Deconvolving the trace by the event waveform collapses each event to a sharp
impulse. That fixes both structural weaknesses of the threshold detector at
once: summating events separate, and the decay tail that made one event look
like several is gone.

Three things here go beyond the 2012 paper:

  1. The kernel is ESTIMATED FROM THE RECORDING -- the average of well-isolated
     events, unit-peak, truncated at 8x its own 1/e decay. No time constant is
     assumed. This is the single most important detail: a fixed 6 ms kernel on
     25 ms data drops precision to ~0.51.
  2. A matching-pursuit pass accepts candidates largest-first and subtracts
     each accepted kernel from the residual, so a noise bump on an accepted
     event's decay tail has no amplitude left and is dropped.
  3. Thresholds are in units of the measured noise, not absolute, so nothing
     needs re-tuning per cell.

Known weakness, stated because the kernel is estimated rather than given:
above ~25 ms decay tau at a normal event rate, nothing is genuinely isolated
within `kernel_isolation_ms`, every "isolated" event still sits on a
predecessor's tail, and the estimated kernel length becomes unstable. Raise
`kernel_isolation_ms` for slow events; the runner warns when the estimate
looks unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..trace import Trace, robust_sd
from .base import Detector


@dataclass(frozen=True)
class DeconvolutionParams:
    thresh_sd: float = 6.0
    """Threshold on the deconvolved trace, in robust SDs of that trace."""

    reg: float = 0.01
    """Wiener regularisation lambda, as a fraction of max|K(f)|^2.

    The noise-amplification/sharpness tradeoff. Larger = smoother, blunter
    impulses, more fragments. 0.01 was best across every condition tested.
    """

    min_amp_sd: float = 2.5
    """Matching-pursuit acceptance floor, in SDs of the raw trace. Kills fragments."""

    min_amp_abs: float = 0.0
    """Optional absolute floor in trace units; 0 disables. Set to 4x the
    reported sigma to enforce the Greger & Watson detection limit explicitly."""

    min_sep_ms: float = 1.5
    smooth_ms: float = 0.3

    kernel_max_ms: float = 400.0
    kernel_tail_frac: float = 0.02
    kernel_tail_run_ms: float = 2.0
    kernel_pre_ms: float = 5.0
    kernel_isolation_ms: float = 60.0
    """An event counts as isolated if nothing else is within this. RAISE IT for
    slow events -- see the module docstring."""
    kernel_min_snr: float = 5.0
    kernel_n_max: int = 300
    kernel_smooth_ms: float = 0.4
    kernel_min_events: int = 5
    fallback_tau_rise_ms: float = 0.6
    fallback_tau_decay_ms: float = 6.0

    direction: str = "negative"
    subtract_drift: bool = True
    drift_win_ms: float = 300.0


def biexp_kernel(dt: float, tau_rise_ms: float = 0.6, tau_decay_ms: float = 6.0,
                 length_ms: float | None = None) -> np.ndarray:
    """Unit-peak biexponential PSC kernel. The analytic fallback."""
    if length_ms is None:
        length_ms = 8.0 * tau_decay_ms
    n = max(4, int(round(length_ms / 1000.0 / dt)))
    t = np.arange(n) * dt * 1000.0
    k = np.exp(-t / tau_decay_ms) - np.exp(-t / tau_rise_ms)
    pk = k.max()
    return k / pk if pk > 0 else k


def estimate_kernel(sv: np.ndarray, dt: float, p: DeconvolutionParams):
    """Average of well-isolated events, unit-peak, adaptively truncated.

    `sv` is already upright and detrended. Returns (kernel, n_isolated);
    kernel is None if too few isolated events were found.
    """
    sd = robust_sd(sv)
    if sv.size < 16 or sd <= 0:
        return None, 0
    iso_n = max(1, int(round(p.kernel_isolation_ms / 1000.0 / dt)))
    cand, props = find_peaks(sv, height=p.kernel_min_snr * sd, distance=iso_n)
    if cand.size < p.kernel_min_events:
        return None, int(cand.size)
    keep = np.argsort(props["peak_heights"])[::-1][:p.kernel_n_max]
    cand = np.sort(cand[keep])

    pre_n = max(2, int(round(p.kernel_pre_ms / 1000.0 / dt)))
    post_n = max(8, int(round(p.kernel_max_ms / 1000.0 / dt)))
    segs = []
    for i, q in enumerate(cand):
        if q - pre_n < 0 or q + post_n >= sv.size:
            continue
        if i > 0 and q - cand[i - 1] < iso_n:
            continue
        if i < cand.size - 1 and cand[i + 1] - q < iso_n:
            continue
        seg = sv[q - pre_n: q + post_n].astype(float)
        seg -= np.median(seg[:pre_n])
        if seg[pre_n] <= 0:
            continue
        segs.append(seg / seg[pre_n])
    if len(segs) < p.kernel_min_events:
        return None, len(segs)

    kern = np.mean(segs, axis=0)[pre_n:].copy()
    sm = max(1, int(round(p.kernel_smooth_ms / 1000.0 / dt)))
    if sm > 1:
        kern = np.convolve(kern, np.ones(sm) / sm, mode="same")
    pk = kern.max()
    if pk <= 0:
        return None, len(segs)

    # Adaptive length. Primary criterion is the 1/e crossing, kept at 8*tau,
    # because it is robust to a constant offset in the average -- and there IS
    # one at long decay times, where no event is truly isolated so the pre-peak
    # median over-subtracts. Keying off 2%-of-peak under that offset truncates
    # the kernel far too early (measured 7-18 ms kernels on 60 ms decays).
    e_cross = np.nonzero(kern <= pk / np.e)[0]
    tau_n = int(e_cross[0]) if e_cross.size else kern.size
    cut = min(kern.size, max(4, 8 * max(tau_n, 1)))

    # Secondary cap: the tail must also STAY below tail_frac for tail_run_ms,
    # so a genuinely short kernel is not padded out with noise.
    run_n = max(2, int(round(p.kernel_tail_run_ms / 1000.0 / dt)))
    low = (kern < p.kernel_tail_frac * pk).astype(int)
    csum = np.concatenate(([0], np.cumsum(low)))
    for i in range(max(0, kern.size - run_n)):
        if csum[i + run_n] - csum[i] == run_n:
            cut = min(cut, max(i, 8 * max(tau_n, 1)))
            break
    cut = min(max(cut, max(4, int(round(2.0 / 1000.0 / dt)))), kern.size)

    kern = kern[:cut].copy()
    taper = np.linspace(1.0, 0.0, max(2, kern.size // 5))
    kern[-taper.size:] *= taper
    kern = np.clip(kern, 0.0, None)
    mx = kern.max()
    return (kern / mx, len(segs)) if mx > 0 else (None, len(segs))


def wiener_deconvolve(v: np.ndarray, kernel: np.ndarray, reg: float = 0.01,
                      smooth_n: int = 0) -> np.ndarray:
    """S(f) = Y(f) K*(f) / (|K(f)|^2 + lambda), lambda = reg * max|K|^2."""
    n = v.size
    nfft = 1 << int(np.ceil(np.log2(n + kernel.size)))
    Y = np.fft.rfft(v, nfft)
    K = np.fft.rfft(kernel, nfft)
    K2 = (K.conj() * K).real
    s = np.fft.irfft(Y * K.conj() / (K2 + reg * K2.max()), nfft)[:n]
    if smooth_n > 1:
        s = np.convolve(s, np.ones(smooth_n) / smooth_n, mode="same")
    return s


def peel_refine(sv: np.ndarray, dt: float, cand: np.ndarray, kernel: np.ndarray,
                heights: np.ndarray, min_amp: float,
                pre_ms: float = 3.0, avg_ms: float = 0.3) -> np.ndarray:
    """Matching pursuit: accept largest-first, subtracting each accepted kernel.

    Returns the accepted indices in time order.
    """
    resid = sv.astype(float).copy()
    klen = kernel.size
    pre_n = max(1, int(round(pre_ms / 1000.0 / dt)))
    avg_n = max(1, int(round(avg_ms / 1000.0 / dt)))
    accepted = []
    for j in np.argsort(heights)[::-1]:
        q = int(cand[j])
        a0 = max(0, q - avg_n // 2)
        peak_v = float(np.mean(resid[a0:min(resid.size, a0 + avg_n)]))
        b1 = q - pre_n
        b0 = max(0, b1 - pre_n)
        base = float(np.median(resid[b0:b1])) if b1 > b0 else 0.0
        amp = peak_v - base
        if amp < min_amp:
            continue
        accepted.append(q)
        e1 = min(resid.size, q + klen)
        resid[q:e1] -= amp * kernel[:e1 - q]
    return np.array(sorted(accepted), dtype=int)


class DeconvolutionDetector(Detector):
    name = "deconvolution"
    label = "Deconvolution (Pernia-Andrade)"
    blurb = ("Estimates the event waveform from this recording, deconvolves by "
             "it so each event becomes an impulse. Nothing to tune per cell.")
    params_cls = DeconvolutionParams

    def find(self, trace: Trace, params: DeconvolutionParams) -> np.ndarray:
        p = params
        sv = self._upright(trace, p.direction, p.subtract_drift, p.drift_win_ms)
        if sv.size < 16:
            return np.array([], dtype=int)

        kern, n_iso = estimate_kernel(sv, trace.dt, p)
        source = "estimated"
        if kern is None:
            kern = biexp_kernel(trace.dt, p.fallback_tau_rise_ms,
                                p.fallback_tau_decay_ms)
            source = "analytic-fallback"

        raw_sd = robust_sd(sv)
        s = wiener_deconvolve(sv, kern, p.reg, trace.samples(p.smooth_ms))
        dsd = robust_sd(s)
        idx, props = find_peaks(s, height=p.thresh_sd * dsd,
                                distance=trace.samples(p.min_sep_ms))

        self.last_diagnostics = {
            "kernel": kern, "kernel_source": source, "n_isolated": n_iso,
            "kernel_ms": kern.size * trace.dt * 1e3,
            "deconv_sd": dsd, "raw_sd": raw_sd, "n_candidates": int(idx.size),
        }
        if idx.size == 0:
            return np.array([], dtype=int)

        # snap each impulse to the nearest local maximum of the real trace
        r = trace.samples(2.0)
        snapped, heights = [], []
        for i, h in zip(idx, props["peak_heights"]):
            lo, hi = max(0, i - r), min(sv.size, i + r + 1)
            snapped.append(lo + int(np.argmax(sv[lo:hi])))
            heights.append(float(h))
        snapped = np.array(snapped, dtype=int)
        heights = np.array(heights, dtype=float)
        _, first = np.unique(snapped, return_index=True)
        snapped, heights = snapped[first], heights[first]

        if raw_sd > 0:
            min_amp = p.min_amp_sd * raw_sd
        else:
            min_amp = 0.01 * float(np.abs(sv).max()) if sv.size else 0.0
        min_amp = max(min_amp, p.min_amp_abs)

        out = peel_refine(sv, trace.dt, snapped, kern, heights, min_amp)
        self.last_diagnostics["n_after_peel"] = int(out.size)
        return out
