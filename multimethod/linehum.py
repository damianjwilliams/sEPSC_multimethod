"""Removing mains hum that comes and goes.

The hum this is for is not a clean 60 Hz sine. On the rigs it was written for
it arrives in bursts of a few seconds, at up to ~15 pA, as a 60 Hz fundamental
plus a comb of odd harmonics reaching past 2.5 kHz. Left in, a burst produces
false events by the hundred -- the high harmonics have exactly the sharp edges
a deconvolution detector is looking for -- and inflates the noise estimate the
whole recording's threshold is set from, so small real events are missed even
where there is no hum.

Notch filters are the wrong tool for that. A handful of notches leaves the
upper harmonics in, and notching every harmonic up to the low-pass corner
takes a measurable bite out of every event in the recording (each notch
removes the event's own energy at that frequency), hum or no hum.

What is done instead is to use the one thing hum has that events do not: it
repeats every cycle. In each short window the trace is folded at the line
period, and the MEDIAN at each phase is the hum's waveform -- every harmonic
at once, whatever its shape. Events fall at random phases, so at any one
phase only a few cycles contain one, and the median ignores them. That
waveform is subtracted only where it is clearly larger than the noise of the
estimate itself; everywhere else the trace is returned untouched, sample for
sample.

Numpy only, and importable as a single file: the miniML worker runs in a
different interpreter and loads this module by path rather than importing the
package.
"""

from __future__ import annotations

import numpy as np

#: Analysis window, s. ~60 cycles: enough for a median per phase that is
#: much quieter than the trace, short enough to follow a burst on and off.
DEFAULT_WINDOW_S = 1.0

#: How far either side of nominal the line frequency is searched for, Hz.
#: Grid frequency wanders by well under this; a template folded at the wrong
#: period smears its upper harmonics, so it has to be measured, not assumed.
SEARCH_HZ = 0.3

#: Subtract only where the hum waveform's RMS is this many times the RMS
#: expected of a median template made from noise alone.
DEFAULT_Z = 4.0

#: Cycles in the running median that smooths the per-cycle hum amplitude.
#: One event can skew one cycle's fit; it cannot move a median of five.
GAIN_MEDIAN_CYCLES = 5


def _fold(x: np.ndarray, fs: float, f0: float, nbins: int) -> np.ndarray:
    """Phase bin of every sample, folding at period 1/f0."""
    return (np.arange(x.size) / fs * f0 % 1.0 * nbins).astype(np.int64) % nbins


def _mean_template(x: np.ndarray, phase: np.ndarray, nbins: int) -> np.ndarray:
    tot = np.bincount(phase, x, nbins)
    cnt = np.bincount(phase, None, nbins)
    return tot / np.maximum(cnt, 1)


def _median_template(x: np.ndarray, phase: np.ndarray, nbins: int) -> np.ndarray:
    vals = x[np.lexsort((x, phase))]           # by phase, then by value
    count = np.bincount(phase, None, nbins).astype(np.int64)
    first = np.concatenate(([0], np.cumsum(count)[:-1]))
    lo = first + np.maximum(count - 1, 0) // 2
    hi = first + count // 2
    top = max(vals.size - 1, 0)
    med = 0.5 * (vals[np.minimum(lo, top)] + vals[np.minimum(hi, top)])
    return np.where(count > 0, med, 0.0)


def _line_frequency(x: np.ndarray, fs: float, line_hz: float, nbins: int) -> float:
    """The fold frequency that gives the strongest template: coarse, then fine.

    The cycle MEAN is used for the search because it is cheap; the high
    harmonics make its power sharply peaked at the true frequency.
    """
    best = line_hz
    for step, half in ((0.02, SEARCH_HZ), (0.002, 0.02)):
        freqs = np.arange(best - half, best + half + step / 2, step)
        power = [np.var(_mean_template(x, _fold(x, fs, f, nbins), nbins))
                 for f in freqs]
        best = float(freqs[int(np.argmax(power))])
    return best


def _running_median(x: np.ndarray, k: int) -> np.ndarray:
    if x.size < 2 or k < 2:
        return x.copy()
    half = k // 2
    padded = np.pad(x, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1),
                     axis=1)


def _cycle_gains(x: np.ndarray, wave: np.ndarray, cycle: np.ndarray,
                 ncycles: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares size of `wave` in each cycle of `x`, and its noise."""
    num = np.bincount(cycle, x * wave, ncycles)
    den = np.bincount(cycle, wave * wave, ncycles)
    gain = num / np.maximum(den, 1e-12)
    resid = x - gain[cycle] * wave
    sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return gain, sd / np.sqrt(np.maximum(den, 1e-12))


def _template_noise(x: np.ndarray, tpl: np.ndarray, phase: np.ndarray,
                    cycles: float) -> float:
    """RMS a median-of-noise template would have: 1.2533 sd / sqrt(cycles)."""
    resid = x - tpl[phase]
    sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return 1.2533 * sd / np.sqrt(max(cycles, 1.0))


def _window_hum(x: np.ndarray, fs: float, line_hz: float, nbins: int,
                z: float) -> np.ndarray | None:
    """The hum in one detrended window, or None if there is none to speak of."""
    cycles = x.size / fs * line_hz
    # Most windows have no hum, and the frequency search below is most of the
    # cost. Folding at the nominal frequency is close enough to see hum at
    # all (the fundamental barely smears); events only ever ADD to a mean
    # template's power, so this cannot skip a window that has hum.
    phase = _fold(x, fs, line_hz, nbins)
    rough = _mean_template(x, phase, nbins)
    rough -= rough.mean()
    if np.sqrt(np.mean(rough ** 2)) <= 0.5 * z * _template_noise(x, rough, phase, cycles):
        return None

    f0 = _line_frequency(x, fs, line_hz, nbins)
    phase = _fold(x, fs, f0, nbins)
    tpl = _median_template(x, phase, nbins)
    tpl -= tpl.mean()

    # Hum must clearly exceed the noise of its own estimate.
    expected = _template_noise(x, tpl, phase, x.size / fs * f0)
    if np.sqrt(np.mean(tpl ** 2)) <= z * max(expected, 1e-12):
        return None

    t = np.arange(x.size) / fs
    cycle = (t * f0).astype(np.int64)
    ncycles = int(cycle[-1]) + 1

    def sized(template):
        gain, noise = _cycle_gains(x, template[phase], cycle, ncycles)
        gain = _running_median(gain, GAIN_MEDIAN_CYCLES)
        # Per-cycle noise is also cut by the median, but not by more than
        # sqrt(k); stay with the unsmoothed figure and a margin.
        gain[gain < z * noise] = 0.0
        return gain

    # Re-make the waveform from only the cycles the burst actually covers,
    # each scaled to unit size -- the first template was diluted by the rest.
    gain = sized(tpl)
    full = gain > 0.5 * gain.max() if gain.max() > 0 else np.zeros_like(gain, bool)
    if full.sum() >= 2 * GAIN_MEDIAN_CYCLES:
        sel = full[cycle]
        refined = _median_template(x[sel] / gain[cycle][sel], phase[sel], nbins)
        tpl = refined - refined.mean()
        gain = sized(tpl)
    if not gain.any():
        return None

    centres = (np.arange(ncycles) + 0.5) / f0
    return np.interp(t, centres, gain) * tpl[phase]


def remove_line_hum(v: np.ndarray, fs: float, line_hz: float = 60.0,
                    window_s: float = DEFAULT_WINDOW_S,
                    z: float = DEFAULT_Z) -> tuple[np.ndarray, float]:
    """`v` with line hum subtracted where there is any, and how much there was.

    Returns `(cleaned, seconds)`: `seconds` is how much of the recording had
    hum taken out, 0.0 for a clean recording -- which then comes back
    unchanged.
    """
    v = np.asarray(v, dtype=float)
    if line_hz <= 0:
        return v.copy(), 0.0
    n = v.size
    win = int(round(window_s * fs))
    nbins = max(8, int(round(fs / line_hz)))
    if n < win or win < 4 * nbins:
        return v.copy(), 0.0

    hop = win // 2
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    taper = np.hanning(win)
    ramp = np.arange(win)
    hum = np.zeros(n)
    weight = np.zeros(n)

    # Overlapping windows are cross-faded, but only windows that found hum
    # take part: one that found none has no estimate to offer, and counting it
    # would dilute its neighbour's estimate at the edge of a burst.
    for s in starts:
        seg = v[s:s + win]
        x = seg - np.polyval(np.polyfit(ramp, seg, 1), ramp)
        est = _window_hum(x, fs, line_hz, nbins, z)
        if est is not None:
            hum[s:s + win] += taper * est
            weight[s:s + win] += taper

    on = (weight > 1e-6) & (hum != 0.0)
    if not on.any():
        return v.copy(), 0.0
    out = v.copy()
    out[on] -= hum[on] / weight[on]
    return out, float(on.sum() / fs)
