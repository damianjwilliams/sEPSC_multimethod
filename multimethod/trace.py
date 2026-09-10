"""Loading and preprocessing a recording, once, for every detector.

Every detector in this package sees the SAME trace object. That is deliberate:
if two detectors ran on differently filtered or differently decimated data,
their event times would not be comparable and the consensus table in
`consensus.py` would be meaningless. So preprocessing happens here, once, and
`Trace` is passed around read-only.

The one exception is a detector that runs in another interpreter (see
detectors/external.py). Those cannot share a numpy array across the process
boundary, so they are handed the .abf path plus the preprocessing settings and
asked to reproduce it -- which is why `Preprocessing` is a small, explicit,
serialisable record rather than a pile of keyword arguments.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

DEFAULT_CUTOFF_HZ = 3000.0
DEFAULT_TARGET_RATE_HZ = 10_000.0
DEFAULT_ORDER = 8


@dataclass(frozen=True)
class Preprocessing:
    """What was done to the raw sweep before any detector saw it.

    Defaults match the rest of these analysis scripts (3 kHz Bessel low-pass,
    decimate to 10 kHz), so event counts are comparable with runs made by the
    sibling `minianalysis` repo.
    """

    enabled: bool = True
    cutoff_hz: float = DEFAULT_CUTOFF_HZ
    target_rate_hz: float = DEFAULT_TARGET_RATE_HZ
    order: int = DEFAULT_ORDER

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessing":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def suffix(self) -> str:
        """Filename fragment recording these settings, so two differently
        preprocessed runs cannot overwrite each other's output."""
        if not self.enabled:
            return ""
        return f"_filt{int(self.cutoff_hz)}Hz{int(self.target_rate_hz)}Hz"

    def describe(self) -> str:
        if not self.enabled:
            return "raw, unfiltered"
        return (f"Bessel {self.cutoff_hz:.0f} Hz order {self.order} "
                f"-> {self.target_rate_hz:.0f} Hz")


@dataclass(frozen=True)
class Trace:
    """One channel of one sweep, after preprocessing.

    `v` is read-only: detectors must not modify the array they are handed, or
    a later detector in the same run would see something different.
    """

    t: np.ndarray
    v: np.ndarray
    dt: float
    unit: str
    abf_path: str
    channel: int
    preprocessing: Preprocessing
    hardware_filter_hz: float | None = None

    def __post_init__(self):
        self.v.setflags(write=False)
        self.t.setflags(write=False)

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def duration_s(self) -> float:
        return self.v.size * self.dt

    def samples(self, ms: float) -> int:
        """Milliseconds to samples, at least 1."""
        return max(1, int(round(ms / 1000.0 / self.dt)))


def bessel_lowpass(v: np.ndarray, fs: float, cutoff_hz: float = DEFAULT_CUTOFF_HZ,
                   order: int = DEFAULT_ORDER) -> np.ndarray:
    """Zero-phase Bessel low-pass.

    Bessel for its flat group delay, and `sosfiltfilt` for zero phase, so
    neither event shape nor event timing shifts -- both matter here, because
    peak times are the interchange currency between detectors.
    """
    from scipy.signal import bessel, sosfiltfilt

    sos = bessel(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, v)


def decimate_to(v: np.ndarray, fs: float,
                target_hz: float = DEFAULT_TARGET_RATE_HZ) -> tuple[np.ndarray, float]:
    """Strided decimation. Assumes the anti-alias filtering already happened."""
    factor = max(1, int(round(fs / target_hz)))
    return v[::factor], fs / factor


def hardware_filter_hz(abf, channel: int) -> float | None:
    """The amplifier's own filter setting, from the ABF header, if it recorded
    one. Advisory only: it is reported so you can notice when you are asking
    for a 3 kHz software filter on a trace already filtered at 2 kHz."""
    try:
        section = abf._adcSection
        if not section.nTelegraphEnable[channel]:
            return None
        return float(section.fTelegraphFilter[channel])
    except (AttributeError, IndexError, TypeError):
        return None


def load(abf_path: str, channel: int = 0,
         preprocessing: Preprocessing | None = None) -> Trace:
    """Read sweep 0 of `channel` and apply `preprocessing`.

    Sweep 0 only: these are gap-free recordings, so there is exactly one.
    """
    import pyabf

    pre = preprocessing if preprocessing is not None else Preprocessing()
    abf = pyabf.ABF(abf_path)
    if channel >= abf.channelCount:
        raise ValueError(
            f"{abf_path} has {abf.channelCount} channel(s); asked for {channel}")
    abf.setSweep(0, channel=channel)
    v = np.asarray(abf.sweepY, dtype=float)
    fs = float(abf.dataRate)
    unit = abf.adcUnits[channel] if channel < len(abf.adcUnits) else "pA"
    hw = hardware_filter_hz(abf, channel)

    if pre.enabled:
        v = bessel_lowpass(v, fs, pre.cutoff_hz, pre.order)
        v, fs = decimate_to(v, fs, pre.target_rate_hz)

    dt = 1.0 / fs
    return Trace(t=np.arange(v.size) * dt, v=v, dt=dt, unit=unit,
                 abf_path=abf_path, channel=channel, preprocessing=pre,
                 hardware_filter_hz=hw)


def output_stem(abf_path: str, channel: int = 0,
                preprocessing: Preprocessing | None = None) -> str:
    """Path prefix for every file a run writes.

    Encodes anything that makes two runs describe *different data* -- a
    different channel, a different filter setting -- so a second run cannot
    quietly overwrite the first, and so the review window can find the files
    again from the same flags.
    """
    import os

    pre = preprocessing if preprocessing is not None else Preprocessing()
    stem = os.path.splitext(abf_path)[0]
    if channel:
        stem += f"_ch{channel}"
    return stem + pre.suffix()


def robust_sd(x: np.ndarray) -> float:
    """MAD-based SD estimate, insensitive to the events themselves."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def track_baseline(v: np.ndarray, dt: float, win_ms: float = 300.0,
                   pct: float = 80.0) -> np.ndarray:
    """Slow drift estimate: rolling percentile, linearly interpolated.

    For inward (negative) events the upper percentile tracks the true baseline,
    since events only pull the trace down. `pct` is mirrored by the caller for
    outward events.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return np.zeros(0)
    w = max(3, int(round(win_ms / 1000.0 / dt)))
    step = max(1, w // 4)
    idx = np.arange(0, n, step)
    half = w // 2
    vals = np.empty(idx.size)
    for i, s in enumerate(idx):
        seg = v[max(0, s - half): s + half]
        vals[i] = np.percentile(seg, pct) if seg.size else v[s]
    if idx.size == 1:
        return np.full(n, vals[0])
    return np.interp(np.arange(n), idx, vals)


def noise_sd(trace: Trace, direction: str = "negative",
             win_ms: float = 300.0) -> float:
    """Trace noise SD, estimated from the event-FREE side of the distribution.

    Events are one-sided, so after detrending the other half of the residual is
    noise only; mirroring it gives a clean symmetric sample.

    Two tempting alternatives are both wrong here. Taking the SD of the whole
    trace is inflated by the events. Taking it from `diff(v)/sqrt(2)`
    UNDERESTIMATES by 3.6-4.9x on a low-pass-filtered trace, because filtering
    correlates adjacent samples -- and this package filters by default.
    """
    pol = -1.0 if direction == "negative" else 1.0
    d = np.asarray(trace.v, dtype=float) - track_baseline(trace.v, trace.dt, win_ms, 50.0)
    quiet = d[pol * d < 0]
    if quiet.size < 10:
        return robust_sd(d)
    return robust_sd(np.concatenate([quiet, -quiet]))
