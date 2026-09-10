"""The event record, and the one measurement routine every detector shares.

The central design decision of this package
-------------------------------------------
A detector's job here is only to say WHERE an event is. It does not report the
amplitude, rise, decay or area -- `measure_at` does, identically, for every
method.

That split matters. Each detector has its own idea of what an amplitude is:
the classical threshold detector averages n samples at the peak against a
fixed pre-peak window, miniML reports its own baseline convention, a template
method reports a fitted scale factor. If each method reported its own numbers,
the amplitude columns of the consensus table would not be comparable, and a
sentence like "method A finds smaller events than method B" could be an
artefact of two measurement conventions rather than a fact about detection.

So: detectors return peak indices. `measure_at` turns an index into an Event
using one shared MeasurementParams. Cross-method amplitude comparison is then
meaningful, and a disagreement is a disagreement about detection alone.

The cost is that a detector's own amplitude estimate is discarded. Where one is
informative it is kept in `DetectorResult.diagnostics` instead of the Event.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trace import Trace

# numpy renamed trapz -> trapezoid in 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Columns of a per-method events CSV, in order.
EVENT_COLUMNS = [
    "peak_idx", "peak_time_s", "baseline", "amplitude",
    "rise_time_ms", "decay_time_ms", "area", "inter_event_interval_ms",
]


@dataclass(frozen=True)
class MeasurementParams:
    """How an event is quantified once a detector has located it.

    Shared by every method, and deliberately NOT per-method: see the module
    docstring. Definitions match the Synaptosoft Mini Analysis conventions so
    numbers are comparable with the sibling `minianalysis` repo.
    """

    direction: str = "negative"
    """"negative" for inward currents (sEPSCs in voltage clamp), else "positive"."""

    n_avg_peak: int = 3
    """Samples averaged at the peak. Reduces noise; biases fast peaks slightly low."""

    baseline_before_ms: float = 3.0
    """Gap between the end of the baseline window and the peak."""

    baseline_avg_ms: float = 3.0
    """Length of the baseline window."""

    onset_fraction: float = 0.1
    """Onset is where the rise crosses this fraction of the peak span."""

    onset_search_ms: float = 10.0
    """How far back to look for that crossing before giving up on the event."""

    decay_fraction: float = 0.5
    """Decay time is measured to this fraction of the peak span (0.5 = half-decay)."""

    decay_search_ms: float = 50.0
    """How far forward to look for that crossing before giving up on the event."""

    subtract_drift: bool = True
    """Measure against a slow rolling-percentile baseline rather than raw level.
    Without this, a drifting trace biases every amplitude by slope x window."""

    drift_win_ms: float = 300.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MeasurementParams":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class Event:
    """One measured event. `method` is filled in by the runner, not the detector."""

    peak_idx: int
    peak_time_s: float
    baseline: float
    amplitude: float          # signed: negative for inward events
    rise_time_ms: float
    decay_time_ms: float
    area: float               # signed, trace-units x ms, onset..decay only
    method: str = ""

    def as_row(self) -> dict:
        return {c: getattr(self, c) for c in EVENT_COLUMNS if c != "inter_event_interval_ms"}


def measure_at(v: np.ndarray, dt: float, peak_idx: int,
               params: MeasurementParams,
               drift: np.ndarray | None = None) -> Event | None:
    """Measure the event at `peak_idx`. None if it cannot be measured there.

    Returning None is not a rejection of the detector's call -- it means this
    routine could not put numbers on it (no onset crossing within the search
    window, the peak is too close to the start of the recording, the deflection
    runs the wrong way). The runner counts these separately so a detector is
    not silently blamed for events that were merely unmeasurable.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    pol = -1.0 if params.direction == "negative" else 1.0

    if drift is not None:
        vv = v - drift
    else:
        vv = v
    sv = pol * vv

    avg_n = max(1, params.n_avg_peak)
    a0 = max(0, peak_idx - avg_n // 2)
    a1 = min(n, a0 + avg_n)
    if a1 <= a0:
        return None
    peak_v = float(np.mean(vv[a0:a1]))

    b1 = peak_idx - max(1, int(round(params.baseline_before_ms / 1000.0 / dt)))
    b0 = b1 - max(1, int(round(params.baseline_avg_ms / 1000.0 / dt)))
    if b0 < 0:
        return None
    baseline = float(np.mean(vv[b0:b1]))

    amplitude = peak_v - baseline
    span = pol * peak_v - pol * baseline
    if span <= 0:
        return None                       # deflection runs the wrong way

    s_base = pol * baseline
    onset_level = s_base + params.onset_fraction * span
    decay_level = s_base + (1.0 - params.decay_fraction) * span

    lo = max(0, peak_idx - max(1, int(round(params.onset_search_ms / 1000.0 / dt))))
    below = np.nonzero(sv[lo:peak_idx + 1] < onset_level)[0]
    if below.size == 0:
        return None
    onset_idx = lo + int(below[-1])

    hi = min(n, peak_idx + max(1, int(round(params.decay_search_ms / 1000.0 / dt))))
    at = np.nonzero(sv[peak_idx:hi] <= decay_level)[0]
    if at.size == 0:
        return None
    decay_idx = peak_idx + int(at[0])

    area = float(_trapz(vv[onset_idx:decay_idx + 1] - baseline, dx=dt * 1e3))
    return Event(
        peak_idx=int(peak_idx),
        peak_time_s=float(peak_idx * dt),
        baseline=baseline,
        amplitude=float(amplitude),
        rise_time_ms=float((peak_idx - onset_idx) * dt * 1e3),
        decay_time_ms=float((decay_idx - peak_idx) * dt * 1e3),
        area=area,
    )


def measure_all(trace: Trace, peak_idxs, params: MeasurementParams,
                method: str = "") -> tuple[list[Event], int]:
    """Measure every index. Returns (events, n_unmeasurable).

    Computes the drift baseline once for the whole trace rather than per event,
    which is what makes this affordable on a ten-minute recording.
    """
    from .trace import track_baseline

    idxs = np.asarray(sorted({int(i) for i in np.asarray(peak_idxs).ravel()}), dtype=int)
    if idxs.size == 0:
        return [], 0

    drift = None
    if params.subtract_drift:
        pct = 80.0 if params.direction == "negative" else 20.0
        drift = track_baseline(trace.v, trace.dt, params.drift_win_ms, pct)

    events: list[Event] = []
    unmeasurable = 0
    for i in idxs:
        if not (0 <= i < trace.v.size):
            unmeasurable += 1
            continue
        ev = measure_at(trace.v, trace.dt, int(i), params, drift)
        if ev is None:
            unmeasurable += 1
            continue
        ev.method = method
        events.append(ev)
    return events, unmeasurable


def events_frame(events) -> pd.DataFrame:
    """Events -> tidy DataFrame, sorted by time, with inter-event intervals.

    An empty frame still carries every column, so downstream code can read a
    zero-event result without special-casing it.
    """
    events = list(events)
    if not events:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in EVENT_COLUMNS})
    df = pd.DataFrame([e.as_row() for e in events])
    df = df.sort_values("peak_time_s").reset_index(drop=True)
    df["inter_event_interval_ms"] = df["peak_time_s"].diff() * 1e3
    df["peak_idx"] = df["peak_idx"].astype("int64")
    return df[EVENT_COLUMNS]


def read_events_frame(path: str) -> pd.DataFrame:
    """Read an events CSV back exactly as written.

    float_precision="round_trip" is not optional: pandas' default C float
    parser is not correctly rounded, and every window the review window draws
    is derived from the baseline and amplitude columns. A level that lands an
    ULP away from a sample sitting right at it moves a marker to a different
    sample -- i.e. the window would draw something the detector did not do,
    on exactly the marginal events an operator looks at hardest.
    """
    return pd.read_csv(path, float_precision="round_trip")
