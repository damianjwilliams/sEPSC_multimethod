"""Classical threshold detection -- the Synaptosoft Mini Analysis method.

Local maxima at least (c) apart, each accepted on amplitude against a baseline
window sitting (d) before the peak and spanning (e). This is the method every
other detector here is measured against, and the one whose parameters are
meant to be tuned per recording.

Its two structural weaknesses are worth stating, because the consensus table
will show them:

  * `find_peaks(distance=...)` keeps only the TALLEST peak in each window, so a
    small event riding near a large one is never a candidate at all -- no
    threshold can recover it.
  * the baseline window is at a fixed offset, so at short inter-event intervals
    it sits on the previous event's decay tail. Noise on that tail then reads
    as extra local maxima that clear every gate: one event reported as several.

Both are why `deconvolution.py` exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..trace import Trace, robust_sd
from .base import Detector


@dataclass(frozen=True)
class ThresholdParams:
    amplitude_threshold: float = 0.0
    """Minimum |amplitude| in trace units. 0 means "use amplitude_sd instead"."""

    amplitude_sd: float = 5.0
    """Minimum |amplitude| in noise SDs, used when amplitude_threshold is 0.

    In SDs rather than pA because an absolute threshold does not transfer
    between cells. Note 4 SD is the detection LIMIT reported by Greger &
    Watson (J Physiol 2025); 5 is used here because at 4 this detector's
    precision on planted events is 0.44 -- it is at the limit, not above it.
    """

    search_local_max_ms: float = 8.0
    """(c) Minimum spacing between candidate peaks.

    The Mini Analysis tutorial suggests ~3 ms, and that is what the sibling
    `minianalysis` repo defaults to. 8 ms is used here deliberately: at 3 ms
    the window is far shorter than an event's decay, so noise on the decay
    tail registers as further local maxima and one event is reported several
    times. On planted events that single change moves mean F1 from 0.603 to
    0.900 (precision 0.44 -> 0.93). Set it to 3.0 if you specifically want to
    reproduce the tutorial's behaviour.
    """

    direction: str = "negative"
    """Event polarity. "negative" for inward currents -- sEPSCs and
    sIPSCs in voltage clamp. "positive" for outward events."""
    subtract_drift: bool = True
    """Detect against a slow rolling baseline rather than the raw level.
    On a drifting trace this is what stops a slope from adding or
    removing events at one end of the recording."""
    drift_win_ms: float = 300.0
    """Window for that rolling baseline, ms. See Measurement."""


class ThresholdDetector(Detector):
    name = "threshold"
    label = "Threshold (Mini Analysis)"
    blurb = ("Local maxima at least (c) apart, kept if they clear an amplitude "
             "threshold. Fully explainable, tuned per recording.")
    params_cls = ThresholdParams

    def find(self, trace: Trace, params: ThresholdParams) -> np.ndarray:
        sv = self._upright(trace, params.direction, params.subtract_drift,
                           params.drift_win_ms)
        thr = params.amplitude_threshold
        if thr <= 0:
            thr = params.amplitude_sd * robust_sd(sv)
        idx, _ = find_peaks(sv, height=thr,
                            distance=trace.samples(params.search_local_max_ms))
        return idx
