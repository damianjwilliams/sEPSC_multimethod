"""Template matching -- the Clements & Bekkers (1997) optimally-scaled template.

At every position, fit the template to the trace by least squares (scale plus
offset, both free), then form the detection criterion

    criterion = scale / standard_error_of_scale

and threshold it. Because the scale is fitted rather than correlated, the
criterion is largely independent of event amplitude, which is the property that
makes a single threshold work across a recording -- unlike a plain
cross-correlation, whose peak height scales with the event.

This is the method Clampfit implements. It is included here as the third
independent in-repo detector because it fails differently from the other two:
it is a matched filter, so it is strong on small events buried in noise and
weak on events whose kinetics differ from the template -- the opposite balance
to the threshold detector.

Implementation note: the closed-form solution needs five running sums over the
template window. All five are computed as convolutions with `np.convolve`
rather than in a Python loop, which is what makes this run in about a second on
a ten-minute recording instead of minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..trace import Trace
from .base import Detector
from .deconvolution import biexp_kernel, estimate_kernel


@dataclass(frozen=True)
class TemplateParams:
    criterion_threshold: float = 14.0
    """Threshold on scale/SE(scale).

    Clements & Bekkers suggest 3-5, and that is right for their white-noise
    case. It is much too permissive here: the SE in the denominator assumes
    independent residuals, and this package low-pass filters by default, which
    correlates adjacent samples and inflates the criterion. On planted events
    with 1/f noise, a threshold of 4 gives F1 0.16; 14 gives 0.76. Lower it if
    you work on unfiltered traces.
    """

    min_sep_ms: float = 15.0
    """Minimum spacing between accepted criterion peaks.

    Not 3 ms: the criterion peak is broad, so one event produces several
    crossings and gets reported repeatedly. Widening this to roughly an
    event's duration moves mean F1 from 0.586 to 0.782. The cost is real --
    two genuinely distinct events closer than this merge, which is why F1 at
    20 Hz is slightly worse than at 8 ms. Lower it for high event rates.
    """

    use_estimated_template: bool = True
    """Build the template from this recording's own well-isolated events
    (the same estimator the deconvolution detector uses). Falls back to the
    analytic biexponential below if too few isolated events are found."""

    tau_rise_ms: float = 0.6
    tau_decay_ms: float = 6.0
    template_ms: float = 30.0
    """Length of the analytic template, when the estimated one is not used."""

    direction: str = "negative"
    subtract_drift: bool = True
    drift_win_ms: float = 300.0


def detection_criterion(sv: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Clements & Bekkers scale/SE(scale), at every sample.

    For a window y and template e, fitting y ~ scale*e + offset gives

        scale  = (Sey - Se*Sy/N) / (See - Se*Se/N)
        SSE    = Syy - scale^2*See - 2*scale*offset*Se - ... (expanded below)
        SE     = sqrt(SSE / (N-2) / (See - Se*Se/N))

    All the y-dependent sums are convolutions of y or y^2 with a flipped
    template or a box, so the whole criterion is four convolutions.
    """
    e = np.asarray(template, dtype=float)
    n = e.size
    if n < 4 or sv.size < n:
        return np.zeros(sv.size)

    Se = float(e.sum())
    See = float((e * e).sum())
    denom = See - Se * Se / n
    if denom <= 0:
        return np.zeros(sv.size)

    y = np.asarray(sv, dtype=float)
    box = np.ones(n)
    # 'valid' convolutions, then padded so index i is the window STARTING at i
    Sy = np.convolve(y, box, mode="valid")
    Syy = np.convolve(y * y, box, mode="valid")
    Sey = np.convolve(y, e[::-1], mode="valid")

    scale = (Sey - Se * Sy / n) / denom
    offset = (Sy - scale * Se) / n
    sse = (Syy + scale * scale * See + n * offset * offset
           - 2.0 * (scale * Sey + offset * Sy - scale * offset * Se))
    sse = np.clip(sse, 0.0, None)
    se = np.sqrt(sse / max(n - 2, 1) / denom)

    with np.errstate(divide="ignore", invalid="ignore"):
        crit = np.where(se > 0, scale / se, 0.0)
    crit = np.nan_to_num(crit, nan=0.0, posinf=0.0, neginf=0.0)

    # Shift so the criterion sits at the template's own peak, not at the start
    # of the window -- otherwise every detection is reported one rise-time early.
    peak_off = int(np.argmax(e))
    out = np.zeros(y.size)
    end = min(y.size, peak_off + crit.size)
    out[peak_off:end] = crit[:end - peak_off]
    return out


class TemplateDetector(Detector):
    name = "template"
    label = "Template match (Clements-Bekkers)"
    blurb = ("Fits an optimally-scaled template at every sample and thresholds "
             "scale/SE. Strong on small events, weak on odd kinetics.")
    params_cls = TemplateParams

    def find(self, trace: Trace, params: TemplateParams) -> np.ndarray:
        p = params
        sv = self._upright(trace, p.direction, p.subtract_drift, p.drift_win_ms)
        if sv.size < 16:
            return np.array([], dtype=int)

        template, source = None, "analytic"
        if p.use_estimated_template:
            from .deconvolution import DeconvolutionParams
            template, n_iso = estimate_kernel(
                sv, trace.dt, DeconvolutionParams(direction=p.direction))
            if template is not None:
                source = f"estimated ({n_iso} isolated events)"
        if template is None:
            template = biexp_kernel(trace.dt, p.tau_rise_ms, p.tau_decay_ms,
                                    p.template_ms)

        crit = detection_criterion(sv, template)
        idx, _ = find_peaks(crit, height=p.criterion_threshold,
                            distance=trace.samples(p.min_sep_ms))

        self.last_diagnostics = {
            "criterion": crit, "template": template, "template_source": source,
            "template_ms": template.size * trace.dt * 1e3,
            "n_candidates": int(idx.size),
        }
        if idx.size == 0:
            return np.array([], dtype=int)

        # snap to the local maximum of the real trace, as the other detectors do
        r = trace.samples(2.0)
        snapped = {max(0, i - r) + int(np.argmax(sv[max(0, i - r):min(sv.size, i + r + 1)]))
                   for i in idx}
        return np.array(sorted(snapped), dtype=int)
