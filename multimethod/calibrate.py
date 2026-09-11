"""Set this package's defaults from your own recordings, not from a fixture.

    python -m multimethod calibrate FOLDER            # report only
    python -m multimethod calibrate FOLDER --write    # and save as defaults

Every default in this package was chosen on planted synthetic events. That is
the right way to catch a broken detector, but the fixture is not your
preparation: its events are biexponential with a fixed rise, its noise is white
or 1/f by construction, and its event rate is whatever the generator was told.
Your cells have their own kinetics, their own noise floor, and their own event
density. This measures those and writes the parameters that follow from them.

Calibration, not training
-------------------------
This deliberately does NOT train a model on detected events. Training a
detector on its own output is circular: it would learn to reproduce the
detectors, errors included, and the events you most need labelled are exactly
the disputed ones the detectors cannot label. What is measured here are
properties of the RECORDINGS -- how long events take to decay, how much the
methods disagree about a peak's position, what the noise floor is -- and those
are facts about your data whichever detector found them.

To reduce the circularity that remains, only high-confidence events are
pooled: by default an event must be found by every method that ran. Those are
biased towards large, clean, isolated events, which is the right bias for
measuring kinetics and the wrong one for measuring a detection limit -- so the
detection limit comes from the noise, not from the events.

What it sets, and from what
---------------------------
    measurement.decay_search_ms      99th percentile of measured decay, +50%
    measurement.onset_search_ms      99th percentile of measured rise, +50%
    template.tau_decay_ms            median decay, converted to a time constant
    template.tau_rise_ms             median rise
    deconvolution.fallback_tau_*     the same, for when kernel estimation
                                     cannot find enough isolated events
    consensus tolerance_ms           95th percentile of the ACTUAL peak-time
                                     disagreement between methods
    *.min_amp_abs / min_amp_pa       4 x the noisiest recording's sigma, the
                                     Greger & Watson (J Physiol 2025)
                                     detection limit

The tolerance is the one worth noticing. Elsewhere it is a guess at how far
apart two methods put the same event; here it is measured.

What it cannot set
------------------
Nothing here can tell you whether a disputed event is real. That needs eyes:
review a recording, and `<stem>_multimethod_reviewed.csv` becomes a labelled
set that could train miniML by transfer learning (~400 events is enough) or an
Eventer-style classifier. Until then the thresholds that trade precision
against recall -- amplitude_sd, thresh_sd, criterion_threshold,
min_state_rise_sd -- are left alone, because choosing them needs a ground truth
this cannot supply.
"""

from __future__ import annotations

import dataclasses
import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import runner
from .settings import Settings
from .trace import load, noise_sd

#: Recordings that are not gap-free voltage-clamp sEPSC data are skipped.
MIN_DURATION_S = 20.0


@dataclass
class Calibration:
    """What was measured, and what it implies."""

    files: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    n_events: int = 0
    n_high_confidence: int = 0
    amplitude: dict = field(default_factory=dict)
    rise_ms: dict = field(default_factory=dict)
    decay_ms: dict = field(default_factory=dict)
    iei_ms: dict = field(default_factory=dict)
    noise_sd_per_file: dict = field(default_factory=dict)
    timing_spread_ms: dict = field(default_factory=dict)
    proposed: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    withheld: list = field(default_factory=list)

    @property
    def worst_noise(self) -> float:
        return max(self.noise_sd_per_file.values()) if self.noise_sd_per_file else 0.0


def _pct(values, *qs) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {}
    out = {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean())}
    for q in qs:
        out[f"p{int(q)}"] = float(np.percentile(v, q))
    return out


def find_recordings(folder: str, min_duration_s: float = MIN_DURATION_S):
    """Gap-free, single-sweep, current-unit recordings long enough to analyse.

    Returns (usable, skipped_with_reason). A folder of patch-clamp data is
    mostly not sEPSC recordings -- access-resistance checks and IV protocols
    outnumbered them 85 to 4 in the dataset this was written against -- so
    saying what was skipped and why matters as much as the list itself.
    """
    import pyabf

    usable, skipped = [], []
    for path in sorted(glob.glob(os.path.join(folder, "*.abf"))):
        name = os.path.basename(path)
        try:
            abf = pyabf.ABF(path, loadData=False)
        except Exception as exc:                 # noqa: BLE001
            skipped.append((name, f"unreadable: {type(exc).__name__}"))
            continue
        if abf.sweepCount != 1:
            skipped.append((name, f"{abf.sweepCount} sweeps (not gap-free)"))
        elif abf.dataLengthSec < min_duration_s:
            skipped.append((name, f"{abf.dataLengthSec:.1f} s, too short"))
        elif not abf.adcUnits or abf.adcUnits[0] != "pA":
            skipped.append((name, f"units {abf.adcUnits[:1]}, not voltage clamp"))
        else:
            usable.append(path)
    return usable, skipped


def collect(folder: str, settings: Settings | None = None, channel: int = 0,
            min_methods: int | None = None, rerun: bool = False,
            progress=None) -> Calibration:
    """Measure the pooled high-confidence events across a folder.

    Reads each recording's existing consensus table when there is one, so a
    calibration after a run costs seconds. `rerun=True` forces detection.
    """
    s = settings or Settings()
    cal = Calibration()
    usable, cal.skipped = find_recordings(folder)

    per_file_events = []
    for path in usable:
        name = os.path.basename(path)
        stem = runner.output_stem(path, channel, s.preprocessing)
        cons_path = stem + runner.CONSENSUS_SUFFIX
        if rerun or not os.path.exists(cons_path):
            if progress:
                progress(f"  {name}: running detection")
            out = runner.run_all(path, channel, s)
            out.write()
            cons = pd.read_csv(cons_path, float_precision="round_trip")
            cal.noise_sd_per_file[name] = out.noise_sd
        else:
            if progress:
                progress(f"  {name}: using existing run")
            cons = pd.read_csv(cons_path, float_precision="round_trip")
            cal.noise_sd_per_file[name] = noise_sd(
                load(path, channel, s.preprocessing), s.measurement.direction)

        methods = runner.methods_in_consensus(cons)
        if not methods:
            cal.skipped.append((name, "no method produced events"))
            continue
        cal.files.append(name)
        cal.n_events += len(cons)

        need = min_methods if min_methods is not None else len(methods)
        high = cons[cons["n_methods"] >= need]
        cal.n_high_confidence += len(high)

        # Per-method peak-time disagreement, from each method's own table.
        # This is what the consensus tolerance should actually be set from.
        spreads = []
        per_method = {}
        for m in methods:
            p = f"{stem}_{m}{runner.EVENTS_SUFFIX}"
            if os.path.exists(p):
                per_method[m] = pd.read_csv(p, float_precision="round_trip")
        for t0 in high["peak_time_s"].to_numpy(dtype=float):
            times = []
            for m, df in per_method.items():
                if df.empty:
                    continue
                d = np.abs(df["peak_time_s"].to_numpy(dtype=float) - t0)
                j = int(np.argmin(d))
                if d[j] <= s.tolerance_ms / 1000.0:
                    times.append(float(df["peak_time_s"].iloc[j]))
            if len(times) > 1:
                spreads.append((max(times) - min(times)) * 1e3)

        # Kinetics come from a single method's measurements to avoid counting
        # the same event several times; every method is measured identically,
        # so any of them is equivalent. Prefer the most agreed-with one.
        ref = _reference_method(cons, methods)
        ref_df = per_method.get(ref)
        if ref_df is not None and not ref_df.empty:
            hi_times = set(np.round(high["peak_time_s"].to_numpy(dtype=float), 4))
            keep = ref_df[np.round(ref_df["peak_time_s"], 4).isin(hi_times)]
            per_file_events.append((keep, spreads))

    if per_file_events:
        pooled = pd.concat([e for e, _ in per_file_events], ignore_index=True)
        spreads = [x for _, sp in per_file_events for x in sp]
        cal.amplitude = _pct(np.abs(pooled["amplitude"]), 5, 50, 95, 99)
        cal.rise_ms = _pct(pooled["rise_time_ms"], 50, 95, 99)
        cal.decay_ms = _pct(pooled["decay_time_ms"], 50, 95, 99)
        cal.iei_ms = _pct(pooled["inter_event_interval_ms"], 5, 50, 95)
        cal.timing_spread_ms = _pct(spreads, 50, 95, 99)
    cal.proposed = propose(cal, s)
    return cal


def _reference_method(cons: pd.DataFrame, methods: list) -> str:
    """The method whose events are most often shared, as the kinetics source.

    Any method would do -- measurement is shared, so the numbers are the same
    -- but the most agreed-with one has the fewest idiosyncratic detections in
    its table, so the pooled kinetics are least skewed by one method's quirks.
    """
    best, best_frac = methods[0], -1.0
    for m in methods:
        col = f"found_{m}"
        if col not in cons.columns:
            continue
        found = cons[cons[col] == 1]
        if found.empty:
            continue
        frac = float((found["n_methods"] > 1).mean())
        if frac > best_frac:
            best, best_frac = m, frac
    return best


def _tau_from_half_decay(half_decay_ms: float) -> float:
    """Half-decay time -> exponential time constant.

    The measured decay_time_ms is the time to fall to decay_fraction (0.5) of
    peak, so tau = t_half / ln 2. The analytic kernels want a tau, not a
    half-time, and conflating them makes the fallback kernel ~30% too fast.
    """
    return float(half_decay_ms / np.log(2.0)) if half_decay_ms > 0 else 0.0


#: Half-decays outside this range, in ms, are treated as a measurement
#: artefact rather than a population.
#:
#: The floor is deliberately low. An earlier version used 1.5 ms, on the
#: reasoning that an AMPA-mediated sEPSC decaying faster than that would beat
#: most reported kinetics -- and that was a prior about synapses, not a fact
#: about any particular recording. It withheld a perfectly good measurement:
#: on the dataset this was written against the median half-decay really is
#: 1.20 ms (tau 1.73 ms), and all five detectors' peak-normalised average
#: waveforms superimpose on that shape, which is about as much corroboration
#: as a measurement can get without a second method of measuring it.
#:
#: 0.4 ms is what the floor should guard: at this package's 10 kHz working
#: rate that is 4 samples, below which a half-decay is being read off too few
#: points to mean anything, whatever the biology.
PLAUSIBLE_HALF_DECAY_MS = (0.4, 40.0)


def propose(cal: Calibration, settings: Settings) -> dict:
    """Parameter values implied by the measurements. Nothing is applied here.

    Two asymmetries make this safe to run unattended:

    WIDEN ONLY. A search window is never narrowed. The pooled events were
    selected BY the current window, so the tail beyond it is invisible to the
    measurement -- narrowing to the p99 of what got through would lock in that
    bias and make genuinely slower events permanently unmeasurable. Widening
    rests on no such circularity.

    KINETICS ARE FLAGGED, NOT FORCED. A template or kernel time constant that
    is too FAST is the worst failure mode in this package (assuming 6 ms on
    25 ms data drops deconvolution's precision to 0.51), and the
    high-confidence pool is biased towards fast, clean, isolated events. So an
    implausible tau is withheld and reported rather than written.
    """
    out: dict = {}
    cal.warnings = []
    cal.withheld = []
    if not cal.decay_ms:
        return out

    rise_p99 = cal.rise_ms.get("p99", 0.0)
    decay_p99 = cal.decay_ms.get("p99", 0.0)
    decay_med = cal.decay_ms.get("median", 0.0)
    tau = _tau_from_half_decay(decay_med)
    rise_med = cal.rise_ms.get("median", 0.0)

    # -- search windows: widen only -----------------------------------------
    for field_name, p99 in (("decay_search_ms", decay_p99),
                            ("onset_search_ms", rise_p99)):
        if p99 <= 0:
            continue
        want = round(p99 * 1.5, 1)
        current = float(getattr(settings.measurement, field_name))
        if want > current:
            out[("measurement", field_name)] = want
        else:
            cal.withheld.append(
                f"measurement.{field_name}: measured p99 implies {want} ms, but "
                f"narrowing from {current} ms would make slower events "
                f"unmeasurable, and the pool was selected by the current "
                f"window -- left at {current}")

    # -- kinetics: only if plausible ----------------------------------------
    lo, hi = PLAUSIBLE_HALF_DECAY_MS
    if tau > 0 and lo <= decay_med <= hi:
        out[("template", "tau_decay_ms")] = round(tau, 2)
        out[("deconvolution", "fallback_tau_decay_ms")] = round(tau, 2)
    elif tau > 0:
        cal.withheld.append(
            f"tau_decay_ms: median half-decay {decay_med:.2f} ms implies "
            f"tau {tau:.2f} ms, outside the {lo}-{hi} ms range this trusts. "
            f"Below the floor a half-decay is read off too few samples to "
            f"mean anything; above it, the events are slow enough that the "
            f"kernel estimator's own limits apply. Not written. Check the "
            f"peak-normalised average in the waveforms figure -- if every "
            f"method agrees on that shape, the measurement is probably right "
            f"and you can set it by hand.")
    if rise_med > 0:
        out[("template", "tau_rise_ms")] = round(rise_med, 2)
        out[("deconvolution", "fallback_tau_rise_ms")] = round(rise_med, 2)

    # -- the measured disagreement, rather than a guess at it ---------------
    spread_p95 = cal.timing_spread_ms.get("p95", 0.0)
    if spread_p95 > 0:
        want = round(max(spread_p95 * 1.2, 0.5), 2)
        out[("consensus", "tolerance_ms")] = want
        if want < float(settings.tolerance_ms):
            cal.warnings.append(
                f"tolerance {settings.tolerance_ms} -> {want} ms is measured, but "
                f"tightening it merges fewer detections, so consensus counts "
                f"will RISE and agreement fractions fall. That is a change in "
                f"bookkeeping, not in detection.")

    # -- the detection limit, from the noise rather than from the events ----
    if cal.worst_noise > 0:
        limit = round(4.0 * cal.worst_noise, 2)
        out[("deconvolution", "min_amp_abs")] = limit
        quietest = min(cal.noise_sd_per_file.values())
        if cal.worst_noise > 1.5 * quietest:
            cal.warnings.append(
                f"noise spans {quietest:.2f}-{cal.worst_noise:.2f} pA across "
                f"these recordings. A single 4-sigma limit of {limit:.2f} pA is "
                f"set from the noisiest, per Greger & Watson, so the quieter "
                f"cells lose real events they could have resolved -- that is "
                f"the price of a limit that is valid across the set. Consider "
                f"excluding the noisiest recording instead.")

    if cal.amplitude.get("median", 0) > 0:
        cal.warnings.append(
            f"pooled |amplitude| median is {cal.amplitude['median']:.1f} pA, well "
            f"above the per-method medians, because only events every method "
            f"found were pooled. That bias is right for measuring kinetics and "
            f"wrong for anything about the detection limit, which is why the "
            f"limit above comes from the noise instead.")
    return out


def apply(cal: Calibration, settings: Settings) -> Settings:
    """A copy of `settings` with the proposals applied."""
    s = Settings.from_dict(settings.to_dict())
    for (section, field_name), value in cal.proposed.items():
        if section == "measurement":
            s.measurement = dataclasses.replace(s.measurement, **{field_name: value})
        elif section == "consensus":
            s.tolerance_ms = value
        else:
            s.set_params_for(section, dataclasses.replace(
                s.params_for(section), **{field_name: value}))
    return s


def _wrap(text: str, width: int = 72, indent: str = "      ") -> str:
    import textwrap

    return ("\n" + indent).join(textwrap.wrap(text, width))


def report(cal: Calibration, settings: Settings) -> str:
    """Human-readable before/after, including what was NOT calibrated."""
    L = []
    L.append(f"Recordings used: {len(cal.files)}")
    for f in cal.files:
        L.append(f"    {f:34s} sigma {cal.noise_sd_per_file.get(f, 0):.2f} pA")
    if cal.skipped:
        L.append(f"Skipped: {len(cal.skipped)}")
        shown = cal.skipped[:6]
        for name, why in shown:
            L.append(f"    {name:34s} {why}")
        if len(cal.skipped) > len(shown):
            L.append(f"    ... and {len(cal.skipped) - len(shown)} more")
    L.append("")
    L.append(f"Events pooled: {cal.n_high_confidence} high-confidence "
             f"of {cal.n_events} consensus events")
    if not cal.decay_ms:
        L.append("Not enough high-confidence events to calibrate.")
        return "\n".join(L)

    def line(label, d, unit=""):
        if not d:
            return
        L.append(f"    {label:22s} median {d['median']:7.2f}{unit}"
                 + (f"   p95 {d['p95']:7.2f}{unit}" if "p95" in d else "")
                 + (f"   p99 {d['p99']:7.2f}{unit}" if "p99" in d else ""))

    L.append("Measured:")
    line("|amplitude|", cal.amplitude, " pA")
    line("rise time", cal.rise_ms, " ms")
    line("half-decay", cal.decay_ms, " ms")
    line("inter-event interval", cal.iei_ms, " ms")
    line("method timing spread", cal.timing_spread_ms, " ms")
    L.append(f"    {'noisiest sigma':22s} {cal.worst_noise:7.2f} pA"
             f"   -> 4 sigma limit {4 * cal.worst_noise:.2f} pA")
    L.append("")

    if not cal.proposed:
        L.append("No parameter changes implied.")
        return "\n".join(L)

    L.append("Proposed defaults:")
    for (section, field_name), value in sorted(cal.proposed.items()):
        if section == "measurement":
            current = getattr(settings.measurement, field_name)
        elif section == "consensus":
            current = settings.tolerance_ms
        else:
            current = getattr(settings.params_for(section), field_name)
        mark = "  (unchanged)" if np.isclose(float(current), float(value)) else ""
        L.append(f"    {section}.{field_name:26s} {current!s:>8} -> {value!s:<8}{mark}")
    if cal.withheld:
        L.append("")
        L.append("WITHHELD (measured, but not safe to write):")
        for w in cal.withheld:
            L.append("    " + _wrap(w))
    if cal.warnings:
        L.append("")
        L.append("READ BEFORE WRITING:")
        for w in cal.warnings:
            L.append("    " + _wrap(w))
    L.append("")
    L.append("Left alone deliberately -- these trade precision against recall,")
    L.append("and choosing them needs a ground truth this cannot supply:")
    L.append("    threshold.amplitude_sd, deconvolution.thresh_sd,")
    L.append("    template.criterion_threshold, autoclassifier.min_state_rise_sd")
    L.append("")
    L.append("To set those, review a recording and use its")
    L.append("*_multimethod_reviewed.csv as a labelled set.")
    return "\n".join(L)
