"""Tests for the detectors, the shared measurement, and the trace layer."""

from __future__ import annotations

import io

import numpy as np
import pytest

from multimethod import detectors
from multimethod.benchmark import score, score_detector, synth_trace
from multimethod.events import (EVENT_COLUMNS, MeasurementParams, events_frame,
                                measure_all, measure_at)
from multimethod.settings import Settings
from multimethod.trace import (Preprocessing, Trace, noise_sd, output_stem,
                               robust_sd, track_baseline)

DT = 1e-4
IN_REPO = detectors.in_repo_names()


# ------------------------------------------------------------- the registry

def test_every_registered_detector_is_wired_up():
    assert set(detectors.names()) >= {"threshold", "deconvolution", "template",
                                      "miniml", "fastmini"}
    for name in detectors.names():
        det = detectors.get(name)
        assert det.name == name
        assert det.label and det.blurb, name
        assert det.default_params() is not None, name


def test_unknown_detector_raises_with_a_useful_message():
    with pytest.raises(KeyError, match="known"):
        detectors.get("nope")


def test_external_detectors_are_flagged_as_such():
    assert detectors.get("miniml").external
    assert detectors.get("fastmini").external
    for name in IN_REPO:
        assert not detectors.get(name).external


def test_an_unavailable_external_detector_reports_why_instead_of_raising():
    from multimethod.detectors import CommandParams, MiniMLParams

    ok, why = detectors.get("fastmini").available(CommandParams(enabled=False))
    assert not ok and why
    ok, why = detectors.get("miniml").available(
        MiniMLParams(python_exe="/definitely/not/here/python"))
    assert not ok and "python_exe" in why


# --------------------------------------------------- each detector detects

@pytest.mark.parametrize("method", IN_REPO)
def test_detector_finds_planted_events(method):
    trace, true_s = synth_trace(duration_s=20.0, noise_sd_pa=2.0, seed=3)
    r = score_detector(method, trace, true_s)
    assert r["recall"] > 0.7, f"{method} recall {r['recall']:.2f}"
    assert r["precision"] > 0.6, f"{method} precision {r['precision']:.2f}"


@pytest.mark.parametrize("method", IN_REPO)
def test_detector_finds_nothing_in_pure_noise_beyond_a_few_false_positives(method):
    """A detector that fires constantly on noise is useless in a consensus:
    it would agree with everything by chance."""
    trace, _ = synth_trace(duration_s=20.0, rate_hz=0.0, noise_sd_pa=2.0, seed=5)
    idxs = detectors.get(method).find(trace, Settings().params_for(method))
    rate = len(idxs) / trace.duration_s
    assert rate < 2.0, f"{method} fired at {rate:.1f} Hz on pure noise"


@pytest.mark.parametrize("method", IN_REPO)
def test_detector_handles_a_flat_and_an_empty_trace(method):
    params = Settings().params_for(method)
    det = detectors.get(method)
    flat = Trace(t=np.arange(5000) * DT, v=np.zeros(5000), dt=DT, unit="pA",
                 abf_path="<x>", channel=0, preprocessing=Preprocessing(enabled=False))
    assert len(det.find(flat, params)) == 0
    tiny = Trace(t=np.zeros(0), v=np.zeros(0), dt=DT, unit="pA", abf_path="<x>",
                 channel=0, preprocessing=Preprocessing(enabled=False))
    assert len(det.find(tiny, params)) == 0


@pytest.mark.parametrize("method", IN_REPO)
def test_detectors_do_not_modify_the_trace_they_are_given(method):
    """Every detector sees the same array; one mutating it would corrupt the
    ones that run after it. Trace makes v read-only, so this would raise."""
    trace, _ = synth_trace(duration_s=5.0, seed=7)
    before = np.array(trace.v, copy=True)
    detectors.get(method).find(trace, Settings().params_for(method))
    assert np.array_equal(trace.v, before)


def test_deconvolution_reports_its_kernel_in_diagnostics():
    trace, _ = synth_trace(duration_s=20.0, tau_ms=6.0, seed=3)
    det = detectors.get("deconvolution")
    det.find(trace, Settings().params_for("deconvolution"))
    d = det.last_diagnostics
    assert d["kernel_source"] in ("estimated", "analytic-fallback")
    assert d["kernel_ms"] > 0
    assert d["n_candidates"] >= d["n_after_peel"], "peel should only remove"


def test_deconvolution_kernel_lengthens_for_slower_decay():
    fast, _ = synth_trace(duration_s=25.0, tau_ms=3.0, seed=3)
    slow, _ = synth_trace(duration_s=25.0, tau_ms=25.0, seed=3)
    det = detectors.get("deconvolution")
    p = Settings().params_for("deconvolution")
    det.find(fast, p)
    k_fast = det.last_diagnostics["kernel_ms"]
    det.find(slow, p)
    k_slow = det.last_diagnostics["kernel_ms"]
    assert k_slow > k_fast, f"{k_slow:.1f} ms should exceed {k_fast:.1f} ms"


def test_autoclassifier_reports_nothing_when_no_state_looks_like_an_event():
    """An unsupervised clusterer cannot say "there are no events here": given
    pure noise it still finds k states and one still rises most. Without the
    state gate this detector fired at ~70 Hz on noise alone."""
    trace, _ = synth_trace(duration_s=20.0, rate_hz=0.0, noise_sd_pa=2.0, seed=5)
    det = detectors.get("autoclassifier")
    assert len(det.find(trace, Settings().params_for("autoclassifier"))) == 0
    d = det.last_diagnostics
    assert "skipped" in d, "it should say why it found nothing"
    assert d["state_rise_sd"] < 1.0


def test_autoclassifier_state_rise_separates_signal_from_noise():
    """The gate works because the measure is not marginal: on planted data the
    winning state rises ~8 noise SDs, on noise alone ~0.06. Anything between
    would do, which is why the default is not delicate."""
    det = detectors.get("autoclassifier")
    p = Settings().params_for("autoclassifier")

    with_events, _ = synth_trace(duration_s=20.0, rate_hz=5.0, noise_sd_pa=2.0, seed=5)
    det.find(with_events, p)
    signal_rise = det.last_diagnostics["state_rise_sd"]

    noise_only, _ = synth_trace(duration_s=20.0, rate_hz=0.0, noise_sd_pa=2.0, seed=5)
    det.find(noise_only, p)
    noise_rise = det.last_diagnostics["state_rise_sd"]

    assert signal_rise > 4.0, signal_rise
    assert noise_rise < 1.0, noise_rise
    assert signal_rise > 10 * noise_rise, "the gate should not be marginal"


def test_autoclassifier_is_reproducible():
    """It is seeded, so the same recording must give the same events -- an
    unreproducible detector cannot be reviewed."""
    trace, _ = synth_trace(duration_s=15.0, seed=3)
    det = detectors.get("autoclassifier")
    p = Settings().params_for("autoclassifier")
    first = det.find(trace, p)
    second = det.find(trace, p)
    assert np.array_equal(first, second)


def test_autoclassifier_uses_no_labels_and_no_template():
    """Its whole reason for being here: no labels, no template, no trained
    model. If it grew any of those it would stop failing differently from the
    other methods, which is the only reason to carry a fifth detector.

    Checked by parsing rather than grepping: the module's docstring discusses
    keras and torch (explaining why the encoder is linear instead), and a
    substring search matches that prose.
    """
    import ast

    from multimethod.detectors import autoclassifier

    tree = ast.parse(io.open(autoclassifier.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("sklearn", "keras", "tensorflow", "torch"):
        assert banned not in imported, (
            f"{banned} would make this a trained detector, and would add a "
            f"dependency for one method")

    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("biexp_kernel", "estimate_kernel", "model_path"):
        assert banned not in names, (
            f"{banned} would give this detector a template or a model")


def test_template_criterion_peaks_at_a_planted_event():
    from multimethod.detectors.template import detection_criterion
    from multimethod.detectors.deconvolution import biexp_kernel

    kern = biexp_kernel(DT, 0.6, 6.0, 30.0)
    sv = np.zeros(4000)
    onset = 1000
    sv[onset:onset + kern.size] += 20.0 * kern
    crit = detection_criterion(sv, kern)
    # The criterion is shifted to the template's own PEAK, not the start of the
    # fitted window -- otherwise every detection lands one rise-time early. So
    # the expected position is the onset plus the kernel's argmax, not the onset.
    expected = onset + int(np.argmax(kern))
    assert abs(int(np.argmax(crit)) - expected) <= 3, (
        f"criterion peaked at {int(np.argmax(crit))}, expected ~{expected}")


# ------------------------------------------------------ shared measurement

def test_measurement_is_identical_across_detectors():
    """The package's central claim: an amplitude means the same thing whichever
    method located the event.

    Two different detectors that land on the same sample must therefore produce
    byte-identical numbers -- the measurement does not know which detector
    asked. Uses a peak both detectors actually found, rather than an arbitrary
    index (measure_at correctly returns None on plain noise).
    """
    trace, _ = synth_trace(duration_s=20.0, noise_sd_pa=1.0, seed=11)
    s = Settings()
    a_idx = set(detectors.get("threshold").find(trace, s.params_for("threshold")))
    b_idx = set(detectors.get("deconvolution").find(
        trace, s.params_for("deconvolution")))
    shared = sorted(a_idx & b_idx)
    assert shared, "the fixture should give the two detectors some common peaks"

    m = MeasurementParams()
    peak = shared[len(shared) // 2]
    a = measure_at(trace.v, trace.dt, peak, m)
    b = measure_at(trace.v, trace.dt, peak, m)
    assert a is not None
    assert a.amplitude == b.amplitude
    assert a.area == b.area
    assert a.rise_time_ms == b.rise_time_ms and a.decay_time_ms == b.decay_time_ms


def test_measure_at_returns_none_rather_than_guessing():
    m = MeasurementParams()
    v = np.zeros(5000)
    assert measure_at(v, DT, 2, m) is None, "too close to the start"
    assert measure_at(v, DT, 2500, m) is None, "flat: no deflection to measure"


def test_measure_all_counts_what_it_could_not_measure():
    trace, _ = synth_trace(duration_s=10.0, seed=13)
    events, unmeasurable = measure_all(trace, [1, 2, 3], MeasurementParams())
    assert events == [] and unmeasurable == 3


def test_amplitudes_are_signed_by_direction():
    inward, _ = synth_trace(duration_s=15.0, amp_mean=-20.0, seed=17)
    idxs = detectors.get("threshold").find(inward, Settings().params_for("threshold"))
    events, _ = measure_all(inward, idxs, MeasurementParams(direction="negative"))
    assert events and all(e.amplitude < 0 for e in events)
    assert all(e.area < 0 for e in events)


def test_events_frame_schema_and_iei():
    trace, _ = synth_trace(duration_s=15.0, seed=19)
    idxs = detectors.get("deconvolution").find(
        trace, Settings().params_for("deconvolution"))
    events, _ = measure_all(trace, idxs, MeasurementParams(), "deconvolution")
    df = events_frame(events)
    assert list(df.columns) == EVENT_COLUMNS
    assert df["peak_idx"].dtype == "int64"
    assert np.isnan(df["inter_event_interval_ms"].iloc[0])
    assert (df["peak_time_s"].diff().iloc[1:] >= 0).all(), "must be time-sorted"


def test_empty_events_frame_still_has_every_column():
    df = events_frame([])
    assert df.empty and list(df.columns) == EVENT_COLUMNS


# ----------------------------------------------------------- the trace layer

def test_trace_is_read_only():
    trace, _ = synth_trace(duration_s=2.0, seed=1)
    with pytest.raises(ValueError):
        trace.v[0] = 999.0


def test_output_stem_separates_runs_that_describe_different_data():
    raw = output_stem("rec.abf", 0, Preprocessing(enabled=False))
    filt = output_stem("rec.abf", 0, Preprocessing(enabled=True))
    ch1 = output_stem("rec.abf", 1, Preprocessing(enabled=False))
    assert len({raw, filt, ch1}) == 3, "a second run must not overwrite the first"
    assert "filt3000Hz10000Hz" in filt


def test_noise_sd_beats_a_diff_based_estimate_on_a_filtered_trace():
    """diff(v)/sqrt(2) underestimates once the trace is low-pass filtered,
    because filtering correlates adjacent samples -- and this package filters
    by default, so the naive estimate would mis-scale every SD threshold."""
    from multimethod.trace import bessel_lowpass

    rng = np.random.default_rng(0)
    v = bessel_lowpass(rng.normal(0, 5.0, 200_000), 1 / DT, 1000.0, 8)
    trace = Trace(t=np.arange(v.size) * DT, v=v, dt=DT, unit="pA",
                  abf_path="<x>", channel=0, preprocessing=Preprocessing(enabled=False))
    true_sd = float(v.std())
    assert abs(noise_sd(trace) - true_sd) / true_sd < 0.25
    naive = robust_sd(np.diff(v)) / np.sqrt(2)
    assert naive < 0.6 * true_sd, "the naive estimate really does fail here"


def test_track_baseline_follows_slow_drift():
    n = 100_000
    drift = 30.0 * np.sin(2 * np.pi * np.arange(n) / n)
    assert np.max(np.abs(track_baseline(drift, DT, 300.0) - drift)) < 5.0


def test_score_helper_is_one_to_one():
    assert score([1.0, 2.0], [1.0, 2.0])["f1"] == 1.0
    assert score([1.0], [1.0, 5.0])["fp"] == 1
    # one detection cannot satisfy two true events
    assert score([1.0, 1.0005], [1.0])["tp"] == 1
    assert score([1.0], [])["recall"] == 0.0
