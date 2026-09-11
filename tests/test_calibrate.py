"""Tests for calibrating the defaults from real recordings.

These mostly cover the two safety rules, because they are what stops an
unattended calibration from making the package worse: search windows widen but
never narrow, and an implausible time constant is withheld rather than written.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from multimethod import calibrate
from multimethod.settings import Settings


def _cal(**kw):
    c = calibrate.Calibration()
    c.decay_ms = kw.pop("decay_ms", {"median": 3.0, "p95": 6.0, "p99": 8.0})
    c.rise_ms = kw.pop("rise_ms", {"median": 0.8, "p95": 1.8, "p99": 2.4})
    c.amplitude = kw.pop("amplitude", {"median": 20.0, "p95": 50.0, "p99": 70.0})
    c.timing_spread_ms = kw.pop("timing_spread_ms", {"median": 0.1, "p95": 0.4})
    c.noise_sd_per_file = kw.pop("noise_sd_per_file", {"a.abf": 2.0})
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ------------------------------------------------------- widen, never narrow

def test_a_search_window_is_widened_when_the_events_need_it():
    s = Settings()
    s.measurement = type(s.measurement)(**{**s.measurement.as_dict(),
                                           "decay_search_ms": 5.0})
    cal = _cal(decay_ms={"median": 3.0, "p95": 9.0, "p99": 12.0})
    proposed = calibrate.propose(cal, s)
    assert proposed[("measurement", "decay_search_ms")] == pytest.approx(18.0)


def test_a_search_window_is_never_narrowed():
    """The pooled events were selected BY the current window, so the tail
    beyond it is invisible to the measurement. Narrowing to the p99 of what
    got through would lock that bias in and make slower events permanently
    unmeasurable."""
    s = Settings()                      # decay_search_ms defaults to 50 ms
    cal = _cal(decay_ms={"median": 1.2, "p95": 2.4, "p99": 3.2})
    proposed = calibrate.propose(cal, s)
    assert ("measurement", "decay_search_ms") not in proposed
    assert any("decay_search_ms" in w for w in cal.withheld)
    assert any("unmeasurable" in w for w in cal.withheld)


# -------------------------------------------- kinetics flagged, not forced

def test_an_implausibly_fast_half_decay_is_withheld():
    """Too fast is the dangerous direction: a kernel faster than the events
    drops deconvolution's precision to about 0.5."""
    cal = _cal(decay_ms={"median": 0.1, "p95": 0.2, "p99": 0.3})
    proposed = calibrate.propose(cal, Settings())
    assert ("template", "tau_decay_ms") not in proposed
    assert any("tau_decay_ms" in w for w in cal.withheld)


def test_a_fast_but_measurable_half_decay_is_accepted():
    """1.2 ms is fast for an sEPSC, but it is what the reference dataset shows,
    corroborated by every method's peak-normalised average waveform. The floor
    guards against too-few-samples, not against unexpected biology."""
    cal = _cal(decay_ms={"median": 1.2, "p95": 2.4, "p99": 3.2})
    proposed = calibrate.propose(cal, Settings())
    tau = proposed[("template", "tau_decay_ms")]
    assert tau == pytest.approx(1.2 / 0.6931, abs=0.01), "half-decay -> tau"
    assert proposed[("deconvolution", "fallback_tau_decay_ms")] == tau


def test_half_decay_is_converted_to_a_time_constant_not_used_raw():
    """Conflating the two makes the fallback kernel about 30% too fast."""
    assert calibrate._tau_from_half_decay(1.0) == pytest.approx(1.4427, abs=1e-3)
    assert calibrate._tau_from_half_decay(0.0) == 0.0


# --------------------------------------------------------------- tolerance

def test_the_tolerance_comes_from_measured_disagreement():
    cal = _cal(timing_spread_ms={"median": 0.5, "p95": 2.0})
    proposed = calibrate.propose(cal, Settings())
    assert proposed[("consensus", "tolerance_ms")] == pytest.approx(2.4)


def test_tightening_the_tolerance_is_warned_about():
    """It changes the bookkeeping, not the detection: fewer detections merge,
    so consensus counts rise and agreement falls."""
    cal = _cal(timing_spread_ms={"median": 0.0, "p95": 0.2})
    calibrate.propose(cal, Settings())
    assert any("bookkeeping" in w for w in cal.warnings)


# ----------------------------------------------------- the detection limit

def test_the_detection_limit_comes_from_noise_not_from_the_events():
    """The pooled events are biased large, so a limit derived from them would
    be far too high. 4 sigma of the noisiest recording, per Greger & Watson."""
    cal = _cal(noise_sd_per_file={"quiet.abf": 1.0, "noisy.abf": 3.0})
    proposed = calibrate.propose(cal, Settings())
    assert proposed[("deconvolution", "min_amp_abs")] == pytest.approx(12.0)


def test_mismatched_noise_across_recordings_is_warned_about():
    cal = _cal(noise_sd_per_file={"quiet.abf": 1.0, "noisy.abf": 3.0})
    calibrate.propose(cal, Settings())
    assert any("noise spans" in w for w in cal.warnings)


def test_the_high_confidence_amplitude_bias_is_stated():
    cal = _cal()
    calibrate.propose(cal, Settings())
    assert any("every method found" in w for w in cal.warnings)


# ------------------------------------------------------- apply and report

def test_apply_does_not_mutate_the_original_settings():
    s = Settings()
    before = s.to_dict()
    cal = _cal(decay_ms={"median": 3.0, "p95": 60.0, "p99": 80.0})
    cal.proposed = calibrate.propose(cal, s)
    out = calibrate.apply(cal, s)
    assert s.to_dict() == before, "the original must be untouched"
    assert out.to_dict() != before, "the copy should carry the changes"


def test_apply_sets_every_section_kind():
    s = Settings()
    cal = _cal()
    cal.proposed = {("measurement", "decay_search_ms"): 99.0,
                    ("consensus", "tolerance_ms"): 1.25,
                    ("template", "tau_decay_ms"): 4.5}
    out = calibrate.apply(cal, s)
    assert out.measurement.decay_search_ms == 99.0
    assert out.tolerance_ms == 1.25
    assert out.params_for("template").tau_decay_ms == 4.5


def test_report_says_what_was_withheld_and_why():
    s = Settings()
    cal = _cal(decay_ms={"median": 1.2, "p95": 2.4, "p99": 3.2})
    cal.proposed = calibrate.propose(cal, s)
    cal.files = ["a.abf"]
    cal.n_events, cal.n_high_confidence = 100, 40
    text = calibrate.report(cal, s)
    assert "WITHHELD" in text
    assert "decay_search_ms" in text
    assert "Left alone deliberately" in text, (
        "the report must name the thresholds it cannot calibrate")
    assert "reviewed.csv" in text, "and point at how to get labels"


def test_no_events_reports_rather_than_proposing():
    cal = calibrate.Calibration()
    assert calibrate.propose(cal, Settings()) == {}
    assert "Not enough high-confidence events" in calibrate.report(cal, Settings())


# ------------------------------------------------------------ file selection

def test_find_recordings_reports_nothing_for_an_empty_folder(tmp_path):
    usable, skipped = calibrate.find_recordings(str(tmp_path))
    assert usable == [] and skipped == []


def test_find_recordings_ignores_non_abf_files(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "events.csv").write_text("x")
    usable, skipped = calibrate.find_recordings(str(tmp_path))
    assert usable == [] and skipped == []


def test_find_recordings_reports_an_unreadable_abf(tmp_path):
    """A truncated or non-ABF .abf must be named, not silently dropped."""
    (tmp_path / "broken.abf").write_bytes(b"not an abf at all")
    usable, skipped = calibrate.find_recordings(str(tmp_path))
    assert usable == []
    assert len(skipped) == 1 and skipped[0][0] == "broken.abf"
