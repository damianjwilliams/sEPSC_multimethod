"""Tests for settings round-tripping, the runner, and the review window.

The behaviour that matters most here is that a failing detector is DATA, not
an exception: external detectors depend on other interpreters, so a partial
run is the normal case and must still produce a usable consensus table.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from multimethod import detectors, runner
from multimethod.benchmark import synth_trace
from multimethod.events import MeasurementParams
from multimethod.settings import Settings, field_specs
from multimethod.trace import Preprocessing

DT = 1e-4


# --------------------------------------------------------------- settings

def test_settings_fills_in_every_registered_detector():
    s = Settings()
    for name in detectors.names():
        assert name in s.detector_params
        assert name in s.enabled
        assert isinstance(s.params_for(name), detectors.get(name).params_cls)


def test_settings_round_trip_through_json(tmp_path):
    import dataclasses

    s = Settings(tolerance_ms=7.5)
    s.measurement = MeasurementParams(direction="positive", n_avg_peak=5)
    s.preprocessing = Preprocessing(enabled=False)
    s.set_params_for("threshold", dataclasses.replace(
        s.params_for("threshold"), amplitude_sd=6.25))
    s.enabled["template"] = False

    path = s.save(str(tmp_path / "cfg.json"))
    back = Settings.load(path)
    assert back.tolerance_ms == 7.5
    assert back.measurement.direction == "positive"
    assert back.measurement.n_avg_peak == 5
    assert back.preprocessing.enabled is False
    assert back.params_for("threshold").amplitude_sd == 6.25
    assert back.enabled["template"] is False


def test_missing_config_file_gives_defaults(tmp_path):
    s = Settings.load(str(tmp_path / "absent.json"))
    assert s.tolerance_ms == Settings().tolerance_ms


def test_config_with_unknown_keys_and_detectors_still_loads(tmp_path):
    """A config from a newer version, or naming a renamed detector, must not
    stop a run."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "version": 99,
        "tolerance_ms": 4.0,
        "some_future_setting": True,
        "preprocessing": {"enabled": True, "unknown_field": 1},
        "detectors": {"threshold": {"amplitude_sd": 5.5, "not_a_field": 2},
                      "a_detector_that_no_longer_exists": {"x": 1}},
    }))
    s = Settings.load(str(path))
    assert s.tolerance_ms == 4.0
    assert s.params_for("threshold").amplitude_sd == 5.5


def test_active_methods_respects_the_enable_flags():
    s = Settings()
    for name in detectors.names():
        s.enabled[name] = False
    s.enabled["deconvolution"] = True
    assert s.active_methods() == ["deconvolution"]


def test_field_specs_drive_the_parameter_window():
    """Every parameter must be discoverable, or the window silently omits it."""
    s = Settings()
    for name in detectors.names():
        specs = field_specs(s.params_for(name))
        assert specs, name
        got = {f for f, _, _, _ in specs}
        import dataclasses
        want = {f.name for f in dataclasses.fields(s.params_for(name))}
        assert got == want, name


def test_field_docstrings_become_tooltips():
    docs = {f: d for f, _, _, d in field_specs(Settings().params_for("deconvolution"))}
    assert "regularisation" in docs["reg"].lower()
    assert docs["thresh_sd"], "thresh_sd should carry its field docstring"


# ----------------------------------------------------------------- runner

def test_run_one_reports_an_unavailable_detector_instead_of_raising():
    import dataclasses

    trace, _ = synth_trace(duration_s=3.0, seed=1)
    s = Settings()
    s.set_params_for("miniml", dataclasses.replace(
        s.params_for("miniml"), python_exe="/definitely/not/here/python"))
    res = runner.run_one(trace, "miniml", s)
    assert res.ok is False
    assert res.events == []
    assert "python_exe" in res.message


def test_run_one_turns_a_detector_exception_into_a_failed_result(monkeypatch):
    trace, _ = synth_trace(duration_s=3.0, seed=1)

    def boom(self, trace, params):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(type(detectors.get("template")), "find", boom)
    res = runner.run_one(trace, "template", Settings())
    assert res.ok is False
    assert "synthetic failure" in res.message


def test_a_failing_detector_does_not_stop_the_others(tmp_path, monkeypatch):
    import dataclasses

    from multimethod import trace as trace_mod

    trace, _ = synth_trace(duration_s=15.0, seed=2)
    monkeypatch.setattr(trace_mod, "load", lambda *a, **k: trace)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)

    s = Settings()
    s.set_params_for("miniml", dataclasses.replace(
        s.params_for("miniml"), python_exe="/nope/python"))
    out = runner.run_all(str(tmp_path / "rec.abf"), 0, s,
                         methods=["threshold", "deconvolution", "miniml"])
    assert out.results["miniml"].ok is False
    assert out.results["threshold"].ok is True
    assert out.results["deconvolution"].ok is True
    assert out.ran == ["threshold", "deconvolution"]
    assert out.consensus, "the remaining methods must still form a consensus"


def test_run_all_writes_every_expected_file(tmp_path, monkeypatch):
    trace, _ = synth_trace(duration_s=15.0, seed=4)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)
    stem = str(tmp_path / "rec")
    out = runner.run_all(stem + ".abf", 0, Settings(),
                         methods=["threshold", "deconvolution"])
    written = out.write()
    names = {os.path.basename(p) for p in written}
    assert "rec_filt3000Hz10000Hz_multimethod_events.csv" in names
    assert "rec_filt3000Hz10000Hz_multimethod_summary.csv" in names
    assert "rec_filt3000Hz10000Hz_multimethod_agreement.csv" in names
    assert "rec_filt3000Hz10000Hz_multimethod_run.json" in names
    assert "rec_filt3000Hz10000Hz_threshold_events.csv" in names
    for p in written:
        assert os.path.getsize(p) > 0


def test_run_sidecar_records_enough_to_reproduce_the_run(tmp_path, monkeypatch):
    trace, _ = synth_trace(duration_s=12.0, seed=6)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)
    out = runner.run_all(str(tmp_path / "rec.abf"), 0, Settings(),
                         methods=["threshold"])
    out.write()
    with open(out.stem + runner.RUN_SUFFIX, encoding="utf-8") as fh:
        side = json.load(fh)
    assert side["settings"]["tolerance_ms"] == Settings().tolerance_ms
    assert "threshold" in side["methods"]
    assert side["methods"]["threshold"]["ok"] is True
    assert side["noise_sd"] > 0
    # arrays must not be dumped into the sidecar
    assert json.dumps(side)  # i.e. it is serialisable at all


def test_consensus_table_round_trips(tmp_path, monkeypatch):
    trace, _ = synth_trace(duration_s=12.0, seed=8)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)
    out = runner.run_all(str(tmp_path / "rec.abf"), 0, Settings(),
                         methods=["threshold", "template"])
    out.write()
    df = runner.load_consensus(out.stem)
    assert runner.methods_in_consensus(df) == ["threshold", "template"]
    assert len(df) == len(out.consensus)


# ----------------------------------------------------------- review window

def test_review_window_builds_and_autosaves(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets

    from multimethod import consensus as consensus_mod
    from multimethod.review import MethodReviewWindow, agreement_color

    trace, _ = synth_trace(duration_s=15.0, seed=10)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)
    s = Settings()
    out = runner.run_all(str(tmp_path / "rec.abf"), 0, s,
                         methods=["threshold", "deconvolution"])
    out.write()

    df = runner.load_consensus(out.stem)
    methods = runner.methods_in_consensus(df)
    per = {m: pd.read_csv(f"{out.stem}_{m}{runner.EVENTS_SUFFIX}") for m in methods}

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    reviewed = out.stem + runner.REVIEWED_SUFFIX
    progress = out.stem + runner.PROGRESS_SUFFIX
    w = MethodReviewWindow(trace, df, methods, per, s.measurement,
                           reviewed, progress, tolerance_ms=s.tolerance_ms)
    try:
        assert len(w.visible_ids) == len(df)
        w.accept_current()
        w.reject_current()
        assert len(w.decisions) == 2
        assert os.path.exists(progress) and os.path.exists(reviewed)
        # the filter narrows to events only one method found
        w.toggle_filter()
        assert all(int(df.at[i, "n_methods"]) == 1 for i in w.visible_ids)
        w.toggle_filter()
        assert len(w.visible_ids) == len(df)
    finally:
        w.window.close()


def test_agreement_colour_is_ordered_by_agreement():
    from multimethod.review import AGREEMENT_COLORS, agreement_color

    assert agreement_color(1, 4) == AGREEMENT_COLORS[0]
    assert agreement_color(4, 4) == AGREEMENT_COLORS[-1]
    assert agreement_color(1, 1) == AGREEMENT_COLORS[-1]


def test_prior_decisions_resume_from_the_progress_file(tmp_path):
    from multimethod.review import load_prior_decisions

    progress = tmp_path / "p.csv"
    pd.DataFrame({"event_id": [1, 2, 3], "decision": [1, 0, 1]}).to_csv(
        progress, index=False)
    got = load_prior_decisions(str(tmp_path / "missing.csv"), str(progress))
    assert got == {1: True, 2: False, 3: True}


def test_no_prior_decisions_is_an_empty_dict(tmp_path):
    from multimethod.review import load_prior_decisions

    assert load_prior_decisions(str(tmp_path / "a.csv"), str(tmp_path / "b.csv")) == {}
