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


def test_every_parameter_carries_an_explanation():
    """The parameter window shows each field's docstring as visible help, so a
    field without one appears in the window as a bare number with no
    explanation. 37 of 64 were undocumented when the window was first built."""
    s = Settings()
    targets = [("preprocessing", s.preprocessing), ("measurement", s.measurement)]
    targets += [(n, s.params_for(n)) for n in detectors.names()]
    undocumented = []
    for name, obj in targets:
        for field, _, _, doc in field_specs(obj):
            if not (doc or "").strip():
                undocumented.append(f"{name}.{field}")
    assert not undocumented, (
        f"{len(undocumented)} parameter(s) have no explanation: {undocumented}")


def test_every_detector_has_parameter_window_guidance():
    """Each detector tab opens with what the method is good and bad at."""
    from multimethod.paramwindow import DETECTOR_NOTES

    for name in detectors.names():
        assert name in DETECTOR_NOTES, f"{name} has no parameter-window notes"
        assert len(DETECTOR_NOTES[name]) > 100, name


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

        # The linkage behind "zoom the trace and the raster follows": the two
        # share an x axis, and the detail panel deliberately does not (it
        # shows one event, not the run).
        shared = w.ax_over.get_shared_x_axes()
        assert shared.joined(w.ax_over, w.ax_raster), (
            "the raster must share the trace's x axis, or zooming the trace "
            "will not narrow the per-method ticks with it")
        assert not shared.joined(w.ax_over, w.ax_detail)
        w.ax_over.set_xlim(2.0, 4.0)
        assert w.ax_raster.get_xlim() == pytest.approx((2.0, 4.0))
        w.reset_view()

        # Drag-to-pan, and the click/drag distinction that protects it.
        import types

        def mouse(ax, xdata, ydata, x, y, button=1, key=None):
            return types.SimpleNamespace(inaxes=ax, xdata=xdata, ydata=ydata,
                                         x=x, y=y, button=button, key=key,
                                         guiEvent=None)

        w.ax_over.set_xlim(2.0, 6.0)
        before = w.ax_over.get_xlim()
        w.on_press(mouse(w.ax_over, 4.0, -60.0, 500, 300))
        w.on_motion(mouse(w.ax_over, 3.0, -60.0, 440, 300))
        after = w.ax_over.get_xlim()
        assert after[0] > before[0], "dragging left should advance the view"
        assert (after[1] - after[0]) == pytest.approx(before[1] - before[0]), (
            "a pan must not resize the view")
        assert w.ax_raster.get_xlim() == pytest.approx(after), (
            "the raster must follow a pan, as it follows a zoom")
        w.on_release(mouse(w.ax_over, 3.0, -60.0, 440, 300))

        # A press that does not move is a click: it selects, it does not pan.
        w.reset_view()
        w.show_event(0)
        held = w.ax_over.get_xlim()
        target = float(df.at[len(df) // 2, "peak_time_s"])
        w.on_press(mouse(w.ax_over, target, -60.0, 500, 300))
        w.on_release(mouse(w.ax_over, target, -60.0, 501, 300))
        assert w.ax_over.get_xlim() == pytest.approx(held), "a click must not pan"
        assert abs(float(df.at[w.current, "peak_time_s"]) - target) < 0.05, (
            "a click should jump to the nearest event")

        # ...and a drag must not steal the selection, even if it starts on a
        # marker: pick_event fires on PRESS, before a drag is distinguishable.
        selected = w.current
        w.on_press(mouse(w.ax_over, target, -60.0, 500, 300))
        w.on_motion(mouse(w.ax_over, target - 1.0, -60.0, 400, 300))
        w.on_release(mouse(w.ax_over, target - 1.0, -60.0, 400, 300))
        assert w.current == selected, "a pan must not change the selected event"

        w.reset_view()
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
