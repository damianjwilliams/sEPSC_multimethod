"""Tests for the analysis figures and the review window's zoom behaviour.

The zoom is tested through `zoom_limits` / `zoom_axes` rather than by driving a
real window. That is not a workaround for a flaky GUI -- it is the better test:
the arithmetic (anchoring, direction, round-tripping) is the part that can be
wrong, and testing it directly makes the assertions exact rather than
approximate. The one thing that genuinely needs a window is the x-axis linkage
between the trace and the raster, and that is asserted structurally on the
shared-axes group, without scrolling anything.

(Driving the window with synthetic scroll events does work -- verified by hand
against a real recording -- but repeatedly building a QMainWindow with a
matplotlib canvas under Qt's offscreen platform crashes the interpreter
natively, with no Python traceback, so it is not something to build a suite on.)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from multimethod import plots, runner
from multimethod.benchmark import synth_trace
from multimethod.review import ZOOM_STEP, agreement_color, zoom_axes, zoom_limits
from multimethod.settings import Settings


# ------------------------------------------------------- zoom arithmetic

def test_zoom_in_narrows_and_zoom_out_widens():
    lo, hi = zoom_limits(0.0, 100.0, 50.0, zoom_in=True)
    assert hi - lo < 100.0
    lo2, hi2 = zoom_limits(lo, hi, 50.0, zoom_in=False)
    assert hi2 - lo2 > hi - lo


def test_zoom_in_then_out_returns_to_the_original_view():
    """A notch each way must land back where it started, or the view drifts as
    you scroll back and forth."""
    lo, hi = zoom_limits(*zoom_limits(0.0, 100.0, 30.0, True), 30.0, False)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(100.0, abs=1e-9)


def test_zoom_is_anchored_on_the_cursor_not_the_centre():
    """Zooming at the left edge keeps that edge put; a centre-anchored zoom
    would pull it inward symmetrically and slide the region of interest away."""
    lo, hi = zoom_limits(0.0, 100.0, 0.0, zoom_in=True)
    assert lo == pytest.approx(0.0, abs=1e-9), "the anchor should not move"
    assert hi < 100.0


def test_the_anchor_stays_inside_the_new_view():
    for anchor in (0.0, 1.0, 50.0, 99.0, 100.0):
        lo, hi = zoom_limits(0.0, 100.0, anchor, zoom_in=True)
        assert lo - 1e-9 <= anchor <= hi + 1e-9, anchor


def test_an_anchor_outside_the_range_is_clamped_not_followed():
    """A scroll event carrying stale coordinates must not throw the view
    somewhere unrelated."""
    lo, hi = zoom_limits(0.0, 100.0, 5000.0, zoom_in=True)
    assert lo >= 0.0 and hi <= 100.0 + 1e-9
    assert hi > lo


def test_the_zoom_factor_is_the_documented_step():
    lo, hi = zoom_limits(0.0, 100.0, 50.0, zoom_in=True)
    assert (hi - lo) == pytest.approx(100.0 / ZOOM_STEP)


def test_a_degenerate_range_is_returned_unchanged():
    assert zoom_limits(5.0, 5.0, 5.0, True) == (5.0, 5.0)
    assert zoom_limits(5.0, 1.0, 3.0, True) == (5.0, 1.0)


def test_scroll_modifiers_reads_the_qt_state_when_event_key_is_none():
    """The bug this fixes: matplotlib only fills MouseEvent.key from modifiers
    it saw in a key_press_event on that canvas, so holding ctrl and scrolling
    without clicking the canvas first gave key=None -- and both axes zoomed
    regardless of the modifier. The Qt event always knows."""
    import types

    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore

    from multimethod.review import scroll_modifiers

    class FakeQtEvent:
        def __init__(self, mods):
            self._mods = mods

        def modifiers(self):
            return self._mods

    cases = {
        QtCore.Qt.NoModifier: (True, True),
        QtCore.Qt.ControlModifier: (True, False),      # time only
        QtCore.Qt.ShiftModifier: (False, True),        # amplitude only
    }
    for mods, expected in cases.items():
        ev = types.SimpleNamespace(guiEvent=FakeQtEvent(mods), key=None)
        assert zoom_axes(scroll_modifiers(ev)) == expected, mods


def test_scroll_modifiers_falls_back_to_event_key():
    """Without a Qt event -- a synthetic event, or another backend -- the
    matplotlib key is still honoured."""
    import types

    from multimethod.review import scroll_modifiers

    assert scroll_modifiers(types.SimpleNamespace(guiEvent=None, key="shift")) == "shift"
    assert scroll_modifiers(types.SimpleNamespace(guiEvent=None, key=None)) == ""


def test_holding_both_modifiers_zooms_both_rather_than_nothing():
    """Asking for "time only" and "amplitude only" at once is a stray
    modifier, not a request to freeze the view, and a scroll that visibly does
    nothing reads as a bug."""
    assert zoom_axes("control+shift") == (True, True)


def test_modifiers_select_which_axes_zoom():
    assert zoom_axes(None) == (True, True), "no modifier: time and amplitude"
    assert zoom_axes("control") == (True, False), "ctrl: time only"
    assert zoom_axes("ctrl") == (True, False)
    assert zoom_axes("shift") == (False, True), "shift: amplitude only"
    assert zoom_axes("SHIFT") == (False, True), "must be case-insensitive"


# ------------------------------------------------------- drag-to-pan maths

def test_dragging_left_advances_the_view_and_right_rewinds_it():
    """The trace should feel stuck to the mouse: drag it left and you see
    later data, drag it right and you see earlier data."""
    from multimethod.review import pan_limits

    later = pan_limits(10.0, 20.0, press=15.0, cursor=13.0)   # dragged left
    assert later[0] > 10.0

    earlier = pan_limits(10.0, 20.0, press=15.0, cursor=17.0)  # dragged right
    assert earlier[0] < 10.0


def test_panning_never_changes_the_width_of_the_view():
    """A pan slides the window; only a zoom may resize it."""
    from multimethod.review import pan_limits

    for cursor in (5.0, 13.0, 15.0, 17.0, 40.0):
        lo, hi = pan_limits(10.0, 20.0, press=15.0, cursor=cursor)
        assert (hi - lo) == pytest.approx(10.0)


def test_panning_puts_the_pressed_point_back_under_the_cursor():
    """What makes the drag self-correcting: after the shift, the data
    coordinate under the pointer is the one that was there on press, so
    re-deriving the delta each motion event cannot accumulate drift."""
    from multimethod.review import pan_limits

    lo, hi = 10.0, 20.0
    press, cursor = 15.0, 12.0
    new_lo, new_hi = pan_limits(lo, hi, press, cursor)
    # the cursor sat at this fraction across the view; after the pan the point
    # at that same fraction must be the pressed coordinate
    frac = (cursor - lo) / (hi - lo)
    assert new_lo + frac * (new_hi - new_lo) == pytest.approx(press)


def test_a_zero_length_drag_leaves_the_view_alone():
    from multimethod.review import pan_limits

    assert pan_limits(10.0, 20.0, 15.0, 15.0) == (10.0, 20.0)


def test_plain_drag_pans_time_and_shift_drag_pans_amplitude():
    """Mirrors the zoom modifiers, so one rule covers both gestures."""
    from multimethod.review import pan_axes

    assert pan_axes(None) == (True, False), "plain drag: time"
    assert pan_axes("") == (True, False)
    assert pan_axes("shift") == (False, True), "shift+drag: amplitude"
    assert pan_axes("SHIFT") == (False, True)
    # ctrl is the zoom modifier, not a pan modifier; a drag stays on time
    assert pan_axes("control") == (True, False)


def test_pan_and_zoom_modifiers_agree_on_what_shift_means():
    """shift means "amplitude" for both gestures. If these ever disagree the
    window would zoom one axis and pan the other from the same modifier."""
    from multimethod.review import pan_axes

    assert zoom_axes("shift")[1] and pan_axes("shift")[1]
    assert not zoom_axes("shift")[0] and not pan_axes("shift")[0]


def test_the_click_slop_is_small_but_not_zero():
    """Zero would make every click a one-pixel pan; too large and a real drag
    would be swallowed as a click."""
    from multimethod.review import CLICK_SLOP_PX

    assert 1.0 <= CLICK_SLOP_PX <= 12.0


# The trace/raster x-axis linkage is asserted in
# test_runner_and_settings.py::test_review_window_builds_and_autosaves,
# which is the one place a review window is constructed. Building a
# second one in this file crashed the interpreter natively under Qt's
# offscreen platform, so the assertion lives with the existing window
# rather than with a new one.


# ------------------------------------------------------------- box + dots

def _new_ax():
    from matplotlib.figure import Figure

    return Figure().subplots()


def _dot_count(ax) -> int:
    return sum(int(np.asarray(l.get_xdata()).size) for l in ax.lines
               if l.get_linestyle() == "None")


def test_box_with_dots_draws_a_box_and_every_point():
    rng = np.random.default_rng(0)
    ax = _new_ax()
    plots.box_with_dots(ax, [rng.normal(20, 5, 40), rng.normal(14, 4, 25)],
                        ["a", "b"])
    assert len(ax.patches) == 2, "expected one box per group"
    assert _dot_count(ax) == 65, "every measurement should be drawn as a dot"
    assert "n=40" in ax.get_xticklabels()[0].get_text()


def test_box_with_dots_subsamples_only_the_dots_not_the_statistics():
    rng = np.random.default_rng(1)
    ax = _new_ax()
    plots.box_with_dots(ax, [rng.normal(20, 5, 5000)], ["big"], max_dots=100)
    assert _dot_count(ax) == 100, "dots should be capped for legibility"
    assert "n=5000" in ax.get_xticklabels()[0].get_text(), (
        "the reported n must come from every point, not the drawn subset")


def test_box_with_dots_survives_an_empty_group():
    ax = _new_ax()
    plots.box_with_dots(ax, [np.array([]), np.array([1.0, 2.0, 3.0])],
                        ["none", "some"])
    assert "n=0" in ax.get_xticklabels()[0].get_text()


def test_nan_values_are_excluded_from_the_reported_n():
    # the first inter-event interval is always NaN, so this case is guaranteed
    ax = _new_ax()
    plots.box_with_dots(ax, [np.array([np.nan, 1.0, 2.0])], ["iei"])
    assert "n=2" in ax.get_xticklabels()[0].get_text()


# ---------------------------------------------------------- the figure set

def _finished_run(tmp_path, monkeypatch, methods=("threshold", "deconvolution")):
    trace, _ = synth_trace(duration_s=20.0, rate_hz=8.0, seed=5)
    monkeypatch.setattr(runner, "load", lambda *a, **k: trace)
    return runner.run_all(str(tmp_path / "rec.abf"), 0, Settings(),
                          methods=list(methods))


def test_write_all_produces_every_figure(tmp_path, monkeypatch):
    out = _finished_run(tmp_path, monkeypatch)
    written = plots.write_all(out)
    names = {os.path.basename(p) for p in written}
    for suffix in ("distributions", "boxplots", "waveforms", "overview"):
        assert any(suffix in n for n in names), suffix
    for p in written:
        assert os.path.getsize(p) > 5000, f"{p} looks empty"


def test_figure_names_do_not_collide_with_the_csv_outputs(tmp_path, monkeypatch):
    """Both `_multimethod_summary.csv` and `_multimethod_agreement.csv` exist,
    so no figure may take either stem -- otherwise one reads as a rendering of
    the other. This caught exactly that collision when the figures were added.
    """
    out = _finished_run(tmp_path, monkeypatch)
    csvs = {os.path.splitext(os.path.basename(p))[0] for p in out.write()}
    pngs = {os.path.splitext(os.path.basename(p))[0] for p in plots.write_all(out)}
    assert not (csvs & pngs), f"figure and table share a stem: {csvs & pngs}"


def test_plots_never_switch_the_global_matplotlib_backend():
    """A library function calling matplotlib.use() sets a PROCESS-WIDE backend
    and takes down a Qt window built in the same interpreter. This module
    constructs Figure objects directly instead, so it needs no backend."""
    import ast

    # Parse rather than grep: the module's own docstring explains that it must
    # not call matplotlib.use(), and a substring check matches that comment.
    tree = ast.parse(open(plots.__file__, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for c in calls:
        name = ast.unparse(c.func) if hasattr(ast, "unparse") else ""
        assert name != "matplotlib.use", "plots.py must not set a global backend"
    imports = [ast.unparse(n) for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("pyplot" in i for i in imports), (
        f"plots.py must not import pyplot: {[i for i in imports if 'pyplot' in i]}")


def test_write_all_with_no_successful_method_writes_nothing(tmp_path, monkeypatch):
    out = _finished_run(tmp_path, monkeypatch)
    out.results.clear()
    out.consensus = []
    assert plots.write_all(out) == []


def test_method_colours_are_stable_and_distinct():
    used = [plots.method_color(m) for m in
            ("threshold", "deconvolution", "template", "miniml", "fastmini")]
    assert len(set(used)) == len(used), "each method needs its own colour"
    assert plots.method_color("threshold") == plots.method_color("threshold")


def test_agreement_colour_is_ordered_by_agreement():
    from multimethod.review import AGREEMENT_COLORS

    assert agreement_color(1, 4) == AGREEMENT_COLORS[0]
    assert agreement_color(4, 4) == AGREEMENT_COLORS[-1]
