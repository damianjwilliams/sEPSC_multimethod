"""The review window: which method found which event, on the trace itself.

Three panels, and the middle one is the point of this package.

  top     the whole trace, every consensus event marked, coloured by HOW MANY
          methods found it. Unanimous events are the boring ones; the
          interesting ones are found by exactly one method, and they are the
          brightest.

  middle  a raster, one row per method, one tick per detection. Read down a
          column and you can see at a glance which methods fired on a given
          event and which missed it. This is the view that answers "is
          method X finding extra events, or are the others missing them?" --
          a question no summary statistic can answer, because the answer is
          different for different events in the same recording.

  bottom  the selected event, zoomed, with the measurement windows drawn, and
          a per-method line: which methods found it, each one's amplitude, and
          how far each one's peak sat from the consensus time.

Accept/reject writes the same two files the sibling repos use, so a curated
event list is interchangeable with theirs.

Both the trace and the detail panel expand in time and in amplitude, and
the two axes are independently controllable:

    scroll           zoom time and amplitude together
    ctrl  + scroll   stretch the TIME base only
    shift + scroll   stretch the AMPLITUDE axis only

Click and hold anywhere on the trace and move the mouse left or right to slide
along the recording -- no toolbar tool to select first. shift+drag moves the
amplitude axis instead. A press that does not move is still a click, and jumps
to the nearest event, so dragging never costs you a selection.

Home (or the 0 key) resets. The raster shares the trace's x axis, so zooming or
dragging up top moves the raster with it and the per-method ticks stay aligned
underneath.

Design note: colouring by agreement count rather than by method is deliberate.
With five methods there are 31 possible combinations, and a colour per
combination is unreadable. Agreement count is one ordered number, it maps onto
a sequential colour map, and it is what you actually scan for.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import runner

# Sequential, colour-blind-safe, dark = more agreement.
AGREEMENT_COLORS = ["#d94801", "#fd8d3c", "#6baed6", "#3182bd", "#08519c", "#062f5c"]
INK = "#1b1b1b"
MUTED = "#8a8a8a"
TRACE_COLOR = "#3a3a3a"
ACCEPT_COLOR = "#2a7f2a"
REJECT_COLOR = "#a33"
OVERVIEW_MAX_POINTS = 200_000


def agreement_color(n: int, n_methods: int) -> str:
    """Colour for an event found by `n` of `n_methods` methods."""
    if n_methods <= 1:
        return AGREEMENT_COLORS[-1]
    frac = (n - 1) / max(n_methods - 1, 1)
    idx = int(round(frac * (len(AGREEMENT_COLORS) - 1)))
    return AGREEMENT_COLORS[min(max(idx, 0), len(AGREEMENT_COLORS) - 1)]


#: One scroll notch. Small enough to feel continuous, large enough that a few
#: notches cross a useful range on a ten-minute recording.
ZOOM_STEP = 1.15


def zoom_limits(lo: float, hi: float, anchor: float, zoom_in: bool,
                step: float = ZOOM_STEP) -> tuple[float, float]:
    """New (lo, hi) after one scroll notch, anchored on `anchor`.

    Anchored rather than centred so zooming into a burst keeps the burst under
    the pointer instead of sliding it out of view. `anchor` is clamped into
    the current range, so a scroll event with stale coordinates cannot throw
    the view somewhere unrelated.

    Kept as a free function, with no axes and no Qt, so the behaviour is
    testable without constructing a window.
    """
    if not (hi > lo):
        return lo, hi
    anchor = min(max(anchor, lo), hi)
    scale = 1.0 / step if zoom_in else step
    return (anchor - (anchor - lo) * scale,
            anchor + (hi - anchor) * scale)


def scroll_modifiers(event) -> str:
    """Which modifier keys were held during a scroll, as a lowercase string.

    Reads the Qt event first, and only then falls back to matplotlib's
    `event.key`.

    That order is the whole reason this function exists. matplotlib populates
    MouseEvent.key only from modifiers it has already seen in a key_press_event
    on that canvas, so holding ctrl and scrolling -- without having clicked the
    canvas first to give it keyboard focus -- yields key=None and the modifier
    is silently ignored, leaving both axes zooming together. The Qt event
    carries the real modifier state unconditionally, so it is authoritative
    whenever it is present.
    """
    gui = getattr(event, "guiEvent", None)
    if gui is not None and hasattr(gui, "modifiers"):
        try:
            from PyQt5 import QtCore

            mods = gui.modifiers()
            parts = []
            if mods & QtCore.Qt.ControlModifier:
                parts.append("control")
            if mods & QtCore.Qt.ShiftModifier:
                parts.append("shift")
            if mods & QtCore.Qt.AltModifier:
                parts.append("alt")
            return "+".join(parts)
        except Exception:            # noqa: BLE001 - fall back to event.key
            pass
    return (getattr(event, "key", None) or "").lower()


#: How far the pointer may move between press and release and still count as a
#: click rather than a drag, in pixels. Small enough that a deliberate drag is
#: never mistaken for a click; large enough to tolerate the hand-shake in a
#: click on a trackpad.
CLICK_SLOP_PX = 4.0


def pan_limits(lo: float, hi: float, press: float, cursor: float
               ) -> tuple[float, float]:
    """New (lo, hi) while dragging, so `press` stays under the pointer.

    `press` is the data coordinate that was under the cursor when the drag
    began; `cursor` is the data coordinate under it now. Shifting the limits by
    their difference puts `press` back under the pointer, which is what makes a
    drag feel like the trace is stuck to the mouse.

    Self-correcting rather than cumulative: after each step the point under the
    cursor is `press` again, so re-deriving the delta from the current view on
    every motion event cannot accumulate drift.
    """
    delta = press - cursor
    return lo + delta, hi + delta


def pan_axes(key: str | None) -> tuple[bool, bool]:
    """Which axes a drag should pan, given the held modifier.

    Deliberately the mirror of `zoom_axes`, so one rule covers both gestures:
    plain drag moves along TIME, shift+drag moves AMPLITUDE. Dragging pans one
    axis at a time on purpose -- a free two-axis drag on a trace that is 60 s
    wide and 100 pA tall slews wildly in whichever direction has more data per
    pixel, and you lose your place.
    """
    k = (key or "").lower()
    return (not ("shift" in k), "shift" in k)


def zoom_axes(key: str | None) -> tuple[bool, bool]:
    """Which axes a scroll should zoom, given the held modifier(s).

    Returns (zoom_time, zoom_amplitude):

        (none)   both axes together
        ctrl     TIME only       -- stretch the time base, amplitude fixed
        shift    AMPLITUDE only  -- stretch the current axis, time fixed

    matching the checker windows in the sibling repos.

    Holding ctrl AND shift zooms both, rather than nothing. Asking for "time
    only" and "amplitude only" at once is a stray modifier, not a request to
    freeze the view, and a scroll that visibly does nothing reads as a bug.
    """
    k = (key or "").lower()
    ctrl = "control" in k or "ctrl" in k
    shift = "shift" in k
    if ctrl and shift:
        return True, True
    return (not shift, not ctrl)


def _closed_hand_cursor():
    """A grabbing-hand cursor, so a drag looks like it has taken hold."""
    from PyQt5 import QtCore

    return QtCore.Qt.ClosedHandCursor


def _arrow_cursor():
    from PyQt5 import QtCore

    return QtCore.Qt.ArrowCursor


def load_prior_decisions(reviewed_path: str, progress_path: str) -> dict:
    """{event_id: accepted} from a previous session, so QC resumes."""
    if os.path.exists(progress_path):
        prior = pd.read_csv(progress_path)
        return dict(zip(prior["event_id"].astype(int), prior["decision"] == 1))
    if os.path.exists(reviewed_path):
        prior = pd.read_csv(reviewed_path)
        return dict(zip(prior["event_id"].astype(int), [True] * len(prior)))
    return {}


class MethodReviewWindow:
    """Built on a Qt QMainWindow with matplotlib canvases.

    Deliberately not pyplot: pyplot's backend is a global, and the runner
    forces the non-interactive Agg backend for its headless plotting. Building
    the Qt window explicitly means import order cannot downgrade this window.
    """

    def __init__(self, trace, consensus_df: pd.DataFrame, methods: list,
                 per_method_df: dict, measurement, reviewed_path: str,
                 progress_path: str, title: str = "Method comparison",
                 tolerance_ms: float = 3.0):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        from matplotlib.figure import Figure
        from PyQt5 import QtWidgets

        self.trace = trace
        self.df = consensus_df.reset_index(drop=True)
        self.methods = list(methods)
        self.per_method_df = per_method_df
        self.measurement = measurement
        self.tolerance_ms = tolerance_ms
        self.reviewed_path = reviewed_path
        self.progress_path = progress_path
        self.decisions = load_prior_decisions(reviewed_path, progress_path)
        self.current = 0

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(title)
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.figure = Figure(figsize=(15, 9), facecolor="#faf9f5")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self.window))
        layout.addWidget(self.canvas)

        gs = self.figure.add_gridspec(
            3, 1, height_ratios=[2.4, 1.0, 2.4], hspace=0.45,
            left=0.13, right=0.80, top=0.96, bottom=0.07)
        self.ax_over = self.figure.add_subplot(gs[0])
        self.ax_raster = self.figure.add_subplot(gs[1], sharex=self.ax_over)
        self.ax_detail = self.figure.add_subplot(gs[2])

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        row = QtWidgets.QHBoxLayout()
        for text, slot, color in (
                ("Accept  (A)", self.accept_current, ACCEPT_COLOR),
                ("Reject  (R)", self.reject_current, REJECT_COLOR),
                ("Accept all unanimous", self.accept_all_unanimous, None),
                ("Only 1-method events", self.toggle_filter, None)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            if color:
                b.setStyleSheet(f"font-weight:600; color:{color};")
            row.addWidget(b)
            if text.startswith("Only"):
                self.filter_btn = b
        layout.addLayout(row)

        # Drag-to-pan state. `_press` holds where a press landed, in data
        # coordinates, so a click can be told from a drag on release.
        self._press = None
        self._dragged = False

        self.only_unique = False
        self._draw_overview()
        self._draw_raster()
        self.show_event(0)

        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    # -- data helpers ---------------------------------------------------

    @property
    def visible_ids(self) -> list:
        if not self.only_unique:
            return list(self.df.index)
        return [i for i in self.df.index if int(self.df.at[i, "n_methods"]) == 1]

    def _methods_for(self, i: int) -> list:
        raw = self.df.at[i, "methods"]
        return [] if pd.isna(raw) else str(raw).split(";")

    # -- drawing --------------------------------------------------------

    def _decimated(self):
        v = np.asarray(self.trace.v)
        step = max(1, v.size // OVERVIEW_MAX_POINTS)
        return np.asarray(self.trace.t)[::step], v[::step]

    def _draw_overview(self):
        ax = self.ax_over
        ax.clear()
        t, v = self._decimated()
        ax.plot(t, v, lw=0.4, color=TRACE_COLOR, zorder=1)

        n_m = len(self.methods)
        for n in range(1, n_m + 1):
            sel = self.df[self.df["n_methods"] == n]
            if sel.empty:
                continue
            idx = sel["peak_idx"].to_numpy(dtype=int)
            idx = idx[(idx >= 0) & (idx < self.trace.v.size)]
            ax.plot(np.asarray(self.trace.t)[idx], np.asarray(self.trace.v)[idx],
                    "v", ms=6, mew=0, color=agreement_color(n, n_m),
                    label=f"{n} of {n_m} method{'s' if n > 1 else ''} "
                          f"({len(sel)})",
                    picker=6, zorder=3)
        ax.set_ylabel(f"current ({self.trace.unit})", color=INK)
        ax.set_title(
            f"{len(self.df)} consensus events, coloured by how many methods "
            f"found each — click one", loc="left", fontsize=10, color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  frameon=True, title="agreement", title_fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _draw_raster(self):
        """One row per method, one tick per detection of that method."""
        ax = self.ax_raster
        ax.clear()
        for k, m in enumerate(self.methods):
            col = f"found_{m}"
            if col not in self.df.columns:
                continue
            hits = self.df[self.df[col] == 1]
            if hits.empty:
                continue
            times = hits["peak_time_s"].to_numpy(dtype=float)
            shared = hits["n_methods"].to_numpy(dtype=int)
            colors = [agreement_color(n, len(self.methods)) for n in shared]
            ax.scatter(times, np.full(times.size, k), s=9, marker="|",
                       linewidths=1.1, c=colors, zorder=3)
        ax.set_yticks(range(len(self.methods)))
        ax.set_yticklabels(
            [f"{m}  ({int(self.df.get(f'found_{m}', pd.Series(dtype=int)).sum())})"
             for m in self.methods], fontsize=8)
        ax.set_ylim(-0.7, len(self.methods) - 0.3)
        ax.invert_yaxis()
        ax.set_xlabel("time (s)", color=INK)
        ax.set_title("per-method detections — read down a column to see who "
                     "agreed", loc="left", fontsize=9.5, color=MUTED)
        ax.grid(axis="y", color="#e6e4dc", lw=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        self._cursor = ax.axvline(0.0, color="#c026a3", lw=1.2, ls="--", zorder=5)

    def show_event(self, i: int):
        vis = self.visible_ids
        if not vis:
            self.status.setText("No events match the current filter.")
            self.canvas.draw_idle()
            return
        i = int(np.clip(i, 0, len(self.df) - 1))
        if i not in vis:
            later = [j for j in vis if j >= i]
            i = later[0] if later else vis[-1]
        self.current = i
        self._draw_detail(i)
        self._cursor.set_xdata([self.df.at[i, "peak_time_s"]] * 2)
        self._update_status(i)
        self.canvas.draw_idle()

    def _draw_detail(self, i: int):
        from .events import measure_at

        ax = self.ax_detail
        ax.clear()
        m = self.measurement
        peak = int(self.df.at[i, "peak_idx"])
        pre = max(1, int(round((m.baseline_before_ms + m.baseline_avg_ms + 5.0)
                               / 1000.0 / self.trace.dt)))
        post = max(1, int(round((m.decay_search_ms + 5.0) / 1000.0 / self.trace.dt)))
        lo, hi = max(0, peak - pre), min(self.trace.v.size, peak + post)
        t_ms = (np.arange(lo, hi) - peak) * self.trace.dt * 1e3
        seg = np.asarray(self.trace.v)[lo:hi]
        ax.plot(t_ms, seg, lw=0.9, color=TRACE_COLOR, zorder=2)

        ev = measure_at(self.trace.v, self.trace.dt, peak, m)
        if ev is not None:
            ax.axhline(ev.baseline, color="#2a8f2a", ls="--", lw=1.1, zorder=1,
                       label="baseline used")
            ax.axvspan(-(m.baseline_before_ms + m.baseline_avg_ms),
                       -m.baseline_before_ms, color="#7fbf7f", alpha=0.30,
                       zorder=0, label="baseline window")
            ax.axvspan(0, m.decay_search_ms, color="#f0c674", alpha=0.18,
                       zorder=0, label="decay search")
            ax.plot(0, ev.baseline + ev.amplitude, "x", ms=10, mew=2,
                    color="crimson", zorder=4, label="consensus peak")

        # Each method's own peak position, so a timing disagreement between
        # methods is visible rather than hidden by the consensus average.
        # One row per method, stacked: with four methods agreeing to within a
        # sample, four labels at the same x are unreadable otherwise.
        found = self._methods_for(i)
        # Reserve headroom ABOVE the trace for the per-method rows, rather than
        # drawing them over the signal -- on a small event the labels sat right
        # on top of the deflection being judged.
        y0, y1 = ax.get_ylim()
        pad = 0.16 * (y1 - y0) * max(len(self.methods), 1) / 4.0
        ax.set_ylim(y0, y1 + pad)
        y0, y1 = ax.get_ylim()
        for k, meth in enumerate(self.methods):
            y = y1 - (0.04 + 0.055 * k) * (y1 - y0)
            if meth not in found:
                ax.text(t_ms[0], y, f"{meth}  —", fontsize=7.5, color="#c4c4c4",
                        ha="left", va="center", zorder=4)
                continue
            off = self._method_offset_ms(i, meth)
            if off is None:
                off = 0.0
            ax.plot([off], [y], marker="|", ms=11, mew=2.0,
                    color=AGREEMENT_COLORS[-1], zorder=5)
            ax.text(t_ms[0], y, f"{meth}  {off:+.1f} ms", fontsize=7.5,
                    color=INK, ha="left", va="center", zorder=5)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("time from consensus peak (ms)", color=INK)
        ax.set_ylabel(f"current ({self.trace.unit})", color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  frameon=True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _method_offset_ms(self, i: int, method: str):
        """Where `method` put this event's peak, relative to the consensus time.

        Read from that method's own events CSV when available: the consensus
        table keeps amplitudes but not each method's individual peak time, and
        a systematic timing offset between methods is worth seeing.
        """
        df = self.per_method_df.get(method)
        if df is None or df.empty:
            return 0.0
        t0 = float(self.df.at[i, "peak_time_s"])
        d = (df["peak_time_s"].to_numpy(dtype=float) - t0) * 1e3
        if d.size == 0:
            return None
        j = int(np.argmin(np.abs(d)))
        return float(d[j]) if abs(d[j]) <= self.tolerance_ms * 2 else None

    def _update_status(self, i: int):
        found = self._methods_for(i)
        n_m = len(self.methods)
        decided = self.decisions.get(int(self.df.at[i, "event_id"]))
        qc = "undecided" if decided is None else ("ACCEPTED" if decided else "REJECTED")
        bits = []
        for meth in self.methods:
            col = f"amp_{meth}"
            if meth in found and col in self.df.columns:
                amp = self.df.at[i, col]
                off = self._method_offset_ms(i, meth)
                off_txt = f" @{off:+.1f} ms" if off is not None else ""
                bits.append(f"<b>{meth}</b> {amp:.1f}{off_txt}")
            else:
                bits.append(f"<span style='color:{MUTED}'>{meth} —</span>")
        vis = self.visible_ids
        pos = vis.index(i) + 1 if i in vis else 0
        self.status.setText(
            f"Event {pos}/{len(vis)} (id {int(self.df.at[i, 'event_id'])}) &nbsp; "
            f"t={self.df.at[i, 'peak_time_s']:.3f}s &nbsp; "
            f"found by {len(found)}/{n_m} &nbsp; QC: {qc}<br>"
            + " &nbsp;|&nbsp; ".join(bits))

    # -- interaction ----------------------------------------------------

    def on_pick(self, event):
        """Kept for the marker artists, but selection is driven by on_release.

        pick_event fires on button PRESS, before it is known whether the press
        will become a drag -- so selecting here would change the current event
        every time a pan happened to start on a marker. on_release knows.
        """
        return

    def on_scroll(self, event):
        """Scroll to zoom, about the cursor, on whichever panel it is over.

        Modifiers match the checker windows in the sibling repos:
            (none)   zoom time and amplitude together
            ctrl     time only
            shift    amplitude only

        Zooming the overview also zooms the raster, because the two share an x
        axis -- which is the point of sharing it. Scrolling the raster does the
        same in reverse. The detail panel has its own axes and zooms alone.

        Anchored on the cursor rather than the axes centre so zooming in on a
        burst keeps the burst under the pointer instead of sliding it away.
        """
        ax = event.inaxes
        if ax is None:
            return
        zoom_in = event.button == "up"
        zoom_x, zoom_y = zoom_axes(scroll_modifiers(event))
        if ax is self.ax_raster:
            zoom_y = False              # its y axis is one row per method

        if zoom_x and event.xdata is not None:
            ax.set_xlim(*zoom_limits(*ax.get_xlim(), event.xdata, zoom_in))
        if zoom_y and event.ydata is not None:
            ax.set_ylim(*zoom_limits(*ax.get_ylim(), event.ydata, zoom_in))
        self.canvas.draw_idle()

    def _toolbar_busy(self) -> bool:
        """True while the toolbar's own pan or zoom tool is armed.

        Without this check, dragging with the toolbar pan tool selected would
        pan twice as fast -- once from the toolbar and once from here.
        """
        toolbar = getattr(self.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", "") or "")

    def on_press(self, event):
        """Record where a left press landed, for panning or for selecting."""
        if event.button != 1 or event.inaxes is None or self._toolbar_busy():
            return
        if event.xdata is None:
            return
        self._press = (event.inaxes, event.xdata, event.ydata,
                       event.x, event.y, event.inaxes.get_xlim(),
                       event.inaxes.get_ylim())
        self._dragged = False

    def on_motion(self, event):
        """Pan while the button is held, so the trace follows the mouse."""
        if self._press is None or event.inaxes is not self._press[0]:
            return
        if event.xdata is None:
            return
        ax, px, py, sx, sy, _x0, _y0 = self._press
        if not self._dragged:
            # Below the slop this is still a click, not a drag. Waiting keeps a
            # click on an event marker from nudging the view a pixel.
            if abs(event.x - sx) < CLICK_SLOP_PX and abs(event.y - sy) < CLICK_SLOP_PX:
                return
            self._dragged = True
            self.canvas.setCursor(_closed_hand_cursor())

        pan_x, pan_y = pan_axes(scroll_modifiers(event))
        if ax is self.ax_raster:
            pan_y = False               # its y axis is one row per method
        if pan_x:
            ax.set_xlim(*pan_limits(*ax.get_xlim(), px, event.xdata))
        if pan_y and event.ydata is not None and py is not None:
            ax.set_ylim(*pan_limits(*ax.get_ylim(), py, event.ydata))
        self.canvas.draw_idle()

    def on_release(self, event):
        """A drag ends the pan; a click without a drag selects an event."""
        if self._press is None:
            return
        ax, _px, _py, _sx, _sy, _x0, _y0 = self._press
        was_drag = self._dragged
        self._press = None
        self._dragged = False
        self.canvas.setCursor(_arrow_cursor())
        if was_drag or event.button != 1:
            return
        # A plain click on the trace or the raster jumps to the nearest event.
        # Handled here rather than through pick_event so that a drag which
        # happens to start on an event marker pans instead of selecting.
        if ax in (self.ax_over, self.ax_raster) and event.xdata is not None:
            times = self.df["peak_time_s"].to_numpy(dtype=float)
            if times.size:
                self.show_event(int(np.argmin(np.abs(times - event.xdata))))

    def reset_view(self, *_a):
        """Back to the full trace, both axes, both linked panels."""
        self._draw_overview()
        self._draw_raster()
        self.show_event(self.current)

    def on_key(self, event):
        key = (event.key or "").lower()
        vis = self.visible_ids
        if not vis:
            return
        if key in ("right", "n"):
            nxt = [j for j in vis if j > self.current]
            self.show_event(nxt[0] if nxt else vis[0])
        elif key in ("left", "p"):
            prv = [j for j in vis if j < self.current]
            self.show_event(prv[-1] if prv else vis[-1])
        elif key in ("a", "up"):
            self.accept_current()
        elif key in ("r", "down"):
            self.reject_current()
        elif key in ("home", "0"):
            self.reset_view()

    def _decide(self, accepted: bool):
        self.decisions[int(self.df.at[self.current, "event_id"])] = bool(accepted)
        self._autosave()
        vis = self.visible_ids
        nxt = [j for j in vis if j > self.current]
        self.show_event(nxt[0] if nxt else self.current)

    def accept_current(self, *_a):
        self._decide(True)

    def reject_current(self, *_a):
        self._decide(False)

    def accept_all_unanimous(self, *_a):
        """Accept every event that every method found.

        The safe bulk action: unanimous events are the ones least likely to
        need an eye. Everything else stays undecided.
        """
        from PyQt5 import QtWidgets

        n_m = len(self.methods)
        ids = [int(self.df.at[i, "event_id"]) for i in self.df.index
               if int(self.df.at[i, "n_methods"]) == n_m]
        if not ids:
            QtWidgets.QMessageBox.information(
                self.window, "None", "No event was found by every method.")
            return
        if QtWidgets.QMessageBox.question(
                self.window, "Accept unanimous",
                f"Accept {len(ids)} event(s) found by all {n_m} methods?"
                ) != QtWidgets.QMessageBox.Yes:
            return
        for e in ids:
            self.decisions[e] = True
        self._autosave()
        self.show_event(self.current)

    def toggle_filter(self, *_a):
        self.only_unique = not self.only_unique
        self.filter_btn.setText("All events" if self.only_unique
                                else "Only 1-method events")
        vis = self.visible_ids
        self.show_event(vis[0] if vis else 0)

    def _autosave(self):
        """Write after every decision, so an interrupted pass loses nothing."""
        decided = self.df[self.df["event_id"].isin(self.decisions.keys())].copy()
        if decided.empty:
            return
        decided["decision"] = decided["event_id"].map(
            lambda e: int(self.decisions[int(e)]))
        decided.to_csv(self.progress_path, index=False)
        (decided[decided["decision"] == 1].drop(columns="decision")
         .to_csv(self.reviewed_path, index=False))

    def show(self):
        self.window.showMaximized()


def open_review(abf_path: str, channel: int = 0, settings=None,
                only_unique: bool = False):
    """Load a finished run and open the review window on it."""
    from PyQt5 import QtWidgets

    from .settings import Settings
    from .trace import load, output_stem

    s = settings or Settings()
    stem = output_stem(abf_path, channel, s.preprocessing)
    cons_path = stem + runner.CONSENSUS_SUFFIX
    if not os.path.exists(cons_path):
        raise FileNotFoundError(
            f"no consensus table at {cons_path}\n"
            f"Run detection first:  python run_all.py {abf_path}")

    df = pd.read_csv(cons_path, float_precision="round_trip")
    methods = runner.methods_in_consensus(df)
    per_method = {}
    for m in methods:
        p = f"{stem}_{m}{runner.EVENTS_SUFFIX}"
        if os.path.exists(p):
            per_method[m] = pd.read_csv(p, float_precision="round_trip")

    trace = load(abf_path, channel, s.preprocessing)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MethodReviewWindow(
        trace, df, methods, per_method, s.measurement,
        reviewed_path=stem + runner.REVIEWED_SUFFIX,
        progress_path=stem + runner.PROGRESS_SUFFIX,
        title=f"Method comparison — {os.path.basename(abf_path)}",
        tolerance_ms=s.tolerance_ms)
    win.only_unique = only_unique
    win.show()
    app.exec_()
    return win
