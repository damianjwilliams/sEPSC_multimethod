"""The figures a synaptic-event analysis is normally reported with.

Four figures, written at the end of a run:

    <stem>_multimethod_distributions.png   amplitude / IEI / rise / decay,
                                           histogram + cumulative, per method
    <stem>_multimethod_boxplots.png        per-method box + whisker + every
                                           individual event as a dot
    <stem>_multimethod_waveforms.png       peak-aligned average event per
                                           method, raw and peak-normalised
    <stem>_multimethod_overview.png        stability over time, agreement
                                           histogram, pairwise agreement matrix,
                                           amplitude by agreement level

Why box-and-whisker with dots instead of bar charts
---------------------------------------------------
A bar of the mean with an error bar hides everything that matters about a
synaptic amplitude distribution: it is skewed, it is truncated at the
detection limit, and its interesting feature is usually a change in shape
rather than in the mean. Two very different distributions can produce the same
bar. So anywhere there is a distribution to show, this module draws a box
(median and IQR), whiskers, and every individual measurement as a jittered
dot, so n and spread and skew are all visible at once.

Bars survive in exactly one place -- counts, in the agreement histogram. A
count of events has no distribution to draw, and a box plot of one number
would be a lie about the data rather than an improvement on the bar.
"""

# NOTE: this module never imports pyplot and never calls matplotlib.use().
# Figures are built as explicit Figure objects and saved with Figure.savefig,
# so no global backend is selected. That matters because the review window
# creates a FigureCanvasQTAgg in the same process -- a library function that
# switched the process-wide backend to Agg would take that window down with it.

from __future__ import annotations


import numpy as np
import pandas as pd

# Per-method colours, distinguishable in greyscale as well as colour.
METHOD_COLORS = {
    "threshold": "#c97a2a",
    "deconvolution": "#2c7fb8",
    "template": "#7a3b9a",
    "autoclassifier": "#0f8b8d",
    "miniml": "#2a8f2a",
    "fastmini": "#b5443a",
}
FALLBACK_COLORS = ["#555555", "#8a6dc9", "#d94801", "#08519c", "#2a7f2a"]
INK = "#1b1b1b"
MUTED = "#8a8a8a"
GRID = "#e6e4dc"
FACE = "#faf9f5"

#: (column, axis label, whether the values are signed-negative for inward events)
MEASURES = [
    ("amplitude", "amplitude", True),
    ("rise_time_ms", "rise time (ms)", False),
    ("decay_time_ms", "decay time (ms)", False),
    ("area", "area", True),
    ("inter_event_interval_ms", "inter-event interval (ms)", False),
]


def method_color(method: str, i: int = 0) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])


def _tidy(ax, xlabel: str = "", ylabel: str = "", title: str = ""):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)


def box_with_dots(ax, groups: list, labels: list, colors: list | None = None,
                  ylabel: str = "", title: str = "", max_dots: int = 400,
                  seed: int = 0):
    """Box + whisker with every individual measurement as a jittered dot.

    The replacement for a bar chart everywhere in this package. Outlier
    markers are off because every point is already drawn -- otherwise the
    tails appear twice and look denser than they are.

    Above `max_dots` per group a random subset is drawn (the box, whiskers and
    the reported n always use every point). Thousands of overlapping dots read
    as a solid block, which hides the distribution this plot exists to show.
    """
    rng = np.random.default_rng(seed)
    colors = colors or [method_color(l, i) for i, l in enumerate(labels)]
    clean = [np.asarray(g, dtype=float) for g in groups]
    clean = [g[np.isfinite(g)] for g in clean]
    positions = np.arange(1, len(clean) + 1)

    drawable = [(p, g) for p, g in zip(positions, clean) if g.size]
    if drawable:
        bp = ax.boxplot([g for _, g in drawable],
                        positions=[p for p, _ in drawable],
                        widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=MUTED, lw=1.1),
                        capprops=dict(color=MUTED, lw=1.1),
                        zorder=2)
        for patch, (p, _) in zip(bp["boxes"], drawable):
            patch.set_facecolor(colors[int(p) - 1])
            patch.set_alpha(0.22)
            patch.set_edgecolor(colors[int(p) - 1])

    for pos, g, c in zip(positions, clean, colors):
        if g.size == 0:
            continue
        shown = g if g.size <= max_dots else rng.choice(g, max_dots, replace=False)
        jitter = rng.uniform(-0.16, 0.16, shown.size)
        ax.plot(pos + jitter, shown, "o", ms=2.2, mew=0, color=c, alpha=0.45,
                zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{l}\nn={g.size}" for l, g in zip(labels, clean)], fontsize=8)
    _tidy(ax, ylabel=ylabel, title=title)
    return ax


def _values(per_method: dict, column: str, signed_abs: bool) -> tuple[list, list]:
    """Column values per method, absolute for signed measures.

    Amplitudes and areas are stored signed-negative for inward events. Plotted
    as-is the y axis runs downward and every distribution looks inverted, so
    magnitudes are plotted and the axis label says so.
    """
    labels, groups = [], []
    for m, df in per_method.items():
        if df is None or column not in df.columns:
            continue
        v = df[column].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        labels.append(m)
        groups.append(np.abs(v) if signed_abs else v)
    return labels, groups


def figure_distributions(per_method: dict, unit: str = "pA", seed: int = 0):
    """Histogram + cumulative for each measure, every method overlaid.

    The cumulative panels are the ones to compare distributions on: a
    histogram's shape depends on the bin width, a CDF does not, which matters
    when the claim is "these two distributions differ".
    """
    from matplotlib.figure import Figure

    rows = [(c, lab, sgn) for c, lab, sgn in MEASURES
            if any(df is not None and c in df.columns for df in per_method.values())]
    fig = Figure(figsize=(11, 2.6 * max(len(rows), 1)), facecolor=FACE)
    axes = fig.subplots(len(rows), 2, squeeze=False)

    for r, (col, label, signed) in enumerate(rows):
        labels, groups = _values(per_method, col, signed)
        unit_txt = f" ({unit})" if col in ("amplitude", "area") else ""
        if col == "area":
            unit_txt = f" ({unit}·ms)"
        ax_h, ax_c = axes[r][0], axes[r][1]

        finite = np.concatenate([g for g in groups if g.size]) if groups else np.zeros(0)
        if finite.size:
            hi = float(np.percentile(finite, 99.5))
            bins = np.linspace(0.0, max(hi, 1e-9), 45)
        else:
            bins = 45

        for i, (m, g) in enumerate(zip(labels, groups)):
            if g.size == 0:
                continue
            c = method_color(m, i)
            ax_h.hist(g, bins=bins, histtype="step", lw=1.5, color=c, label=m)
            xs = np.sort(g)
            ax_c.plot(xs, np.arange(1, xs.size + 1) / xs.size, lw=1.6, color=c,
                      label=m)

        _tidy(ax_h, xlabel=("|" + label + "|" if signed else label) + unit_txt,
              ylabel="count", title=f"{label} — distribution")
        _tidy(ax_c, xlabel=("|" + label + "|" if signed else label) + unit_txt,
              ylabel="cumulative fraction", title=f"{label} — cumulative")
        ax_c.set_ylim(0, 1.02)
        if r == 0:
            ax_h.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def figure_summary(per_method: dict, unit: str = "pA", seed: int = 0):
    """Per-method box + whisker + every event as a dot, one panel per measure."""
    from matplotlib.figure import Figure

    measures = [(c, lab, sgn) for c, lab, sgn in MEASURES
                if any(df is not None and c in df.columns
                       for df in per_method.values())]
    n = len(measures)
    cols = min(3, max(n, 1))
    rows = int(np.ceil(n / cols)) if n else 1
    fig = Figure(figsize=(4.4 * cols, 3.8 * rows), facecolor=FACE)
    axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False)).ravel()

    for ax, (col, label, signed) in zip(axes, measures):
        labels, groups = _values(per_method, col, signed)
        unit_txt = f" ({unit})" if col in ("amplitude", "area") else ""
        if col == "area":
            unit_txt = f" ({unit}·ms)"
        box_with_dots(ax, groups, labels, ylabel=("|" + label + "|" if signed
                                                  else label) + unit_txt,
                      title=label, seed=seed)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Per-method distributions — box: median and IQR; "
                 "dots: individual events", x=0.01, ha="left", fontsize=10,
                 color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def figure_waveforms(trace, per_method: dict, pre_ms: float = 5.0,
                     post_ms: float = 25.0):
    """Peak-aligned average event per method, raw and peak-normalised.

    Two panels because they answer different questions. The raw average shows
    whether the methods disagree about event SIZE; the peak-normalised one
    shows whether they disagree about event SHAPE, which is the more telling
    difference -- a method whose average has a slower rise is probably picking
    up something other than what the others are.
    """
    from matplotlib.figure import Figure

    dt = trace.dt
    pre_n = max(1, int(round(pre_ms / 1000.0 / dt)))
    post_n = max(2, int(round(post_ms / 1000.0 / dt)))
    t_ms = (np.arange(-pre_n, post_n) * dt) * 1e3
    v = np.asarray(trace.v, dtype=float)

    fig = Figure(figsize=(11, 4.2), facecolor=FACE)
    ax_raw, ax_norm = fig.subplots(1, 2)

    for i, (m, df) in enumerate(per_method.items()):
        if df is None or df.empty or "peak_idx" not in df.columns:
            continue
        idx = df["peak_idx"].to_numpy(dtype=int)
        segs = []
        for q in idx:
            a, b = q - pre_n, q + post_n
            if a < 0 or b > v.size:
                continue
            seg = v[a:b].astype(float)
            segs.append(seg - np.median(seg[:pre_n]))    # baseline at 0
        if not segs:
            continue
        avg = np.mean(segs, axis=0)
        c = method_color(m, i)
        ax_raw.plot(t_ms, avg, lw=1.6, color=c, label=f"{m} (n={len(segs)})")
        peak = avg[np.argmax(np.abs(avg))]
        if peak != 0:
            ax_norm.plot(t_ms, avg / abs(peak), lw=1.6, color=c, label=m)

    _tidy(ax_raw, xlabel="time from peak (ms)", ylabel=f"current ({trace.unit})",
          title="average event — raw (do the methods agree on size?)")
    _tidy(ax_norm, xlabel="time from peak (ms)", ylabel="normalised",
          title="average event — peak-normalised (…on shape?)")
    ax_raw.legend(fontsize=8, frameon=False)
    ax_raw.axhline(0, color=MUTED, lw=0.8)
    ax_norm.axhline(0, color=MUTED, lw=0.8)
    fig.tight_layout()
    return fig


def figure_agreement(consensus_df: pd.DataFrame, methods: list,
                     agreement: pd.DataFrame, duration_s: float,
                     unit: str = "pA", seed: int = 0):
    """Stability over time, the agreement histogram, and the agreement matrix."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 8), facecolor=FACE)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)
    ax_stab = fig.add_subplot(gs[0, :2])
    ax_rate = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[1, 1])
    ax_amp = fig.add_subplot(gs[1, 2])

    n_m = max(len(methods), 1)

    # -- amplitude against time: is the recording stable? ---------------
    for i, m in enumerate(methods):
        col, fcol = f"amp_{m}", f"found_{m}"
        if col not in consensus_df.columns:
            continue
        sel = consensus_df[consensus_df[fcol] == 1]
        if sel.empty:
            continue
        ax_stab.plot(sel["peak_time_s"], np.abs(sel[col]), "o", ms=2.0, mew=0,
                     alpha=0.35, color=method_color(m, i), label=m)
        # rolling median, so a drift is visible through the scatter
        order = np.argsort(sel["peak_time_s"].to_numpy())
        y = np.abs(sel[col].to_numpy())[order]
        x = sel["peak_time_s"].to_numpy()[order]
        w = max(5, y.size // 25)
        if y.size > w:
            med = pd.Series(y).rolling(w, center=True, min_periods=1).median()
            ax_stab.plot(x, med, lw=1.8, color=method_color(m, i))
    _tidy(ax_stab, xlabel="time (s)", ylabel=f"|amplitude| ({unit})",
          title="stability — amplitude over the recording (line: rolling median)")
    # Build the legend from proxy lines: the scatter markers are 2 px at
    # alpha 0.35, which is right on the plot and invisible in a legend.
    from matplotlib.lines import Line2D
    ax_stab.legend(handles=[Line2D([], [], color=method_color(m, i), lw=2.4,
                                   label=m) for i, m in enumerate(methods)],
                   fontsize=8, frameon=False, ncol=max(1, len(methods) // 2))

    # -- event frequency per method (a count -> stays a bar) ------------
    rates = [(consensus_df.get(f"found_{m}", pd.Series(dtype=int)).sum()
              / max(duration_s, 1e-9)) for m in methods]
    ax_rate.bar(range(len(methods)), rates,
                color=[method_color(m, i) for i, m in enumerate(methods)],
                alpha=0.75)
    ax_rate.set_xticks(range(len(methods)))
    ax_rate.set_xticklabels(methods, fontsize=8, rotation=30, ha="right")
    _tidy(ax_rate, ylabel="events / s", title="event frequency")
    for i, r in enumerate(rates):
        ax_rate.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8,
                     color=INK)

    # -- how many methods found each event (also a count) ---------------
    counts = [int((consensus_df["n_methods"] == k).sum()) for k in range(1, n_m + 1)]
    ax_hist.bar(range(1, n_m + 1), counts, color="#3182bd", alpha=0.75)
    for k, c in zip(range(1, n_m + 1), counts):
        if c:
            ax_hist.text(k, c, str(c), ha="center", va="bottom", fontsize=8,
                         color=INK)
    ax_hist.set_xticks(range(1, n_m + 1))
    _tidy(ax_hist, xlabel="number of methods", ylabel="events",
          title="agreement — events found by n methods")

    # -- pairwise agreement matrix --------------------------------------
    if agreement is not None and not agreement.empty:
        vals = agreement.to_numpy(dtype=float).copy()
        off = vals.copy()
        np.fill_diagonal(off, np.nan)
        im = ax_mat.imshow(off, cmap="Blues", vmin=0, vmax=1)
        ax_mat.set_xticks(range(len(methods)))
        ax_mat.set_yticks(range(len(methods)))
        ax_mat.set_xticklabels(methods, fontsize=8, rotation=30, ha="right")
        ax_mat.set_yticklabels(methods, fontsize=8)
        for a in range(len(methods)):
            for b in range(len(methods)):
                if a == b:
                    ax_mat.text(b, a, f"n={int(vals[a, b])}", ha="center",
                                va="center", fontsize=7.5, color=MUTED)
                else:
                    ax_mat.text(b, a, f"{off[a, b]:.2f}", ha="center",
                                va="center", fontsize=8,
                                color="white" if off[a, b] > 0.55 else INK)
        ax_mat.set_title("pairwise agreement (Jaccard)", loc="left", fontsize=10,
                         color=INK)
        ax_mat.grid(False)
        fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

    # -- amplitude by agreement level: are 1-method events smaller? -----
    groups, labels = [], []
    for k in range(1, n_m + 1):
        sel = consensus_df[consensus_df["n_methods"] == k]
        if sel.empty:
            continue
        groups.append(np.abs(sel["amplitude_median"].to_numpy(dtype=float)))
        labels.append(f"{k} of {n_m}")
    if groups:
        box_with_dots(ax_amp, groups, labels,
                      colors=["#3182bd"] * len(groups),
                      ylabel=f"|amplitude| ({unit})",
                      title="amplitude by agreement level", seed=seed)
    # No tight_layout here: this figure's gridspec already sets its spacing,
    # and tight_layout cannot handle the colorbar axes -- it warns and can
    # lay the panels out incorrectly.
    return fig


def write_all(outcome, per_method: dict | None = None, seed: int = 0) -> list[str]:
    """Write every figure for a finished run. Returns the paths.

    Takes the in-memory RunOutcome, so nothing has to be read back from CSV.
    """
    from . import consensus as consensus_mod
    from .events import events_frame

    methods = outcome.ran
    if not methods:
        return []
    if per_method is None:
        per_method = {m: events_frame(outcome.results[m].events) for m in methods}
    cons = consensus_mod.frame(outcome.consensus, methods,
                               outcome.settings.tolerance_ms)
    agreement = consensus_mod.agreement_matrix(outcome.consensus, methods)
    unit = outcome.trace.unit

    figures = [
        ("distributions", figure_distributions(per_method, unit, seed)),
        ("boxplots", figure_summary(per_method, unit, seed)),
        ("waveforms", figure_waveforms(outcome.trace, per_method)),
        # 'overview', not 'agreement': <stem>_multimethod_agreement.csv is
        # the pairwise matrix, and a .png sharing that stem invites reading
        # one as a rendering of the other.
        ("overview", figure_agreement(cons, methods, agreement,
                                      outcome.trace.duration_s, unit, seed)),
    ]
    written = []
    for name, fig in figures:
        path = f"{outcome.stem}_multimethod_{name}.png"
        fig.savefig(path, dpi=120, facecolor=FACE)
        written.append(path)
    return written
