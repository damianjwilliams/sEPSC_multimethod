"""Matching each method's detections into shared events.

The question this answers is "did these two methods find the SAME event?", and
the honest answer is that it depends on a tolerance. Two detectors can put the
peak of one event a sample or two apart -- different smoothing, different
snapping -- so an exact index match would report near-total disagreement even
where the methods agree perfectly.

So: detections within `tolerance_ms` of each other are treated as one event.
The tolerance is an explicit, reported parameter, not a hidden constant,
because the consensus counts move with it and anyone reading the output needs
to know what it was.

Algorithm
---------
Single-linkage clustering along the time axis would chain: with a 3 ms
tolerance, detections at 0, 2.5, 5, 7.5 ms would collapse into one "event"
spanning 7.5 ms. Instead each detection joins a cluster only if it is within
tolerance of that cluster's MEAN time, which bounds a cluster's width.

One further rule: a cluster holds at most one detection per method. If a
method fires twice inside the tolerance, the nearer one joins and the other
starts a new cluster. Without that rule a method that fragments an event
(reporting it three times) would look like three methods agreeing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_TOLERANCE_MS = 3.0


@dataclass
class ConsensusEvent:
    """One event, and which methods found it."""

    event_id: int
    peak_time_s: float
    """Mean of the contributing detections' times."""
    peak_idx: int
    methods: list[str] = field(default_factory=list)
    per_method_time_s: dict = field(default_factory=dict)
    per_method_amplitude: dict = field(default_factory=dict)

    @property
    def n_methods(self) -> int:
        return len(self.methods)


def build(per_method_events: dict, dt: float,
          tolerance_ms: float = DEFAULT_TOLERANCE_MS,
          method_order: list | None = None) -> list[ConsensusEvent]:
    """Cluster every method's events into consensus events.

    `per_method_events` maps method name -> list[Event]. Methods with no
    events still count as having run: they contribute a column of zeros, which
    is meaningful (it says the method found nothing, not that it was absent).
    """
    tol = tolerance_ms / 1000.0
    order = list(method_order) if method_order else list(per_method_events)

    detections = []
    for method in order:
        for ev in per_method_events.get(method, []):
            detections.append((float(ev.peak_time_s), method, float(ev.amplitude)))
    if not detections:
        return []
    detections.sort(key=lambda d: d[0])

    clusters: list[dict] = []
    for t, method, amp in detections:
        target = None
        for c in reversed(clusters):
            if t - c["mean"] > tol:
                break                        # sorted by time: no earlier cluster can match
            if abs(t - c["mean"]) <= tol and method not in c["by_method"]:
                target = c
                break
        if target is None:
            clusters.append({"times": [t], "mean": t,
                             "by_method": {method: (t, amp)}})
        else:
            target["by_method"][method] = (t, amp)
            target["times"].append(t)
            target["mean"] = float(np.mean(target["times"]))

    out = []
    for i, c in enumerate(sorted(clusters, key=lambda c: c["mean"])):
        found = [m for m in order if m in c["by_method"]]
        out.append(ConsensusEvent(
            event_id=i,
            peak_time_s=c["mean"],
            peak_idx=int(round(c["mean"] / dt)),
            methods=found,
            per_method_time_s={m: c["by_method"][m][0] for m in found},
            per_method_amplitude={m: c["by_method"][m][1] for m in found},
        ))
    return out


def frame(consensus: list, methods: list, tolerance_ms: float = DEFAULT_TOLERANCE_MS
          ) -> pd.DataFrame:
    """Consensus events -> one row per event, one column per method.

    Columns:
        event_id, peak_time_s, peak_idx, n_methods, methods
        found_<method>      1/0, did that method detect this event
        amp_<method>        that method's amplitude for it, NaN if not found
        amplitude_median    median amplitude across the methods that found it
        amplitude_spread    max - min across those methods
    """
    methods = list(methods)
    cols = ["event_id", "peak_time_s", "peak_idx", "n_methods", "methods"]
    if not consensus:
        empty = {c: pd.Series(dtype="float64") for c in cols}
        for m in methods:
            empty[f"found_{m}"] = pd.Series(dtype="int64")
            empty[f"amp_{m}"] = pd.Series(dtype="float64")
        empty["amplitude_median"] = pd.Series(dtype="float64")
        empty["amplitude_spread"] = pd.Series(dtype="float64")
        return pd.DataFrame(empty)

    rows = []
    for ce in consensus:
        row = {
            "event_id": ce.event_id,
            "peak_time_s": ce.peak_time_s,
            "peak_idx": ce.peak_idx,
            "n_methods": ce.n_methods,
            "methods": ";".join(ce.methods),
        }
        amps = []
        for m in methods:
            hit = m in ce.per_method_amplitude
            row[f"found_{m}"] = int(hit)
            row[f"amp_{m}"] = ce.per_method_amplitude.get(m, np.nan)
            if hit:
                amps.append(ce.per_method_amplitude[m])
        row["amplitude_median"] = float(np.median(amps)) if amps else np.nan
        row["amplitude_spread"] = (float(max(amps) - min(amps))
                                   if len(amps) > 1 else 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["peak_idx"] = df["peak_idx"].astype("int64")
    df["n_methods"] = df["n_methods"].astype("int64")
    for m in methods:
        df[f"found_{m}"] = df[f"found_{m}"].astype("int64")
    ordered = cols + [f"{p}_{m}" for m in methods for p in ("found", "amp")]
    return df[ordered + ["amplitude_median", "amplitude_spread"]]


def agreement_matrix(consensus: list, methods: list) -> pd.DataFrame:
    """Pairwise Jaccard agreement: |both| / |either|, per method pair.

    The diagonal is each method's own event count, which is the context you
    need to read the off-diagonal: 0.5 agreement means something different
    when one method found 800 events and the other 2400.
    """
    methods = list(methods)
    sets = {m: {ce.event_id for ce in consensus if m in ce.per_method_amplitude}
            for m in methods}
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            if a == b:
                out.loc[a, b] = float(len(sets[a]))
            else:
                union = len(sets[a] | sets[b])
                out.loc[a, b] = (len(sets[a] & sets[b]) / union) if union else np.nan
    return out


def summarise(consensus: list, methods: list) -> pd.DataFrame:
    """Per-method summary: how many events, how many unique to that method.

    "Unique" is the column worth looking at. A method finding many events no
    other method sees is either more sensitive or generating false positives,
    and the trace is the only way to tell which -- which is what the review
    window is for.
    """
    methods = list(methods)
    rows = []
    total = len(consensus)
    for m in methods:
        found = [ce for ce in consensus if m in ce.per_method_amplitude]
        unique = [ce for ce in found if ce.n_methods == 1]
        rows.append({
            "method": m,
            "n_events": len(found),
            "unique_to_it": len(unique),
            "frac_of_consensus": (len(found) / total) if total else 0.0,
            "median_amplitude": (float(np.median([ce.per_method_amplitude[m]
                                                  for ce in found])) if found
                                 else np.nan),
        })
    return pd.DataFrame(rows)


def by_agreement(consensus: list, n_methods: int) -> pd.DataFrame:
    """How many events were found by exactly 1, 2, ... n methods."""
    counts = np.zeros(n_methods + 1, dtype=int)
    for ce in consensus:
        counts[min(ce.n_methods, n_methods)] += 1
    total = max(int(counts.sum()), 1)
    return pd.DataFrame({
        "n_methods": np.arange(n_methods + 1),
        "n_events": counts,
        "fraction": counts / total,
    }).iloc[1:].reset_index(drop=True)
