"""Run every enabled detector on one recording and write the results.

Order of operations, and why:

  1. Load and preprocess ONCE. Every detector sees the same array, so their
     event times are comparable.
  2. Run each detector to get peak indices. A detector that cannot run, or
     that raises, becomes a failed DetectorResult -- it does not take the run
     down. External detectors depend on other interpreters being present, so
     partial success is the normal case, not an error case.
  3. Measure every located event with ONE shared measurement routine, so the
     amplitude columns mean the same thing across methods.
  4. Cluster into consensus events and write the tables.

Output, all next to the .abf and all keyed off the same stem:

    <stem>_<method>_events.csv        per method, one row per event
    <stem>_multimethod_events.csv     the consensus table (found_*/amp_* columns)
    <stem>_multimethod_summary.csv    per method: n, unique-to-it, median amp
    <stem>_multimethod_agreement.csv  pairwise agreement matrix
    <stem>_multimethod_run.json       settings + per-method status, for reproducing
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import pandas as pd

from . import consensus as consensus_mod
from . import detectors
from .detectors import DetectorResult
from .events import events_frame, measure_all
from .settings import Settings
from .trace import Trace, load, noise_sd, output_stem

EVENTS_SUFFIX = "_events.csv"
CONSENSUS_SUFFIX = "_multimethod_events.csv"
SUMMARY_SUFFIX = "_multimethod_summary.csv"
AGREEMENT_SUFFIX = "_multimethod_agreement.csv"
RUN_SUFFIX = "_multimethod_run.json"
REVIEWED_SUFFIX = "_multimethod_reviewed.csv"
PROGRESS_SUFFIX = "_multimethod_review_progress.csv"


@dataclass
class RunOutcome:
    """Everything one run produced, in memory. `write` puts it on disk."""

    trace: Trace
    settings: Settings
    results: dict = field(default_factory=dict)
    consensus: list = field(default_factory=list)
    noise_sd: float = 0.0
    stem: str = ""
    warnings: list = field(default_factory=list)

    @property
    def ran(self) -> list:
        """Methods that actually produced events (in registry order)."""
        return [m for m in self.results if self.results[m].ok]

    @property
    def per_method_events(self) -> dict:
        return {m: r.events for m, r in self.results.items() if r.ok}

    def status_report(self) -> str:
        lines = [f"Running {len(self.results)} detector(s) on "
                 f"{os.path.basename(self.trace.abf_path)}", ""]
        lines += [r.status_line() for r in self.results.values()]
        ok = len(self.ran)
        lines.append("")
        lines.append(f"{ok} of {len(self.results)} method(s) ran"
                     f" -> {len(self.consensus)} consensus events")
        for w in self.warnings:
            lines.append(f"NOTE: {w}")
        return "\n".join(lines)

    def write(self) -> list[str]:
        written = []
        for method, res in self.results.items():
            if not res.ok:
                continue
            path = f"{self.stem}_{method}{EVENTS_SUFFIX}"
            events_frame(res.events).to_csv(path, index=False)
            written.append(path)

        methods = self.ran
        cons_path = self.stem + CONSENSUS_SUFFIX
        consensus_mod.frame(self.consensus, methods,
                            self.settings.tolerance_ms).to_csv(cons_path, index=False)
        written.append(cons_path)

        summary_path = self.stem + SUMMARY_SUFFIX
        consensus_mod.summarise(self.consensus, methods).to_csv(summary_path, index=False)
        written.append(summary_path)

        agree_path = self.stem + AGREEMENT_SUFFIX
        consensus_mod.agreement_matrix(self.consensus, methods).to_csv(agree_path)
        written.append(agree_path)

        run_path = self.stem + RUN_SUFFIX
        with open(run_path, "w", encoding="utf-8") as fh:
            json.dump({
                "abf_path": os.path.abspath(self.trace.abf_path),
                "channel": self.trace.channel,
                "duration_s": self.trace.duration_s,
                "sample_rate_hz": self.trace.fs,
                "unit": self.trace.unit,
                "hardware_filter_hz": self.trace.hardware_filter_hz,
                "noise_sd": self.noise_sd,
                "detection_limit_4sd": 4 * self.noise_sd,
                "n_consensus_events": len(self.consensus),
                "settings": self.settings.to_dict(),
                "methods": {m: {
                    "ok": r.ok, "message": r.message, "n_events": r.n_events,
                    "n_located": r.n_located,
                    "n_unmeasurable": r.n_unmeasurable,
                    "elapsed_s": round(r.elapsed_s, 3),
                    "diagnostics": _jsonable(r.diagnostics),
                } for m, r in self.results.items()},
                "warnings": self.warnings,
            }, fh, indent=2)
        written.append(run_path)
        return written


def _jsonable(d: dict) -> dict:
    """Drop the array-valued diagnostics; keep the scalars.

    The deconvolved trace and the estimated kernel are megabytes and belong in
    the review window, not in a sidecar meant to be read by a human.
    """
    import numpy as np

    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, np.ndarray):
            if v.size <= 8:
                out[k] = v.tolist()
            else:
                out[k] = f"<array, {v.size} values>"
        elif isinstance(v, str):
            # Tracebacks and stderr tails are multi-line strings and are the
            # whole point of the failure sidecar, so they are kept in full.
            out[k] = v
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
    return out


def run_one(trace: Trace, method: str, settings: Settings) -> DetectorResult:
    """Run one detector, turning any failure into a reported result."""
    det = detectors.get(method)
    params = settings.params_for(method)

    ok, why = det.available(params)
    if not ok:
        return DetectorResult(method=method, ok=False, message=why)

    det.last_diagnostics = {}
    started = time.perf_counter()
    try:
        idxs = det.find(trace, params)
    except Exception as exc:                     # noqa: BLE001 - reported, not raised
        # Keep the diagnostics on the FAILURE path too. A detector that fails
        # is precisely when its diagnostics matter -- the external ones stash
        # the worker's traceback and stderr there -- and dropping them left
        # the sidecar with a one-line message and no way to find the cause.
        return DetectorResult(method=method, ok=False,
                              message=f"{type(exc).__name__}: {exc}",
                              elapsed_s=time.perf_counter() - started,
                              diagnostics=dict(
                                  getattr(det, "last_diagnostics", {}) or {}))
    events, unmeasurable = measure_all(trace, idxs, settings.measurement, method)
    return DetectorResult(
        method=method, ok=True, events=events,
        elapsed_s=time.perf_counter() - started,
        n_located=int(len(idxs)), n_unmeasurable=unmeasurable,
        diagnostics=dict(getattr(det, "last_diagnostics", {}) or {}),
    )


def run_all(abf_path: str, channel: int = 0, settings: Settings | None = None,
            methods: list | None = None, progress=None) -> RunOutcome:
    """Run every enabled detector and build the consensus.

    `progress` is called with each DetectorResult as it completes, so a caller
    can print or update a window without waiting for the slowest method.
    """
    s = settings or Settings()
    trace = load(abf_path, channel, s.preprocessing)
    want = list(methods) if methods else s.active_methods()

    outcome = RunOutcome(trace=trace, settings=s,
                         noise_sd=noise_sd(trace, s.measurement.direction),
                         stem=output_stem(abf_path, channel, s.preprocessing))

    for method in want:
        res = run_one(trace, method, s)
        outcome.results[method] = res
        if progress is not None:
            progress(res)
        _collect_warnings(outcome, res)

    outcome.consensus = consensus_mod.build(
        outcome.per_method_events, trace.dt, s.tolerance_ms, outcome.ran)
    if not outcome.ran:
        outcome.warnings.append(
            "no detector ran -- the consensus table is empty. Check the SKIP "
            "reasons above; external detectors need their own interpreter.")
    return outcome


def _collect_warnings(outcome: RunOutcome, res: DetectorResult) -> None:
    """Surface the things worth saying out loud after a detector runs."""
    d = res.diagnostics or {}
    if res.ok and res.method == "deconvolution":
        kernel_ms = d.get("kernel_ms")
        iso = outcome.settings.params_for("deconvolution").kernel_isolation_ms
        if d.get("kernel_source") == "analytic-fallback":
            outcome.warnings.append(
                "deconvolution: too few isolated events to estimate a kernel, "
                "so the analytic fallback was used. Its tau may not match your "
                "events; check the review window.")
        elif kernel_ms and kernel_ms > 0.5 * iso:
            outcome.warnings.append(
                f"deconvolution: kernel ({kernel_ms:.0f} ms) is long relative "
                f"to the isolation window ({iso:.0f} ms), so the events "
                f"averaged into it were not really isolated. Try "
                f"kernel_isolation_ms {max(4 * kernel_ms, 2 * iso):.0f}.")
    if res.ok and res.n_unmeasurable > max(5, 0.05 * max(res.n_located, 1)):
        outcome.warnings.append(
            f"{res.method}: {res.n_unmeasurable} of {res.n_located} located "
            f"events could not be measured (no onset/decay crossing in the "
            f"search windows). Widen measurement.onset_search_ms / "
            f"decay_search_ms if that seems high.")


def load_consensus(stem: str) -> pd.DataFrame:
    """Read back a consensus table written by `write`."""
    return pd.read_csv(stem + CONSENSUS_SUFFIX, float_precision="round_trip")


def methods_in_consensus(df: pd.DataFrame) -> list[str]:
    """Which methods a consensus table has columns for, in column order."""
    return [c[len("found_"):] for c in df.columns if c.startswith("found_")]
