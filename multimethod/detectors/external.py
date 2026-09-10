"""Detectors that run in another interpreter.

Why this exists
---------------
miniML needs TensorFlow 2.14-2.15, which requires numpy<2. A current analysis
environment has numpy 2.x. Those cannot coexist in one process, so miniML
cannot be imported alongside the rest of this package -- not as a packaging
inconvenience, but as a hard constraint. The only workable answer is a
subprocess boundary.

The contract across that boundary is deliberately minimal: the child is given
the .abf path plus the preprocessing settings, and returns EVENT TIMES IN
SECONDS as JSON. Not sample indices (the child may resample differently), not
amplitudes (this package measures every method identically -- see events.py).
Times are the one quantity that survives the crossing unambiguously.

Two adapters are built on it:

  MiniMLDetector   runs a worker script this package ships, in the interpreter
                   where miniML is installed.
  CommandDetector  runs any command you configure and reads a time column out
                   of a CSV it leaves behind. `fastmini` is wired this way,
                   because it needs a third-party checkout whose internals are
                   not ours to depend on.

Both report "not available" rather than raising when their environment is
missing, so the in-repo detectors still run. That is what makes the consensus
table honest about what did and did not contribute.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..trace import Trace
from .base import Detector

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_miniml_worker.py")


def _tail(text: str | None, n: int) -> str:
    """The last n non-blank lines of a stream, for a failure report."""
    lines = [L for L in (text or "").splitlines() if L.strip()]
    return "\n".join(lines[-n:])


def _last_meaningful_line(*streams) -> str:
    """The most informative single line from a dead child's output.

    Prefers a line that looks like an exception over the last line, because
    TensorFlow prints pages of INFO after the real error and the final line is
    usually noise.
    """
    lines = []
    for stream in streams:
        lines += [L.strip() for L in (stream or "").splitlines() if L.strip()]
    for line in reversed(lines):
        if "Error" in line or "error:" in line.lower() or "Exception" in line:
            return line
    return lines[-1] if lines else "(the worker printed nothing)"


def _times_to_indices(times_s, trace: Trace) -> np.ndarray:
    """Event times in seconds -> sample indices in OUR trace.

    Times outside the trace are dropped rather than clipped: clipping would
    pile several events onto sample 0 and invent agreement with other methods
    that never happened.
    """
    t = np.asarray(times_s, dtype=float)
    t = t[np.isfinite(t)]
    idx = np.round(t / trace.dt).astype(int)
    return np.unique(idx[(idx >= 0) & (idx < trace.v.size)])


@dataclass(frozen=True)
class MiniMLParams:
    python_exe: str = ""
    """Interpreter where miniML is installed. Empty = try the usual suspects
    (see resolve_python). Set it in the parameter window or the config file."""

    model_path: str = "models/GC_lstm_model.h5"
    """Relative to miniML's own resources, or an absolute path."""

    threshold: float = 0.5
    """miniML's own sigmoid cutoff."""

    window_ms: float = 12.0
    """Detection window in REAL TIME, not samples.

    miniML's shipped models were trained with a 600-sample window at 50 kHz,
    i.e. 12 ms. window_size is in raw samples, so passing 600 at another
    sampling rate silently changes the real-time span -- 6 ms at 100 kHz, which
    starves the model of most of the event. The worker converts this to samples
    for the recording's actual rate.
    """

    batch_size: int = 512
    """Windows per inference batch. Larger is faster and uses more
    memory; it does not change the result."""
    direction: str = "negative"
    """Event polarity passed to miniML: "negative" for inward currents."""
    timeout_s: float = 900.0
    """Give up on the subprocess after this long, s. miniML on CPU takes
    roughly 30 s per minute of recording, so allow generously."""
    extra_args: str = ""
    """Passed through to the worker verbatim, for miniML flags not exposed here."""


CANDIDATE_PYTHONS = [
    r"C:\Users\damia\anaconda3\envs\clampex_miniml\python.exe",
    os.path.expanduser("~/anaconda3/envs/clampex_miniml/python.exe"),
    os.path.expanduser("~/miniconda3/envs/clampex_miniml/python.exe"),
    os.path.expanduser("~/anaconda3/envs/clampex_miniml/bin/python"),
]


def resolve_python(configured: str) -> str | None:
    """Find an interpreter that has miniML. Configured value wins."""
    if configured:
        return configured if os.path.isfile(configured) else None
    for c in CANDIDATE_PYTHONS:
        if os.path.isfile(c):
            return c
    return None


class MiniMLDetector(Detector):
    """miniML (O'Neill et al., eLife 2025) -- a CNN-LSTM event classifier."""

    name = "miniml"
    label = "miniML (CNN-LSTM)"
    blurb = ("Deep-learning classifier slid over the trace. Runs in its own "
             "interpreter: TensorFlow needs numpy<2.")
    params_cls = MiniMLParams
    external = True

    def available(self, params: MiniMLParams) -> tuple[bool, str]:
        exe = resolve_python(params.python_exe)
        if exe is None:
            where = params.python_exe or "any known conda env"
            return False, (f"no interpreter with miniML found ({where}). "
                           f"Set it in Parameters > miniml > python_exe.")
        if not os.path.isfile(WORKER):
            return False, f"worker script missing: {WORKER}"
        return True, ""

    def find(self, trace: Trace, params: MiniMLParams) -> np.ndarray:
        exe = resolve_python(params.python_exe)
        pre = trace.preprocessing
        request = {
            "abf_path": os.path.abspath(trace.abf_path),
            "channel": trace.channel,
            "preprocessing": pre.as_dict(),
            "model_path": params.model_path,
            "threshold": params.threshold,
            "window_ms": params.window_ms,
            "batch_size": params.batch_size,
            "direction": params.direction,
            "extra_args": params.extra_args,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "result.json")
            request["out_path"] = out_path
            req_path = os.path.join(tmp, "request.json")
            with open(req_path, "w", encoding="utf-8") as fh:
                json.dump(request, fh)

            proc = subprocess.run([exe, WORKER, req_path],
                                  capture_output=True, text=True,
                                  timeout=params.timeout_s)
            if not os.path.exists(out_path):
                # The child died before it could report anything -- a missing
                # dependency, an import error, or a hard crash. Everything
                # useful is in its stderr, so keep it rather than reducing the
                # whole failure to one line.
                self.last_diagnostics = {
                    "python_exe": exe,
                    "returncode": proc.returncode,
                    "stderr_tail": _tail(proc.stderr, 40),
                    "stdout_tail": _tail(proc.stdout, 10),
                }
                raise RuntimeError(
                    f"the miniML worker produced no result (exit "
                    f"{proc.returncode}). Interpreter: {exe}\n"
                    f"{_last_meaningful_line(proc.stderr, proc.stdout)}\n"
                    f"The worker's full stderr is in the run sidecar under "
                    f"methods.miniml.diagnostics.stderr_tail.")
            with open(out_path, encoding="utf-8") as fh:
                result = json.load(fh)

        if not result.get("ok"):
            # The worker caught its own exception and reported it, traceback
            # and all. Keep the traceback: without it the message is often
            # just a type name, and the cause is three frames down inside
            # miniML or TensorFlow.
            self.last_diagnostics = {
                "python_exe": exe,
                "error": result.get("error", ""),
                "traceback": result.get("traceback", ""),
                "stderr_tail": _tail(proc.stderr, 40),
            }
            raise RuntimeError(
                f"{result.get('error', 'the miniML worker failed')}\n"
                f"Interpreter: {exe}\n"
                f"Full traceback is in the run sidecar under "
                f"methods.miniml.diagnostics.traceback.")
        self.last_diagnostics = {k: v for k, v in result.items()
                                 if k not in ("event_times_s",)}
        return _times_to_indices(result.get("event_times_s", []), trace)


@dataclass(frozen=True)
class CommandParams:
    """A detector you configure rather than one this package implements.

    `command` is run with {abf}, {channel}, {stem} substituted, then
    `csv_glob` is searched for its output and `time_column` read out of it.
    Times may be in seconds (`time_in_samples=False`) or in samples.
    """

    command: str = ""
    """Command to run. {abf}, {channel} and {stem} are substituted.
    Example:  C:/env/python.exe -m sepsc fastmini {abf}"""
    csv_glob: str = "{stem}_fastmini_events.csv"
    """Where to find the CSV the command leaves behind. Same
    substitutions; the newest match is read if several exist."""
    time_column: str = "location_s"
    """Column in that CSV holding each event position. Times are the only
    thing read from it -- everything else is measured here, so this
    detector is comparable with the rest."""
    time_in_samples: bool = False
    """Tick if that column is in SAMPLES rather than seconds. Getting this
    wrong scales every event position by the sample rate."""
    cwd: str = ""
    """Working directory for the command. Leave blank to inherit."""
    timeout_s: float = 1800.0
    """Give up on the command after this long, s."""
    enabled: bool = False
    """Off by default: an unconfigured command detector should not appear to
    fail on every run. Turn it on once `command` points somewhere real."""


class CommandDetector(Detector):
    """Runs an external command and reads event times out of a CSV.

    `fastmini` is the worked example: it needs the Mini_Scripts checkout, whose
    API is a third party's to change, so this package shells out to a command
    you control and reads only its documented output columns.
    """

    name = "fastmini"
    label = "fastmini (per-recording MLP)"
    blurb = ("Any external detector, run as a command. Pre-wired for fastmini; "
             "reads a time column out of the CSV it writes.")
    params_cls = CommandParams
    external = True

    def available(self, params: CommandParams) -> tuple[bool, str]:
        if not params.enabled:
            return False, "disabled (set enabled=true and a command to use it)"
        if not params.command.strip():
            return False, "no command configured (Parameters > fastmini > command)"
        return True, ""

    def find(self, trace: Trace, params: CommandParams) -> np.ndarray:
        import pandas as pd

        from ..trace import output_stem

        stem = output_stem(trace.abf_path, trace.channel, trace.preprocessing)
        fields = {"abf": os.path.abspath(trace.abf_path),
                  "channel": str(trace.channel), "stem": stem}
        cmd = params.command.format(**fields)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              cwd=params.cwd or None, timeout=params.timeout_s)
        pattern = params.csv_glob.format(**fields)
        matches = sorted(glob.glob(pattern))
        if not matches:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            hint = tail[-1] if tail else f"exit code {proc.returncode}"
            raise RuntimeError(f"no CSV matched {pattern!r} after running the "
                               f"command ({hint})")
        df = pd.read_csv(matches[-1])
        if params.time_column not in df.columns:
            raise RuntimeError(f"{os.path.basename(matches[-1])} has no column "
                               f"{params.time_column!r} (has: {list(df.columns)})")
        col = df[params.time_column].to_numpy(dtype=float)
        times = col * trace.dt if params.time_in_samples else col
        self.last_diagnostics = {"csv": matches[-1], "n_rows": int(len(df))}
        return _times_to_indices(times, trace)
