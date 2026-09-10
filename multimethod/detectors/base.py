"""What a detector is, and what a run of one produces.

A detector answers exactly one question: WHERE are the events? It returns
sample indices into the trace it was handed. It does not measure them -- see
events.measure_at for why that is shared.

Adding a detector means: subclass Detector, give it a name, a params
dataclass, and a `find` that returns indices; then register it in
detectors/__init__.py. Nothing else in the package needs to change -- the
runner, the parameter window and the review window all read the registry.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..events import Event
from ..trace import Trace


@dataclass
class DetectorResult:
    """The outcome of running one detector, successful or not.

    A detector that could not run is not an exception: with external detectors
    behind a subprocess boundary, "the environment isn't there" is an ordinary
    outcome that must not take the other detectors down with it. So failure is
    data -- `ok=False` plus a `message` saying what to do about it.
    """

    method: str
    ok: bool
    events: list[Event] = field(default_factory=list)
    message: str = ""
    elapsed_s: float = 0.0
    n_located: int = 0
    """How many indices the detector returned, before measurement dropped any."""
    n_unmeasurable: int = 0
    """Located but not measurable (too near the start, no onset crossing, ...)."""
    diagnostics: dict = field(default_factory=dict)
    """Per-detector extras: the estimated kernel, the detection criterion trace,
    a model's own amplitudes. Not written to the events CSV."""

    @property
    def n_events(self) -> int:
        return len(self.events)

    def status_line(self, width: int = 14) -> str:
        name = self.method.ljust(width)
        if not self.ok:
            return f"  {name} SKIP   {self.message}"
        extra = ""
        if self.n_unmeasurable:
            extra = f"   ({self.n_unmeasurable} located but unmeasurable)"
        return (f"  {name} OK   {self.n_events:6d} events  "
                f"{self.elapsed_s:6.1f} s{extra}")


class Detector:
    """Base class. Subclasses set `name`, `label`, `params_cls` and `find`."""

    name: str = ""
    label: str = ""
    #: One line for the parameter window and the README table.
    blurb: str = ""
    params_cls: type = dict
    #: True if this detector runs in another interpreter (see external.py).
    external: bool = False

    def default_params(self):
        return self.params_cls()

    def available(self, params) -> tuple[bool, str]:
        """Can this detector run here? (ok, reason_if_not).

        In-repo detectors are always available. External ones check for their
        interpreter and their input files.
        """
        return True, ""

    def find(self, trace: Trace, params) -> np.ndarray:
        """Sample indices of located events. Must not modify `trace.v`."""
        raise NotImplementedError

    # -- helpers shared by subclasses ------------------------------------

    @staticmethod
    def _polarity(direction: str) -> float:
        return -1.0 if direction == "negative" else 1.0

    @staticmethod
    def _upright(trace: Trace, direction: str, subtract_drift: bool,
                 drift_win_ms: float = 300.0) -> np.ndarray:
        """The trace flipped so events point UP and slow drift removed.

        Every in-repo detector works on this, so 'direction' and drift are
        handled once rather than three times with three subtly different
        conventions.
        """
        from ..trace import track_baseline

        pol = Detector._polarity(direction)
        v = np.asarray(trace.v, dtype=float)
        if subtract_drift:
            pct = 80.0 if direction == "negative" else 20.0
            v = v - track_baseline(v, trace.dt, drift_win_ms, pct)
        return pol * v


def params_as_dict(params: Any) -> dict:
    """Serialise a params dataclass (or pass a plain dict through)."""
    if dataclasses.is_dataclass(params):
        return dataclasses.asdict(params)
    return dict(params or {})


def params_from_dict(params_cls: type, d: dict):
    """Rebuild a params dataclass, ignoring keys it does not define.

    Tolerant on purpose: a config file written by a newer version of this
    package, or one carrying a run's diagnostics alongside its parameters,
    must not crash an older reader.
    """
    if not dataclasses.is_dataclass(params_cls):
        return dict(d or {})
    known = {f.name for f in dataclasses.fields(params_cls)}
    return params_cls(**{k: v for k, v in (d or {}).items() if k in known})
