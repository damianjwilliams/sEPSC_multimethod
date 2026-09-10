"""Every parameter for every method, in one serialisable object.

One `Settings` holds the preprocessing, the shared measurement parameters, the
consensus tolerance, and one params object per detector. The parameter window
edits it, the runner consumes it, and it is written next to the results so a
run is reproducible from its own output.

Interpreter paths and external commands live here too, which is why the
default config file is gitignored: those are properties of a machine, not of
the analysis.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

from . import detectors
from .consensus import DEFAULT_TOLERANCE_MS
from .events import MeasurementParams
from .trace import Preprocessing

#: Written next to the package by the parameter window's Save button. Holds
#: machine-specific paths, so it is gitignored rather than committed.
DEFAULT_CONFIG_NAME = "multimethod.config.json"


def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        DEFAULT_CONFIG_NAME)


@dataclass
class Settings:
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    measurement: MeasurementParams = field(default_factory=MeasurementParams)
    tolerance_ms: float = DEFAULT_TOLERANCE_MS
    """How close two methods' detections must be to count as the same event."""

    enabled: dict = field(default_factory=dict)
    """method -> run it? Missing means "yes" for in-repo detectors and "yes,
    if available" for external ones -- an unconfigured external detector
    reports itself unavailable rather than being silently switched off."""

    detector_params: dict = field(default_factory=dict)
    """method -> that detector's params dataclass instance."""

    def __post_init__(self):
        for name, det in detectors.REGISTRY.items():
            self.detector_params.setdefault(name, det.default_params())
            self.enabled.setdefault(name, True)

    # -- per-detector access -------------------------------------------

    def params_for(self, name: str):
        return self.detector_params[name]

    def set_params_for(self, name: str, params) -> None:
        self.detector_params[name] = params

    def is_enabled(self, name: str) -> bool:
        return bool(self.enabled.get(name, True))

    def active_methods(self) -> list[str]:
        """Registered detectors the user has not switched off, in registry order."""
        return [n for n in detectors.names() if self.is_enabled(n)]

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "preprocessing": self.preprocessing.as_dict(),
            "measurement": self.measurement.as_dict(),
            "tolerance_ms": self.tolerance_ms,
            "enabled": dict(self.enabled),
            "detectors": {name: detectors.params_as_dict(p)
                          for name, p in self.detector_params.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        """Rebuild from a config dict, ignoring anything it does not recognise.

        Deliberately forgiving: a config written by a newer version, or one
        naming a detector that has since been renamed, must not stop a run.
        Unknown keys are dropped, missing keys take defaults.
        """
        d = d or {}
        s = cls(
            preprocessing=Preprocessing.from_dict(d.get("preprocessing")),
            measurement=MeasurementParams.from_dict(d.get("measurement")),
            tolerance_ms=float(d.get("tolerance_ms", DEFAULT_TOLERANCE_MS)),
            enabled={k: bool(v) for k, v in (d.get("enabled") or {}).items()
                     if k in detectors.REGISTRY},
        )
        for name, raw in (d.get("detectors") or {}).items():
            if name not in detectors.REGISTRY:
                continue
            s.detector_params[name] = detectors.params_from_dict(
                detectors.get(name).params_cls, raw)
        return s

    def save(self, path: str | None = None) -> str:
        path = path or default_config_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        """Load from `path`, or return defaults if it does not exist."""
        path = path or default_config_path()
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def describe(self) -> str:
        lines = [f"preprocessing: {self.preprocessing.describe()}",
                 f"consensus tolerance: {self.tolerance_ms:g} ms",
                 f"measurement: {self.measurement.direction}, "
                 f"baseline {self.measurement.baseline_before_ms:g}+"
                 f"{self.measurement.baseline_avg_ms:g} ms, "
                 f"decay to {self.measurement.decay_fraction:g}"]
        for name in detectors.names():
            mark = "on " if self.is_enabled(name) else "off"
            lines.append(f"  [{mark}] {name}")
        return "\n".join(lines)


def field_specs(params) -> list[tuple]:
    """(field_name, type, current_value, help_text) for a params dataclass.

    Drives the parameter window, so every detector's tab is generated from its
    own dataclass rather than hand-written -- a new parameter appears in the
    window as soon as it exists on the dataclass, with its docstring as the
    tooltip.
    """
    if not dataclasses.is_dataclass(params):
        return []
    docs = _field_docs(type(params))
    out = []
    for f in dataclasses.fields(params):
        out.append((f.name, f.type, getattr(params, f.name),
                    docs.get(f.name, "")))
    return out


def _field_docs(cls) -> dict:
    """Field docstrings: the string literal on the line after a field.

    dataclasses do not keep these, so read them off the source. Best-effort:
    a missing tooltip is not worth an exception.
    """
    import ast
    import inspect
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(cls))
        tree = ast.parse(src).body[0]
    except (OSError, TypeError, SyntaxError, IndexError):
        return {}
    docs, pending = {}, None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            pending = node.target.id
        elif (pending and isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)):
            docs[pending] = " ".join(node.value.value.split())
            pending = None
        else:
            pending = None
    return docs
