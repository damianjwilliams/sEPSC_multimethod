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

#: Written next to the package by the parameter window's Save button, or by
#: `calibrate --write`. Holds machine-specific paths, so it is gitignored
#: rather than committed.
#:
#: It is picked up automatically by every command, which is the point of it and
#: also its one hazard: a file written weeks ago silently changes what a fresh
#: run does. So nothing in this package loads it without also reporting it --
#: see `ConfigSource` and `load_with_source`.
DEFAULT_CONFIG_NAME = "multimethod.config.json"


def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        DEFAULT_CONFIG_NAME)


def _flatten(d: dict, prefix: str = "") -> dict:
    """`{"a": {"b": 1}}` -> `{"a.b": 1}`, so two configs can be diffed field by
    field rather than section by section."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


@dataclass(frozen=True)
class ConfigSource:
    """Where a run's parameters came from, and how they differ from the defaults.

    This exists because of a real incident: a `multimethod.config.json` written
    by `calibrate --write` was in effect for later runs, and those runs gave no
    sign of it. The parameters were recorded in each run's sidecar, so nothing
    was lost -- but "recoverable afterwards" is not the same as "visible at the
    time", and a 4-sigma amplitude floor arriving from an old file changed the
    event count by 11% with nothing on screen to say so.

    Every command that loads settings reports one of these before it does any
    work. A config file is a feature; a silent one is not.
    """

    path: str
    """The file that was consulted, whether or not it exists."""

    exists: bool
    """Whether it was there. If not, the built-in defaults are in effect."""

    explicit: bool
    """True if the user named it with `--config`. An explicit path needs no
    warning -- they just asked for it. The default path is the hazardous one,
    because it applies to commands that never mention it."""

    overrides: tuple = ()
    """`(dotted_field, builtin_value, value_in_effect)` for every field that
    differs from a stock `Settings()`. Empty when the file matches the defaults,
    which is worth saying too: it means the file is inert."""

    ignored: bool = False
    """True when a config file was present but deliberately not read, because
    the user asked for the built-in defaults. Reported rather than silent: a
    file sitting there having no effect is exactly as confusing as one having
    an unannounced effect."""

    @property
    def in_effect(self) -> bool:
        return self.exists and bool(self.overrides)

    def describe(self) -> str:
        """One block of text, safe to print before every command."""
        name = os.path.basename(self.path)
        if self.ignored:
            return (f"Parameters: built-in defaults (--defaults given; "
                    f"{name} is present but not being read)."
                    if self.exists else
                    f"Parameters: built-in defaults (--defaults given; "
                    f"there is no {name} anyway).")
        if not self.exists:
            return f"Parameters: built-in defaults (no {name})."
        where = "--config" if self.explicit else "found beside the package"
        if not self.overrides:
            return (f"Parameters: {name} ({where}) is in effect, but every value "
                    f"in it matches the built-in defaults.")
        width = max(len(k) for k, _, _ in self.overrides)
        lines = [f"Parameters: {name} ({where}) overrides "
                 f"{len(self.overrides)} built-in default(s):"]
        for key, was, now in self.overrides:
            lines.append(f"    {key:<{width}}  {was!r} -> {now!r}")
        if not self.explicit:
            lines.append("  This file is picked up automatically. To run on "
                         "the built-in defaults, rename it or pass --config")
            lines.append("  with a different path.")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """For the run sidecar, so the provenance is recorded and not just shown."""
        return {"path": os.path.abspath(self.path) if self.exists else self.path,
                "exists": self.exists,
                "explicit": self.explicit,
                "ignored": self.ignored,
                "overrides": {k: {"builtin": was, "in_effect": now}
                              for k, was, now in self.overrides}}


def describe_source(settings: "Settings", path: str, exists: bool,
                    explicit: bool) -> ConfigSource:
    """Diff `settings` against a stock `Settings()` to build a ConfigSource."""
    mine = _flatten(settings.to_dict())
    stock = _flatten(Settings().to_dict())
    keys = sorted(set(mine) | set(stock))
    overrides = tuple((k, stock.get(k), mine.get(k)) for k in keys
                      if mine.get(k) != stock.get(k))
    return ConfigSource(path=path, exists=exists, explicit=explicit,
                        overrides=overrides)


def load_with_source(path: str | None = None,
                     ignore: bool = False) -> tuple["Settings", ConfigSource]:
    """`Settings.load`, plus what it loaded and how that differs from default.

    Prefer this over `Settings.load` anywhere a human is watching: it is the
    only way a command can tell the user that a config file is in effect.

    `ignore=True` returns the built-in defaults without reading the file, and
    says so. That is what `--defaults` is for: the parameter window pre-fills
    from whatever this returns, so without it there is no way to open the
    window on the values the package actually ships with.
    """
    explicit = path is not None
    resolved = path or default_config_path()
    exists = os.path.exists(resolved)
    if ignore:
        return Settings(), ConfigSource(path=resolved, exists=exists,
                                        explicit=explicit, ignored=True)
    settings = Settings.load(resolved)
    return settings, describe_source(settings, resolved, exists, explicit)


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
