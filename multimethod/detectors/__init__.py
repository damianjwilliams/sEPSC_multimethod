"""The detector registry.

Everything else in the package reads this: the runner iterates it, the
parameter window builds a tab per entry, the review window colours events by
it, and the config file keys off the names. Adding a detector here is the only
wiring step.

Order matters and is deliberate -- in-repo detectors first, cheapest first,
external ones last -- because the runner reports progress in this order and a
40-second miniML subprocess should not hold up the first useful output.
"""

from __future__ import annotations

from .base import Detector, DetectorResult, params_as_dict, params_from_dict
from .deconvolution import DeconvolutionDetector, DeconvolutionParams
from .external import CommandDetector, CommandParams, MiniMLDetector, MiniMLParams
from .template import TemplateDetector, TemplateParams
from .threshold import ThresholdDetector, ThresholdParams

#: name -> Detector instance. Instances, not classes: they are stateless apart
#: from `last_diagnostics`, which the runner reads immediately after `find`.
REGISTRY: dict[str, Detector] = {}


def register(detector: Detector) -> Detector:
    if detector.name in REGISTRY:
        raise ValueError(f"detector {detector.name!r} is already registered")
    REGISTRY[detector.name] = detector
    return detector


for _d in (ThresholdDetector(), DeconvolutionDetector(), TemplateDetector(),
           MiniMLDetector(), CommandDetector()):
    register(_d)


def get(name: str) -> Detector:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; known: {sorted(REGISTRY)}") from None


def names() -> list[str]:
    return list(REGISTRY)


def in_repo_names() -> list[str]:
    """Detectors that need nothing but this package's own dependencies."""
    return [n for n, d in REGISTRY.items() if not d.external]


__all__ = [
    "REGISTRY", "register", "get", "names", "in_repo_names",
    "Detector", "DetectorResult", "params_as_dict", "params_from_dict",
    "ThresholdDetector", "ThresholdParams",
    "DeconvolutionDetector", "DeconvolutionParams",
    "TemplateDetector", "TemplateParams",
    "MiniMLDetector", "MiniMLParams",
    "CommandDetector", "CommandParams",
]
