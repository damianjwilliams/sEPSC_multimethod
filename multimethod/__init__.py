"""Run every synaptic-event detector on one recording, then review which
method found which event.

Start here:
    from multimethod import Settings, run_all
    outcome = run_all("recording.abf", settings=Settings())
    print(outcome.status_report())
    outcome.write()
"""

from .consensus import ConsensusEvent
from .events import Event, MeasurementParams, events_frame, measure_at
from .runner import RunOutcome, run_all, run_one
from .settings import Settings
from .trace import Preprocessing, Trace, load, noise_sd, output_stem

__version__ = "0.1.0"

__all__ = [
    "Settings", "Preprocessing", "MeasurementParams",
    "Trace", "load", "noise_sd", "output_stem",
    "Event", "events_frame", "measure_at",
    "run_all", "run_one", "RunOutcome", "ConsensusEvent",
    "__version__",
]
