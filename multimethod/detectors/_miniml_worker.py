#!/usr/bin/env python
"""Run miniML in ITS OWN interpreter and report event times as JSON.

This file is never imported by the package. It is executed by
`external.MiniMLDetector` with the interpreter where miniML and TensorFlow
2.14-2.15 are installed -- an environment whose numpy<2 pin makes it
unimportable alongside the rest of this package.

Because of that it must stay dependency-minimal and stdlib-only at import
time: no `multimethod` imports, no numpy 2 idioms. Anything it needs from the
parent arrives in the request JSON.

    python _miniml_worker.py request.json

Request keys: abf_path, channel, preprocessing{enabled,cutoff_hz,
target_rate_hz,order}, model_path, threshold, window_ms, batch_size,
direction, extra_args, out_path.

It always writes out_path, even on failure, with {"ok": false, "error": ...} --
the parent distinguishes "detector reported a problem" from "child died
without saying anything", and the two need different messages.
"""

import json
import os
import sys
import traceback


def main(argv):
    if len(argv) != 2:
        print("usage: _miniml_worker.py request.json", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        req = json.load(fh)
    out_path = req["out_path"]

    try:
        result = detect(req)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        result = {"ok": False,
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-2000:]}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0 if result.get("ok") else 1


def detect(req):
    import numpy as np
    import pyabf

    # Reproduce the parent's preprocessing so event times refer to the same
    # trace the other detectors saw. Duplicated from trace.py on purpose:
    # this file cannot import the package.
    pre = req.get("preprocessing") or {}
    abf = pyabf.ABF(req["abf_path"])
    abf.setSweep(0, channel=req["channel"])
    v = np.asarray(abf.sweepY, dtype=float)
    fs = float(abf.dataRate)
    if pre.get("enabled", True):
        from scipy.signal import bessel, sosfiltfilt
        sos = bessel(int(pre.get("order", 8)), float(pre.get("cutoff_hz", 3000.0)),
                     btype="low", fs=fs, output="sos")
        v = sosfiltfilt(sos, v)
        factor = max(1, int(round(fs / float(pre.get("target_rate_hz", 10000.0)))))
        v = v[::factor]
        fs = fs / factor

    from miniml import EventDetection
    from miniml.core.trace import MiniTrace

    unit = abf.adcUnits[req["channel"]] if req["channel"] < len(abf.adcUnits) else "pA"
    trace = MiniTrace(data=v, sampling_interval=1.0 / fs, y_unit=unit,
                      filename=os.path.basename(req["abf_path"]))

    model_path = req["model_path"]
    if not os.path.isabs(model_path):
        from miniml.resources.util import get_resource_file_path
        model_path = get_resource_file_path(model_path)

    # window_size is in SAMPLES; the shipped models were trained on a 600-sample
    # window at 50 kHz = 12 ms. Convert from real time so the model sees the
    # same span of signal whatever this recording's rate is.
    window_size = max(16, int(round(float(req["window_ms"]) / 1000.0 * fs)))

    detection = EventDetection(
        data=trace,
        model_path=model_path,
        window_size=window_size,
        event_direction=req.get("direction", "negative"),
        model_threshold=float(req["threshold"]),
        batch_size=int(req["batch_size"]),
        compile_model=True,
        verbose=0,
    )
    detection.detect_events(eval=True, peak_w=5, rel_prom_cutoff=0.25,
                            convolve_win=15, gradient_convolve_win=40,
                            use_legacy_baseline_method=True)

    times = _event_times(detection, fs)
    return {
        "ok": True,
        "event_times_s": [float(t) for t in times],
        "n_events": len(times),
        "sample_rate_hz": fs,
        "window_samples": window_size,
        "window_ms": float(req["window_ms"]),
        "model_path": model_path,
        "miniml_threshold": float(req["threshold"]),
    }


def _event_times(detection, fs):
    """Peak times in seconds, whichever attribute this miniML version exposes.

    miniML has moved these around between releases, so try the documented
    options in order rather than pinning one and breaking on upgrade.
    """
    import numpy as np

    for attr, in_samples in (("event_peak_times", False),
                             ("event_times", False),
                             ("event_peak_locations", True),
                             ("event_locations", True)):
        vals = getattr(detection, attr, None)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=float).ravel()
        if arr.size == 0:
            return arr
        return arr / fs if in_samples else arr
    raise RuntimeError(
        "could not find event times on the EventDetection object "
        f"(looked for event_peak_times/event_times/event_peak_locations; "
        f"available: {[a for a in dir(detection) if 'event' in a.lower()][:12]})")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
