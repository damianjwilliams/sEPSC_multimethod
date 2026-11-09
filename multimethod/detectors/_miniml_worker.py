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
target_rate_hz,order,line_hum_hz}, model_path, threshold, window_ms, batch_size,
direction, extra_args, miniml_rate_hz, use_legacy_baseline, out_path.

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
        # Decimate toward miniML's OWN native rate, not the package-wide
        # target. The rest of the package resamples to 10 kHz because that is
        # ample for measuring an event; miniML is different, because its
        # internal arithmetic is in samples and its shipped models were
        # trained at 50 kHz with a 600-sample window. Decimating a 100 kHz
        # recording to 10 kHz leaves a 12 ms window only 120 samples long,
        # which starves the model and breaks miniML's own baseline code.
        #
        # This does NOT make the results incomparable: the only thing crossing
        # back is event times in seconds, and the parent re-measures every
        # event on its own trace.
        target = float(req.get("miniml_rate_hz") or 50_000.0)
        factor = max(1, int(round(fs / target)))     # never upsample
        if factor > 1:
            v = v[::factor]
            fs = fs / factor
        line_hz = float(pre.get("line_hum_hz", 0.0) or 0.0)
        if line_hz > 0:
            # By file path, not as part of the package: linehum.py is numpy
            # only for exactly this reason.
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_mm_linehum", os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "linehum.py"))
            linehum = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(linehum)
            v, _ = linehum.remove_line_hum(v, fs, line_hz)

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

    # miniML's legacy baseline routine computes int(window_size * 0.1 / 10) and
    # then halves it, so below 200 samples that is 0 and the baseline slice is
    # empty -- mean([]) is NaN and miniML raises "Baseline could not be
    # determined". Say so here rather than letting it fail six frames down.
    legacy = bool(req.get("use_legacy_baseline", False))
    if legacy and window_size < 200:
        raise ValueError(
            f"window is {window_size} samples at {fs:.0f} Hz, and miniML's "
            f"legacy baseline method needs at least 200 (it computes "
            f"int(window*0.1/10)//2, which is 0 below that, giving an empty "
            f"baseline slice). Either leave use_legacy_baseline off, or raise "
            f"miniml_rate_hz so the window spans more samples.")

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
    # use_legacy_baseline_method=False by default: the v2 routine floors every
    # window size at 3 samples and cannot produce the empty slice the legacy
    # one does. The legacy flag was inherited from an older driver script and
    # is what made this fail on a 100 kHz recording.
    detection.detect_events(eval=True, peak_w=5, rel_prom_cutoff=0.25,
                            convolve_win=15, gradient_convolve_win=40,
                            use_legacy_baseline_method=legacy)

    times = _event_times(detection, fs)
    return {
        "ok": True,
        "event_times_s": [float(t) for t in times],
        "n_events": len(times),
        "sample_rate_hz": fs,
        "window_samples": window_size,
        "window_ms": float(req["window_ms"]),
        "use_legacy_baseline": legacy,
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
