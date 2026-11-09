"""Mains hum removal: takes the hum out, and leaves everything else alone."""

from __future__ import annotations

import numpy as np

from multimethod import detectors
from multimethod.benchmark import score, synth
from multimethod.linehum import remove_line_hum
from multimethod.trace import Preprocessing, Trace

FS = 10_000.0


def hum_burst(n: int, fs: float, t0: float, t1: float, f0: float = 60.03,
              amp: float = 12.0) -> np.ndarray:
    """A burst of rig-like hum: 60 Hz plus odd harmonics out to ~2.5 kHz,
    slightly off nominal, switched on and off with short ramps."""
    t = np.arange(n) / fs
    wave = amp * np.sin(2 * np.pi * f0 * t)
    for k in range(3, 43, 2):
        wave += amp * 0.06 * np.sin(2 * np.pi * k * f0 * t + 0.7 * k)
    gate = np.clip(np.minimum(t - t0, t1 - t) / 0.05, 0, 1)
    return wave * gate


def as_trace(v: np.ndarray) -> Trace:
    return Trace(t=np.arange(v.size) / FS, v=v, dt=1 / FS, unit="pA",
                 abf_path="<synthetic>", channel=0,
                 preprocessing=Preprocessing(enabled=False))


def test_a_recording_without_hum_comes_back_unchanged():
    v, _ = synth(fs=FS, duration_s=20.0, seed=3)
    out, seconds = remove_line_hum(v, FS)
    assert seconds == 0.0
    assert np.array_equal(out, v)


def test_hum_is_removed_and_only_where_it_was():
    v, _ = synth(fs=FS, duration_s=30.0, seed=4)
    hum = hum_burst(v.size, FS, 10.0, 14.0)
    out, seconds = remove_line_hum(v + hum, FS)

    assert 3.0 <= seconds <= 6.0
    t = np.arange(v.size) / FS
    inside = (t > 10.5) & (t < 13.5)
    left = np.sqrt(np.mean((out - v)[inside] ** 2))
    assert left < 0.1 * np.sqrt(np.mean(hum[inside] ** 2))
    # Well away from the burst, not one sample differs.
    far = (t < 8.0) | (t > 16.0)
    assert np.array_equal(out[far], v[far])


def test_removal_gives_back_the_events_hum_hides_and_drops_the_ones_it_fakes():
    v, true_s = synth(fs=FS, duration_s=40.0, rate_hz=5.0, seed=5)
    hummy = v + hum_burst(v.size, FS, 8.0, 14.0) + hum_burst(v.size, FS, 25.0, 30.0)
    cleaned, _ = remove_line_hum(hummy, FS)

    det = detectors.get("deconvolution")
    params = det.default_params()
    found = {name: score(true_s, det.find(as_trace(x), params) / FS)
             for name, x in (("clean", v), ("hum", hummy), ("removed", cleaned))}
    assert found["hum"]["fp"] > found["clean"]["fp"] + 20
    assert found["removed"]["fp"] <= found["clean"]["fp"] + 3
    assert found["removed"]["recall"] >= found["clean"]["recall"] - 0.02


def test_off_is_off():
    v, _ = synth(fs=FS, duration_s=10.0, seed=6)
    hummy = v + hum_burst(v.size, FS, 2.0, 8.0)
    out, seconds = remove_line_hum(hummy, FS, line_hz=0.0)
    assert seconds == 0.0 and np.array_equal(out, hummy)


def test_hum_setting_is_in_the_output_name():
    on = Preprocessing().suffix()
    off = Preprocessing(line_hum_hz=0.0).suffix()
    assert on != off and "hum60Hz" in on
