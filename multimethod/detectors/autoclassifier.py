"""Unsupervised state-clustering detection, after Pircher et al. 2022.

Pircher, Pircher & Feigenspan, "A novel machine learning-based approach for
the detection and analysis of spontaneous synaptic currents", PLOS ONE 17(9):
e0273501. https://doi.org/10.1371/journal.pone.0273501

The idea, and why it is worth having here
-----------------------------------------
Every other learned detector in this package needs labels: miniML is trained
on hand-scored events, and a template or threshold encodes a human's model of
what an event looks like. This one needs none. It encodes short windows of the
trace into a low-dimensional space, clusters that space, and then treats an
event as a characteristic *progression through cluster states*. Nothing tells
it what an event is.

That makes it fail differently from everything else here, which is the whole
point of the package: it cannot inherit a labelling bias, but it also has no
way to know that the structure it found is synaptic rather than, say, the
recording's own noise texture.

What this is NOT
----------------
This is an implementation of the paper's approach, not a port of the authors'
`openseapy`. That package is GitHub-only, installs via the deprecated
`setup.py install`, depends on a second GitHub-only package (`eventsearch`),
and exposes no command line, so it cannot be reached through this package's
external-command adapter either. Two deliberate deviations:

  1. **The encoder is linear.** The paper uses an autoencoder (1024 -> 256 ->
     8, mirrored decoder). Here the 8-dimensional code comes from PCA, via an
     SVD of centred windows. The method's novelty is the clustering of window
     states, not the nonlinearity of the encoder, and a linear encoder keeps
     this detector on numpy/scipy -- adding Keras or torch for one detector
     would cost every user of the package an install.

  2. **Which cluster means "event" is decided by a slope rule.** The paper
     trains on retinal horizontal cells and interprets the learned states
     against that preparation. Unsupervised clustering gives k states but no
     names, so something has to say which state is an event onset. Here it is
     the cluster whose centroid has the steepest rise, which is the property
     that distinguishes an event's leading edge from baseline or decay. That
     rule is mine, not the paper's.

Both deviations are why this is registered as `autoclassifier` rather than as
the authors' name for it, and why its benchmark numbers should not be read as
a test of their method.

One addition of mine that the paper does not need: a gate asking whether the
winning state actually rises like an event onset (`min_state_rise_sd`). An
unsupervised clusterer has no way to say "this recording contains no events" --
given pure noise it still finds k states and one still rises most -- so without
it the detector fires at ~70 Hz on noise. The gate is on the STATE, not on
individual events, so it decides whether an event-like state was found at all
rather than filtering events by amplitude.

A further caveat from the paper itself: the window is 5.12 ms (128 samples at
25 kHz), which is shorter than most sEPSC decays. For detection that is
tolerable -- this package measures every event itself and only needs times --
but the encoder never sees a whole event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..trace import Trace, robust_sd
from .base import Detector


@dataclass(frozen=True)
class AutoclassifierParams:
    window_ms: float = 5.12
    """Length of each encoded window, ms. The paper's 128 samples at 25 kHz.
    Shorter than an sEPSC decay on purpose -- the encoder is meant to capture
    the leading edge, not the whole event."""

    n_components: int = 8
    """Size of the encoded representation. The paper's autoencoder bottleneck.
    More components capture more waveform detail and blur the clusters."""

    n_clusters: int = 7
    """Number of window states to discover. The paper's value. Raising it
    splits baseline into more textures; lowering it merges the event onset
    state into the decay state, which loses the transition being detected."""

    min_state_rise_sd: float = 2.0
    """How much the chosen onset state must rise across its window, in noise
    SDs, before any events are reported at all.

    An unsupervised clusterer has no concept of "this recording contains no
    events": given pure noise it still finds k states, and one of them still
    has the steepest rise, so without this gate the detector reports noise
    excursions at ~70 Hz. Asking whether the winning state actually looks like
    an event onset separates the two cases cleanly -- measured on planted data
    the ratio is 8.1 with events present and 0.06 on noise alone, two orders
    of magnitude apart, so anything from ~1 to ~5 works.

    This gates the STATE, not individual events: it decides whether the
    clustering found an event-like state at all, and does not filter
    event-by-event on amplitude. That distinction is what keeps this a
    state-clustering detector rather than a threshold detector wearing one as
    a hat.
    """

    min_sep_ms: float = 8.0
    """Minimum spacing between accepted events, ms. Needed because a single
    event spends several consecutive windows in the onset state."""

    max_fit_windows: int = 20_000
    """Windows sampled to fit the encoder and the clusters. Fitting on every
    window of a ten-minute recording is unnecessary -- the states are a
    property of the signal, not of how many windows you show it -- and the
    full design matrix would be gigabytes."""

    chunk_windows: int = 100_000
    """Windows projected at a time. The windows are a stride view, so they
    cost nothing until projected; doing it in chunks keeps that bounded."""

    seed: int = 0
    """Seed for the window subsample and the k-means start, so a rerun on the
    same recording gives the same events."""

    direction: str = "negative"
    """Event polarity: "negative" for inward currents, "positive" for
    outward."""

    subtract_drift: bool = True
    """Encode a drift-corrected trace. Without it the slowest PCA component
    is the drift and one whole cluster is spent on "which part of the
    recording is this", rather than on window shape."""

    drift_win_ms: float = 300.0
    """Window for that rolling baseline, ms. See Measurement."""


def _windows(sv: np.ndarray, n: int) -> np.ndarray:
    """Sliding windows as a VIEW, not a copy.

    A 120 s recording at 10 kHz gives 1.2 M windows; at 128 samples each, a
    materialised matrix is ~1.2 GB. `sliding_window_view` costs nothing until
    the data is actually read, which is why the projection below is chunked.
    """
    return np.lib.stride_tricks.sliding_window_view(sv, n)


def fit_encoder(windows: np.ndarray, n_components: int, rng, max_fit: int):
    """PCA basis for window SHAPE, from a random subsample.

    Each window is centred on its own mean first. Without that the leading
    component is simply the window's DC level, the clusters separate by
    absolute current rather than by shape, and the detector reduces to a
    slow threshold.
    """
    n_win = windows.shape[0]
    take = min(max_fit, n_win)
    idx = rng.choice(n_win, take, replace=False) if take < n_win else np.arange(n_win)
    sample = np.asarray(windows[np.sort(idx)], dtype=float)
    sample = sample - sample.mean(axis=1, keepdims=True)
    # economy SVD of the centred sample: rows are windows, so the right
    # singular vectors are the window-shape basis
    _u, _s, vt = np.linalg.svd(sample, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return vt[:k], sample


def encode(windows: np.ndarray, basis: np.ndarray, chunk: int) -> np.ndarray:
    """Project every window onto the basis, in chunks."""
    n_win = windows.shape[0]
    out = np.empty((n_win, basis.shape[0]), dtype=float)
    for start in range(0, n_win, chunk):
        stop = min(start + chunk, n_win)
        block = np.asarray(windows[start:stop], dtype=float)
        block = block - block.mean(axis=1, keepdims=True)
        out[start:stop] = block @ basis.T
    return out


def event_cluster(centroids: np.ndarray, basis: np.ndarray) -> tuple[int, float]:
    """Which cluster is the event-onset state, and how much it rises.

    Decoded back to window space, the onset state is the one whose mean
    waveform rises most across the window -- baseline states are flat and
    decay states fall. The rise is returned in trace units so the caller can
    ask whether it is large enough to be an event at all: on pure noise the
    steepest of k states still wins, but it is flat.

    This rule is this implementation's, not the paper's: unsupervised
    clustering yields states without names, and something has to name one.
    """
    shapes = centroids @ basis                      # back to window space
    n = shapes.shape[1]
    third = max(1, n // 3)
    rise = shapes[:, -third:].mean(axis=1) - shapes[:, :third].mean(axis=1)
    best = int(np.argmax(rise))
    return best, float(rise[best])


class AutoclassifierDetector(Detector):
    name = "autoclassifier"
    label = "Autoclassifier (Pircher 2022, adapted)"
    blurb = ("Unsupervised: encodes short windows, clusters them into states, "
             "and detects the state transition an event onset makes. No labels.")
    params_cls = AutoclassifierParams

    def find(self, trace: Trace, params: AutoclassifierParams) -> np.ndarray:
        from scipy.cluster.vq import kmeans2

        p = params
        sv = self._upright(trace, p.direction, p.subtract_drift, p.drift_win_ms)
        n = max(8, trace.samples(p.window_ms))
        if sv.size < n * 4:
            return np.array([], dtype=int)

        # A flat or near-flat trace has no states to find. Returning early
        # also keeps scipy's k-means from warning about empty clusters and
        # dividing by a zero distance sum on degenerate input.
        noise = robust_sd(sv)
        if not np.isfinite(noise) or noise <= 0:
            self.last_diagnostics = {"skipped": "trace has no variation",
                                     "encoder": "pca-linear"}
            return np.array([], dtype=int)

        rng = np.random.default_rng(p.seed)
        windows = _windows(sv, n)
        basis, sample = fit_encoder(windows, p.n_components, rng, p.max_fit_windows)

        # Cluster in the encoded space, fitted on the same subsample. kmeans2
        # with a fixed seed so a rerun reproduces the events exactly.
        sample_codes = sample @ basis.T
        k = min(p.n_clusters, max(2, sample_codes.shape[0] // 10))
        centroids, _ = kmeans2(sample_codes, k, minit="++", seed=p.seed)

        codes = encode(windows, basis, p.chunk_windows)
        # nearest centroid = the paper's classifier, exactly rather than
        # approximated by a second network
        d2 = ((codes[:, None, :] - centroids[None, :, :]) ** 2).sum(-1) \
            if codes.shape[0] <= 20_000 else None
        if d2 is None:
            labels = np.empty(codes.shape[0], dtype=int)
            for start in range(0, codes.shape[0], p.chunk_windows):
                stop = min(start + p.chunk_windows, codes.shape[0])
                block = codes[start:stop]
                dist = ((block[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
                labels[start:stop] = np.argmin(dist, axis=1)
        else:
            labels = np.argmin(d2, axis=1)

        target, rise = event_cluster(centroids, basis)
        rise_sd = rise / noise if noise > 0 else 0.0
        in_state = labels == target
        # transitions INTO the onset state, which is the event's leading edge
        entries = np.flatnonzero(in_state & ~np.concatenate(([False], in_state[:-1])))

        self.last_diagnostics = {
            "window_samples": n,
            "n_windows": int(windows.shape[0]),
            "n_clusters": int(centroids.shape[0]),
            "event_cluster": target,
            "state_rise": rise,
            "state_rise_sd": rise_sd,
            "noise_sd": noise,
            "cluster_occupancy": np.bincount(labels, minlength=centroids.shape[0])
                                  / max(labels.size, 1),
            "n_state_entries": int(entries.size),
            "encoder": "pca-linear",
        }
        if rise_sd < p.min_state_rise_sd:
            # No discovered state looks like an event onset, so report none
            # rather than the steepest flat state's noise excursions.
            self.last_diagnostics["skipped"] = (
                f"no event-like state: best state rises {rise_sd:.2f} noise SDs, "
                f"below min_state_rise_sd={p.min_state_rise_sd}")
            return np.array([], dtype=int)
        if entries.size == 0:
            return np.array([], dtype=int)

        # A window is labelled by its START, and the onset state is entered
        # before the peak: snap forward to the local maximum.
        reach = n * 2
        peaks = []
        for i in entries:
            lo = int(i)
            hi = min(sv.size, lo + reach)
            if hi - lo < 2:
                continue
            peaks.append(lo + int(np.argmax(sv[lo:hi])))
        if not peaks:
            return np.array([], dtype=int)

        # One event occupies several consecutive onset windows, so dedupe.
        sep = trace.samples(p.min_sep_ms)
        peaks = np.array(sorted(set(peaks)), dtype=int)
        kept = [peaks[0]]
        for q in peaks[1:]:
            if q - kept[-1] >= sep:
                kept.append(q)
            elif sv[q] > sv[kept[-1]]:
                kept[-1] = q                        # keep the taller of a pair
        return np.array(kept, dtype=int)
