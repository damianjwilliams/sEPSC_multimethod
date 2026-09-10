# sepsc-multimethod

Run **every** synaptic-event detector on one gap-free voltage-clamp `.abf`
recording, measure every event the same way, then review **which method found
which event** — on the trace itself.

Detecting sEPSCs is not a solved problem, and the honest state of the field is
that production tools disagree wildly on the same data. This package does not
try to pick a winner. It runs the methods side by side and shows you where they
agree, where they don't, and what the disputed events actually look like.

```
python launch.py                          # parameter window -> run all -> review
python run_all.py recording.abf           # run every method, write the tables
python review_methods.py recording.abf    # who found what, on the trace
python -m multimethod benchmark           # score the in-repo methods on planted events
python -m multimethod methods             # what can run on this machine
```

---

## The detectors

| Method | Kind | What it does |
|---|---|---|
| `threshold` | in-repo | Local maxima at least (c) apart, kept if they clear an amplitude threshold. The Synaptosoft **Mini Analysis** method. Fully explainable. |
| `deconvolution` | in-repo | Estimates the event waveform from *this* recording and deconvolves by it, so each event becomes a sharp impulse ([Pernía-Andrade 2012](https://pubmed.ncbi.nlm.nih.gov/23062335/)). Separates overlapping events; nothing to tune per cell. |
| `template` | in-repo | Fits an optimally-scaled template at every sample and thresholds scale/SE ([Clements & Bekkers 1997](https://pmc.ncbi.nlm.nih.gov/articles/PMC1180923/)). What Clampfit implements. |
| `miniml` | **external** | CNN-LSTM deep-learning classifier ([O'Neill 2025, eLife](https://elifesciences.org/articles/98485)). Runs in its own interpreter. |
| `fastmini` | **external** | Any detector you can run as a command. Pre-wired for a per-recording MLP. |

They fail *differently*, which is the point. The threshold detector fragments
decay tails; deconvolution is strongest on overlap; template matching is a
matched filter, so it reaches smaller events but is sensitive to kinetics it
wasn't built for.

### Why two of them run in another interpreter

miniML needs TensorFlow 2.14–2.15, which requires **numpy < 2**. A current
analysis environment has numpy 2.x. Those cannot coexist in one process, so
miniML is not importable alongside the rest of this package — a hard
constraint, not a packaging inconvenience. It therefore runs behind a
subprocess boundary, and the contract across it is deliberately minimal: the
child gets the `.abf` path plus the preprocessing settings, and returns
**event times in seconds** as JSON.

A missing environment is reported, not raised. The other methods still run:

```
  threshold      OK      775 events     2.9 s
  deconvolution  OK      752 events     3.0 s
  template       OK     1129 events     4.3 s
  miniml         OK      563 events    26.6 s
  fastmini       SKIP   disabled (set enabled=true and a command to use it)

4 of 5 method(s) ran -> 1474 consensus events
```

---

## The one design decision worth knowing

**A detector's only job here is to say WHERE an event is.** It returns sample
indices. It does not report the amplitude, rise, decay or area —
`events.measure_at` does, identically, for every method.

That split matters. Each detector has its own idea of what an amplitude is: the
threshold detector averages n samples at the peak against a fixed pre-peak
window, miniML has its own baseline convention, a template method reports a
fitted scale factor. If each reported its own numbers, the `amp_*` columns of
the consensus table would not be comparable, and "method A finds smaller events
than method B" could be an artefact of two conventions rather than a fact about
detection.

So detection is per-method and measurement is shared. A disagreement in the
output is then a disagreement about detection alone. The cost is that a
detector's own amplitude estimate is discarded; where one is informative it
goes in the run sidecar instead.

For the same reason the trace is loaded and preprocessed **once** and passed
read-only to every detector.

---

## Install

Python 3.10+.

```bash
cd sEPSC_multimethod
python -m venv .venv
.venv\Scripts\activate         # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

That runs the three in-repo detectors, the runner, and both windows. The
external detectors need their own environments — see below. Note the interactive
windows need PyQt5 and a display; detection itself needs neither.

### Wiring up miniML

miniML is not a dependency of this package and must not be installed into the
same environment. Point the package at an interpreter that has it:

```bash
python -m multimethod methods       # says whether it was found
```

If it wasn't, set `python_exe` in **Parameters ▸ miniml**, or in the config
file. A conda env named `clampex_miniml` is auto-detected.

### Wiring up any other detector

`fastmini` is a generic external adapter: give it a command and tell it which
CSV column holds the event times.

```json
{
  "detectors": {
    "fastmini": {
      "enabled": true,
      "command": "C:/path/to/env/python.exe -m sepsc fastmini {abf}",
      "csv_glob": "{stem}_fastmini_events.csv",
      "time_column": "location_s"
    }
  }
}
```

`{abf}`, `{channel}` and `{stem}` are substituted. This is how a third-party
detector gets in without this package depending on its internals.

---

## Output

Everything lands next to the `.abf`, keyed off one stem that encodes the
channel and the preprocessing — so two differently preprocessed runs cannot
overwrite each other.

| File | Contents |
|---|---|
| `<stem>_<method>_events.csv` | per method, one row per event: `peak_idx`, `peak_time_s`, `baseline`, `amplitude`, `rise_time_ms`, `decay_time_ms`, `area`, `inter_event_interval_ms` |
| `<stem>_multimethod_events.csv` | **the consensus table** — one row per event, `found_<method>` 0/1 and `amp_<method>` per method, plus `n_methods`, `amplitude_median`, `amplitude_spread` |
| `<stem>_multimethod_summary.csv` | per method: n events, how many it found *alone*, median amplitude |
| `<stem>_multimethod_agreement.csv` | pairwise Jaccard agreement matrix |
| `<stem>_multimethod_run.json` | every setting, every method's status and timing — enough to reproduce the run |

The per-method CSVs use the same eight columns as the sibling `minianalysis`
repo, so anything that reads one reads these.

### How "the same event" is decided

Detections within `tolerance_ms` (default 3 ms) of each other are treated as
one event. Two rules keep that honest:

- **Clusters don't chain.** Each detection must be within tolerance of the
  cluster's *mean*, so a 3 ms tolerance cannot absorb detections spanning 7 ms.
- **One detection per method per event.** A method that fragments one event into
  three gets three separate consensus events, rather than looking like three
  methods agreeing.

The tolerance is a real choice — the agreement numbers move with it — so it is
written into the run sidecar rather than left implicit.

---

## The review window

```bash
python review_methods.py recording.abf
python review_methods.py recording.abf --only-unique   # start on the disputed ones
```

Three panels:

- **top** — the whole trace, every consensus event marked, coloured by *how
  many* methods found it. Click one.
- **middle** — a raster, one row per method, one tick per detection. Read down a
  column to see who agreed and who missed. This answers the question no summary
  statistic can: *is this method finding extra events, or are the others missing
  them?* — because the answer differs between events in the same recording.
- **bottom** — the selected event, zoomed, with the measurement windows drawn
  and one row per method: found or not, its amplitude, and how far its peak sat
  from the consensus time.

Colouring is by agreement count rather than by method on purpose. With five
methods there are 31 possible combinations and a colour per combination is
unreadable; agreement count is one ordered number that maps onto a sequential
scale, and it is what you actually scan for.

`A`/`R` accept or reject, autosaved after every decision to
`<stem>_multimethod_reviewed.csv` and `..._review_progress.csv`, and resumed
next time. "Accept all unanimous" is the safe bulk action.

---

## The parameter window

```bash
python launch.py
```

One tab per method, plus three that apply to all of them:

- **Preprocessing** — filter/resample, applied once for every detector
- **Measurement** — how an event is quantified, shared by all methods
- **Consensus** — how close two methods must be to count as agreeing

Each tab is generated from that detector's own params dataclass, with the field
docstrings as tooltips. A new parameter shows up in the window as soon as it
exists on the dataclass — a hand-written form would drift, and a parameter
window that silently omits a parameter is worse than none.

"Save as defaults" writes `multimethod.config.json`. That file holds
interpreter paths and external commands, which are properties of a machine
rather than of the analysis, so it is gitignored.

---

## Benchmark

```bash
python -m multimethod benchmark
```

Plants biexponential events at known times and scores each in-repo detector
across eight conditions (decay τ, event rate, white and 1/f noise, drift, small
amplitudes):

```
       condition    threshold  deconvoluti     template
   baseline 5 Hz        0.965        0.992        0.953
        tau 3 ms        0.985        0.985        0.947
       tau 25 ms        0.796        0.978        0.708
      rate 20 Hz        0.887        0.960        0.761
      rate 40 Hz        0.716        0.907        0.618
      pink noise        0.980        0.992        0.759
     drift 40 pA        0.965        0.992        0.953
   small (SNR 4)        0.873        0.919        0.841
         mean F1        0.896        0.966        0.818
  mean precision        0.950        0.997        0.909
     mean recall        0.866        0.939        0.779
```

**Read that with the obvious caveat.** The planted events are biexponential,
and every detector here uses a biexponential template or kernel, so the fixture
flatters all of them — and flatters most the one that estimates its kernel from
the data. It is for catching a detector that is outright broken, and for
comparing a change against the same fixture. It is **not** evidence about which
method is best on your recordings; for that, run them all and use the review
window.

Two defaults were set from this benchmark rather than from the literature, and
both differ from the textbook value for a reason documented in the code:

- `threshold.search_local_max_ms` is **8 ms**, not the tutorial's 3 ms. At 3 ms
  the window is far shorter than an event's decay, so noise on the tail
  registers as further local maxima and one event is reported several times.
  That single change moves mean F1 from 0.603 to 0.900 (precision 0.44 → 0.93).
- `template.criterion_threshold` is **14**, not Clements & Bekkers' 3–5. Their
  value is right for white noise; the SE in the denominator assumes independent
  residuals, and this package low-pass filters by default, which correlates
  adjacent samples and inflates the criterion. At 4 the F1 on 1/f noise is 0.16;
  at 14 it is 0.76.

---

## Layout

```
run_all.py   launch.py   review_methods.py
   └─ thin wrappers, so each entry point runs as a plain script from the root

multimethod/
    trace.py        loading + preprocessing, once, shared read-only
    events.py       the Event record and the ONE shared measurement routine
    consensus.py    matching each method's detections into shared events
    runner.py       run every detector, collect failures, write the tables
    settings.py     every parameter for every method, serialisable
    benchmark.py    planted ground truth and scoring
    paramwindow.py  the parameter window (tabs generated from the dataclasses)
    review.py       the review window (who found what, on the trace)
    cli.py          `python -m multimethod <command>` dispatch
    detectors/
        base.py           what a detector is; DetectorResult
        threshold.py      classical Mini Analysis
        deconvolution.py  Pernía-Andrade, data-estimated kernel
        template.py       Clements & Bekkers
        external.py       the subprocess boundary; miniML + command adapters
        _miniml_worker.py runs in miniML's interpreter, never imported here

tests/              61 tests: `pytest` from the repo root
```

## Tests

```bash
pip install pytest
pytest
```

61 tests. The consensus matcher gets the most attention because it is this
package's own logic rather than a wrapper: clusters must not chain, one method
must not occupy a cluster twice, and the tolerance must actually change the
answer. Each detector is checked to find planted events, to stay quiet on pure
noise, to survive a flat and an empty trace, and not to mutate the trace it was
handed. The runner is checked to keep going when one detector fails, since with
external detectors a partial run is the normal case. The GUI tests drive Qt
through its offscreen platform and skip if PyQt5 is absent.

## Credit

`threshold` reimplements the Synaptosoft **Mini Analysis Program** method
(tutorial by C. Justin Lee, PhD). `deconvolution` follows Pernía-Andrade et al.
2012. `template` follows Clements & Bekkers 1997. `miniml` is a wrapper around
[miniML](https://github.com/delvendahl/miniML) by the Delvendahl lab, which is
neither vendored nor modified here. None of these authors are affiliated with
this package.

MIT licensed.
