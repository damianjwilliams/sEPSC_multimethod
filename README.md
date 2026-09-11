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
python launch.py --defaults               # ...pre-filled with the built-in values
python run_all.py recording.abf           # headless; add --gui for the window first
python review_methods.py recording.abf    # who found what, on the trace
python -m multimethod batch FOLDER --list # what protocols are in this folder?
python -m multimethod batch FOLDER --protocol gap_free   # run the ones that match
python -m multimethod benchmark           # score the in-repo methods on planted events
python -m multimethod methods             # what can run on this machine
python -m multimethod calibrate FOLDER    # set defaults from your own data
```

---

## The detectors

| Method | Kind | What it does |
|---|---|---|
| `threshold` | in-repo | Local maxima at least (c) apart, kept if they clear an amplitude threshold. The Synaptosoft **Mini Analysis** method. Fully explainable. |
| `deconvolution` | in-repo | Estimates the event waveform from *this* recording and deconvolves by it, so each event becomes a sharp impulse ([Pernía-Andrade 2012](https://pubmed.ncbi.nlm.nih.gov/23062335/)). Separates overlapping events; nothing to tune per cell. |
| `template` | in-repo | Fits an optimally-scaled template at every sample and thresholds scale/SE ([Clements & Bekkers 1997](https://pmc.ncbi.nlm.nih.gov/articles/PMC1180923/)). What Clampfit implements. |
| `autoclassifier` | in-repo | **Unsupervised**: encodes short windows, clusters them into states, detects the state transition an onset makes ([Pircher 2022](https://doi.org/10.1371/journal.pone.0273501)). No labels, no template. |
| `miniml` | **external** | CNN-LSTM deep-learning classifier ([O'Neill 2025, eLife](https://elifesciences.org/articles/98485)). Runs in its own interpreter. |
| `fastmini` | **external** | Any detector you can run as a command. Pre-wired for a per-recording MLP. |

They fail *differently*, which is the point. The threshold detector fragments
decay tails; deconvolution is strongest on overlap; template matching is a
matched filter, so it reaches smaller events but is sensitive to kinetics it
wasn't built for; the autoclassifier uses no labels or template at all, so it
cannot inherit a labelling bias -- but it also has no way to know that the
structure it found is synaptic rather than the recording's own noise texture.

`autoclassifier` is an **adaptation** of Pircher et al., not a port of their
`openseapy` (which is GitHub-only, needs a second GitHub-only package, and has
no CLI). Its encoder is linear (PCA) rather than an autoencoder, and the rule
naming the onset state is this package's. See the module docstring.

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

**miniML gets its own sample rate.** The rest of the package resamples to
10 kHz, which is ample for measuring an event. miniML is different: its
internal arithmetic is in samples and its shipped models were trained at 50 kHz
with a 600-sample window, so it is given 50 kHz (`miniml_rate_hz`) and never
upsampled above the recording's own rate. Decimating a 100 kHz recording to
10 kHz would leave a 12 ms window only 120 samples long, which starves the
model and trips miniML's own baseline code. This does not make its results
incomparable: only event times cross the subprocess boundary, and every event
is re-measured here.

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

**Expanding the trace, each axis independently.**

| Input | Effect |
|---|---|
| **click and drag** | **slide along the recording** - the trace follows the mouse |
| `shift` + drag | slide the amplitude axis instead |
| click (no drag) | jump to the nearest event |
| scroll | zoom time **and** amplitude together |
| `ctrl` + scroll | stretch the **time base** only |
| `shift` + scroll | stretch the **amplitude** axis only |
| `Home` or `0` | reset |

Dragging needs no toolbar tool selected. A press that does not move is still a
click, so panning never costs you a selection - and a drag that happens to
start on an event marker pans rather than selecting, which is why selection is
handled on button release rather than through matplotlib's pick event (that
fires on press, before a drag is distinguishable from a click).

Zoom is anchored on the cursor, so magnifying a burst keeps it under the
pointer. The modifier state is read from Qt rather than from matplotlib's
`event.key`, which is only populated once the canvas has seen a key press --
reading it meant holding `ctrl` and scrolling did nothing unless you had
clicked the canvas first. The raster **shares the
trace's x axis**, so zooming into a burst up top narrows the raster to the same
window and the per-method ticks stay aligned under it. The detail panel has its
own axes and zooms independently, so you can magnify one event without losing
your place in the recording.

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

It opens before anything runs, with a tab per method plus three shared ones --
**Preprocessing, Measurement, Consensus, threshold, deconvolution, template,
autoclassifier, miniml, fastmini** -- every field pre-filled and editable, an
enable checkbox per method, and OK / Cancel / Save as defaults / Reset to
defaults. Cancel runs nothing. Reach it with:

```bash
python launch.py                       # the front door
python launch.py --defaults            # pre-filled with the built-in values
python run_all.py recording.abf --gui  # same window, then run
```

Pre-filled *from whatever settings were resolved* — which is a config file if
one exists, not necessarily the built-in defaults. The banner at the top of the
window says which, and `--defaults` opens it on the values the package ships
with, leaving any config file untouched. Without that flag there is no way to
see the built-in values once a config exists, because a calibrated 0.5 ms in a
spin box looks exactly like a default.

Each tab is generated from that detector's own params dataclass, with the field
docstrings as tooltips. A new parameter shows up in the window as soon as it
exists on the dataclass — a hand-written form would drift, and a parameter
window that silently omits a parameter is worse than none.

"Save as defaults" writes `multimethod.config.json`. That file holds
interpreter paths and external commands, which are properties of a machine
rather than of the analysis, so it is gitignored.

---

## The config file, and why every command announces it

`multimethod.config.json` beside the package is picked up automatically by
every command. Two things write it: the parameter window's "Save as defaults"
button, and `calibrate --write`.

That convenience has a sharp edge, and it drew blood here. A config written by
`calibrate --write` was in effect for later runs, and those runs gave no sign
of it — an 11.13 pA amplitude floor arriving from an earlier day changed
deconvolution's event count by 11% with nothing on screen to say why. The
values were in each run's sidecar, so nothing was unrecoverable, but
*recoverable afterwards* is not *visible at the time*.

So no command loads settings without reporting them. Every run, `methods`,
`review`, and `calibrate` starts with either

```
Parameters: built-in defaults (no multimethod.config.json).
```

or the file, named, with every value that differs from the built-in default:

```
Parameters: multimethod.config.json (found beside the package) overrides 6 built-in default(s):
    detectors.deconvolution.min_amp_abs            0.0 -> 11.13
    detectors.template.tau_decay_ms                6.0 -> 1.73
    tolerance_ms                                   3.0 -> 0.5
    ...
  This file is picked up automatically. To run on the built-in defaults, rename it or pass --config
  with a different path.
```

The same fact is recorded under `config_source` in each run's
`_multimethod_run.json`, so a result on disk can be traced to the file that
shaped it — not just to the values, but to where they came from. A config that
exists but matches the defaults is reported as inert rather than passed over in
silence, and a path you gave with `--config` is stated without the warning,
since you asked for it.

To go back to the built-in defaults, rename or delete the file. Nothing deletes
it for you; the fix here was to make it loud, not to take the feature away.

---

## The analysis figures

Written at the end of every run (`--no-plots` to skip):

| Figure | Contents |
|---|---|
| `<stem>_multimethod_distributions.png` | amplitude, rise, decay, area and IEI - histogram **and** cumulative, every method overlaid |
| `<stem>_multimethod_boxplots.png` | per method: box (median + IQR), whiskers, and **every individual event as a dot**, with n |
| `<stem>_multimethod_waveforms.png` | peak-aligned average event per method, raw *and* peak-normalised |
| `<stem>_multimethod_overview.png` | amplitude over time (stability), event frequency, agreement histogram, pairwise agreement matrix, amplitude by agreement level |

**No bar-of-the-mean charts.** A bar with an error bar hides everything that
matters about a synaptic amplitude distribution: it is skewed, truncated at the
detection limit, and its interesting feature is usually a change in *shape*
rather than in the mean - two very different distributions give the same bar.
So wherever there is a distribution, this draws a box, whiskers, and every
measurement as a jittered dot. Above 400 points per group the dots are
subsampled for legibility; the box, the whiskers and the reported n always use
every point.

Bars survive in exactly two places, both pure counts - event frequency and the
agreement histogram. A count has no distribution, and a box plot of one number
would be worse than the bar rather than better.

The cumulative panels are the ones to compare distributions on: a histogram's
shape depends on its bin width, a CDF does not.

One result worth looking for on your own data: in `_multimethod_overview.png`,
median |amplitude| rises monotonically with agreement level (8 -> 13 -> 17 ->
20 pA on the test recording). Events only one method finds are small events
near the noise floor - the expected pattern, and a useful check that the
consensus is behaving.

## Running a whole folder, by protocol

A day's folder holds every protocol you ran. Only some are the gap-free sEPSC
recordings, and the recording knows which it is -- pCLAMP writes the protocol
name into the ABF header. So the selector is the protocol, matched by a
case-insensitive **substring**, rather than a filename convention you have to
maintain.

```bash
python -m multimethod batch FOLDER --list                 # look first
python -m multimethod batch FOLDER --protocol gap --dry-run
python -m multimethod batch FOLDER --protocol gap
```

`--list` reads headers only, so it is near-instant even on a large folder --
88 recordings survey in about a second:

```
protocol                                        n
--------------------------------------------------
DW_Ra                                          55
DW_IC_IV                                       20
IC_IV                                           8
DW_Gap_free_-70_mV                              5
```

Each recording gets the usual per-recording files, plus one
**`multimethod_batch_summary.csv`** for the folder. Its first column is the
**absolute file path** -- everything downstream joins on it, and a bare
filename is not unique across a folder tree. Then the protocol, duration,
**`noise_sd`** and its 4-sigma limit, `n_` and `rate_..._hz` per method,
consensus and unanimous counts, which methods ran, which failed, and the output
stem.

Three things it refuses to do quietly:

- **A multi-sweep protocol is skipped, with the sweep count as the reason.**
  This package reads sweep 0 of a gap-free recording; analysing the first sweep
  of a 50-sweep IV protocol is a wrong answer, which is worse than a missing
  one. On the folder above that guard alone holds back 55 `DW_Ra` files.
- **A substring spanning two protocols is flagged.** `IC_IV` matches both
  `IC_IV` and `DW_IC_IV`; pooling them silently would merge two experiments.
- **One bad file costs one row, not the batch.** Failures become rows with a
  reason, and the summary is rewritten after every recording, so an interrupted
  run keeps what it already did.

It finishes by reporting the spread of `noise_sd` across the set, and points at
the document below when that spread is wide enough to matter -- which on real
data it usually is.

---

## Comparing groups

Before using this for a genotype or treatment comparison, read
**[docs/comparing-groups.md](docs/comparing-groups.md)**. The short version:
the cell is the unit of analysis, and a difference in recording noise alone
produces a 30-38% frequency change and a ~65% amplitude change on real
recordings here -- in a self-consistent direction that reads as a real
phenotype. The document derives the shared amplitude floor that corrects it
(it is detector-specific, and higher than 4 sigma), and prices sigma as a
cell-exclusion criterion. Every figure in it is reproduced by the three
scripts beside it.

---

## Calibrating the defaults to your own data

```bash
python -m multimethod calibrate FOLDER            # report only
python -m multimethod calibrate FOLDER --write    # save as defaults
```

Every default here was chosen on planted synthetic events. That catches a
broken detector, but the fixture is not your preparation. This pools the
high-confidence events across a folder of recordings, measures what they
actually are, and writes the parameters that follow.

Calibrated from the data: the measurement search windows, the template and
fallback-kernel time constants, the **consensus tolerance** (from the *measured*
peak-time disagreement between methods, rather than a guess at it), and the
4-sigma detection limit (from the noise, not from the events).

Two rules keep it safe to run unattended:

- **Widen only.** A search window is never narrowed. The pooled events were
  selected *by* the current window, so the tail beyond it is invisible --
  narrowing to the p99 of what got through would lock that bias in and make
  slower events permanently unmeasurable.
- **Kinetics are flagged, not forced.** A kernel faster than the real events is
  this package's worst failure mode, so an implausible time constant is
  withheld and reported instead of written.

It reports what it withheld and why, warns where a change alters bookkeeping
rather than detection, and names the parameters it deliberately leaves alone.

### Calibration is not training

This does **not** train a model on detected events. That would be circular: the
model would learn to reproduce the detectors, errors included, and the events
you most need labelled are exactly the disputed ones the detectors cannot
label. What it measures are properties of the *recordings* -- kinetics, noise,
inter-method timing spread -- which are facts about your data whichever method
found them.

The thresholds that trade precision against recall (`amplitude_sd`,
`thresh_sd`, `criterion_threshold`, `min_state_rise_sd`) are therefore left
alone. To set those you need ground truth: review a recording, and its
`*_multimethod_reviewed.csv` becomes a labelled set that can drive miniML
transfer learning (~400 events is enough) or an Eventer-style classifier.

`--write` saves the proposals to `multimethod.config.json`, which then applies
to every later command whether or not it mentions the file. Each of those
commands says so on start — see [the config file
section](#the-config-file-and-why-every-command-announces-it).

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
    calibrate.py    set the defaults from a folder of real recordings
    batch.py        run a folder, selected by protocol name
    plots.py        the analysis figures (box + whisker + dots, no bar means)
    paramwindow.py  the parameter window (tabs generated from the dataclasses)
    review.py       the review window (who found what, on the trace)
    cli.py          `python -m multimethod <command>` dispatch
    detectors/
        base.py           what a detector is; DetectorResult
        threshold.py      classical Mini Analysis
        deconvolution.py  Pernía-Andrade, data-estimated kernel
    template.py       Clements & Bekkers
        autoclassifier.py unsupervised state clustering (Pircher, adapted)
        external.py       the subprocess boundary; miniML + command adapters
        _miniml_worker.py runs in miniML's interpreter, never imported here

docs/
    comparing-groups.md  using this for a genotype comparison: the noise
                         confound, the shared amplitude floor, sigma as an
                         exclusion criterion, and what is still manual
    noise_effect.py      the measurements that document cites, rerunnable:
    floor_search.py        noise in isolation, where a floor fixes it, and
    sigma_cutoff.py        what a sigma exclusion costs

tests/             171 tests: `pytest` from the repo root
```

## Tests

```bash
pip install pytest
pytest
```

120 tests. The consensus matcher gets the most attention because it is this
package's own logic rather than a wrapper: clusters must not chain, one method
must not occupy a cluster twice, and the tolerance must actually change the
answer. Each detector is checked to find planted events, to stay quiet on pure
noise, to survive a flat and an empty trace, and not to mutate the trace it was
handed. The runner is checked to keep going when one detector fails, since with
external detectors a partial run is the normal case. The GUI tests drive Qt
through its offscreen platform and skip if PyQt5 is absent.

The review window's zoom is tested through the pure `zoom_limits` / `zoom_axes`
functions rather than by driving a window: the arithmetic - anchoring on the
cursor, round-tripping exactly on zoom-in-then-out, clamping a stale anchor -
is the part that can be wrong, and testing it directly makes the assertions
exact. The trace/raster x-axis linkage does need a window, and is asserted on
the shared-axes group in the one test that builds one.

## Credit

`threshold` reimplements the Synaptosoft **Mini Analysis Program** method
(tutorial by C. Justin Lee, PhD). `deconvolution` follows Pernía-Andrade et al.
2012. `template` follows Clements & Bekkers 1997. `miniml` is a wrapper around
[miniML](https://github.com/delvendahl/miniML) by the Delvendahl lab, which is
neither vendored nor modified here. None of these authors are affiliated with
this package.

MIT licensed.
