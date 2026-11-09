# Comparing sEPSCs between genotypes, with multiple cells

Guidance for using this package in a group comparison, and the measurements
behind it. Every number here was produced on real recordings in
`C:\Users\damia\sEPSC_analysis` by the three scripts alongside this file, so it
can be re-derived rather than taken on trust:

```bash
python docs/noise_effect.py     # what a noise difference does on its own
python docs/floor_search.py     # where a shared amplitude floor fixes it
python docs/sigma_cutoff.py     # what a sigma exclusion costs and buys
```

Written 2026-09-11 against three cells. **Three cells is not a cohort** — the
thresholds derived below are provisional and should be re-derived on a real
dataset. The reasoning is what transfers; the specific cut points are not.

---

## The short version

1. **The cell is the unit of analysis**, not the event. One number per cell,
   n = cells.
2. **A noise difference alone produces a 30–38% frequency change and a ~65%
   amplitude change** on these recordings. That is larger than most real
   phenotypes, and it points in a self-consistent direction (fewer, bigger
   events) that reads as a plausible result.
3. **Fix it with one shared amplitude floor across every cell in both groups**,
   set from the noisiest cell — but the floor must clear that cell's *own*
   detection threshold, which is detector-specific and higher than 4σ.
4. **A σ exclusion criterion and the shared floor are one decision.** Excluding
   noisy cells lowers the floor the study needs, which raises how many real
   events survive. Pick the pair together.
5. **Declare one primary detector in advance.** The other four are a robustness
   claim, not five chances at significance.
6. **Calibrate once across both genotypes**, never per group.

---

## 1. The unit of analysis is the cell

These recordings yield 575–864 events each. Testing genotypes with n ≈ 2000
events is pseudoreplication: it asks whether *these events* differ, not whether
the genotypes do, and the p-value it returns is meaningless.

Collapse each cell to one value per measure, then compare with n = cells:

| measure | per-cell statistic | why |
|---|---|---|
| frequency | events passing the shared floor ÷ `duration_s` | rate is the point; use the recorded duration, not an assumed one |
| amplitude | **median** \|amplitude\| | the distribution is truncated at the detection limit and right-skewed; a mean is neither robust nor interpretable |
| rise time | median `rise_time_ms` | |
| decay | median `decay_time_ms` | |

If cells come from several animals the design is nested. Average to animal, or
fit a mixed model with animal as a random effect. Do not treat cells from one
animal as independent.

`duration_s` and `noise_sd` for every run are in `<stem>_multimethod_run.json`.

---

## 2. The noise confound, measured

### What was done

`docs/noise_effect.py` takes the quietest cell (0006, σ = 1.40 pA), adds
broadband Gaussian noise to the **raw** trace, then passes it through the
package's own preprocessing (3 kHz Bessel order 8 → 10 kHz decimation) so the
added noise is shaped exactly like real recording noise. Same cell, same
parameter set, built-in defaults. The only variable is σ.

The raw noise SD needed is several times the target, because the 3 kHz filter
discards most white-noise power — 22 pA of raw noise raises filtered σ from
1.40 to 3.73.

### Result

| added raw SD (pA) | σ (pA) | threshold Hz | threshold median \|amp\| | deconvolution Hz | deconvolution median \|amp\| |
|---|---|---|---|---|---|
| 0 | 1.40 | 10.03 | 16.3 | 7.20 | 25.0 |
| 4 | 1.54 | 9.49 | 17.5 | 6.86 | 26.5 |
| 8 | 1.90 | 8.46 | 20.6 | 6.03 | 30.8 |
| 12 | 2.37 | 7.51 | 24.4 | 5.12 | 35.6 |
| 15 | **2.76** | **6.92** | **26.6** | **4.46** | **41.1** |
| 18 | 3.17 | 6.52 | 28.8 | 4.06 | 43.0 |
| 22 | 3.73 | 6.11 | 30.9 | 3.58 | 47.5 |

At σ = 2.76 — matching the noisiest real cell (0010, σ = 2.78) — with nothing
else changed:

- **threshold: −31% frequency, +64% median amplitude**
- **deconvolution: −38% frequency, +65% median amplitude**

The dose–response is monotone across all seven levels, so this is not a single
unlucky noise draw.

### Why the direction matters

A noisier cell reports **fewer, larger** events. If one genotype is recorded
slightly worse, the artifact presents as "reduced sEPSC frequency with
increased amplitude" — internally consistent, mechanistically plausible, and
entirely false. This is the failure mode Greger & Watson (2025, *J Physiol*)
describe: they report a detection limit of 6.2 pA ≈ 3.9σ, with the modal CA1
mEPSC falling below it, and show a 4 pA amplitude increase reading out as
0.38 → 0.80 Hz with true amplitude flat.

### Limits of this evidence

- Noise was varied in isolation. A genuinely worse recording usually also has
  attenuated and slowed events from series resistance, so real-world bias can
  **exceed** these figures.
- Frequency and amplitude only. Rise and decay were not tested, so the
  noise-sensitivity of kinetics here is unknown.
- One cell, one detector pair.

---

## 3. The shared amplitude floor

Apply a single amplitude floor to every cell in the study, at analysis time.
This is legitimate and needs no re-detection, because `measure_at` quantifies
every event identically regardless of which detector found it — amplitudes are
comparable across cells and across methods by construction.

### The floor must clear the worst cell's own detection threshold

A post-hoc floor can only discard events a detector **already found**. It cannot
recover events the detector never located. So if a cell's own σ-relative
threshold sits above the shared floor, the floor is not binding for that cell
and its deficit survives.

`docs/floor_search.py`, σ 1.40 versus σ 2.76, spurious frequency error
after applying each floor. It draws its own noise independently of
`noise_effect.py`, so the no-floor row reads −30%/−36% here against
−31%/−38% in §2 — the same effect within ~2%:

| shared floor | threshold (5σ) | deconvolution (6σ) |
|---|---|---|
| none | −30% | −36% |
| 8 pA | −10% | −34% |
| 10 pA | **+0.5%** | −32% |
| 11.13 pA (4σ of σ_max) | +6% | **−28%** |
| 16.7 pA (6σ of σ_max) | +4% | −10% |
| 20 pA | +6% | **−2%** |
| 25 pA | +6% | +4% |

The rule:

> **floor ≥ (the detector's σ multiplier) × (the largest σ in the study),
> times a margin of ~1.2**

For deconvolution (`thresh_sd = 6.0`) at σ_max = 2.78 that is ≥ 16.7 pA, and
empirically ~20 pA to get inside 2%. For threshold (`amplitude_sd = 5.0`) it is
~10 pA. **The required floor is therefore detector-specific** — one more reason
to fix a single primary method up front.

A floor set too low is worse than none in one specific way: it bites only the
quiet cells. Redetecting these three with built-in defaults changed 0006 from
766 to 864 events but left 0010 and 0023 untouched, because the 11.13 pA floor
in `multimethod.config.json` was *below* their own 6σ thresholds of 16.7 and
11.8 pA. The correction was applied where it was least needed.

---

## 4. σ per cell as an exclusion criterion

Excluding cells above a σ cutoff is a recording-quality criterion, in the same
family as excluding on series resistance or holding current, and is standard
practice. σ is a property of the recording, so removing cells on it does not
select on phenotype — **provided** three conditions hold.

### Three conditions

1. **Pre-specify the cutoff** before looking at event data, ideally while still
   blind to genotype.
2. **Report exclusions per genotype, and test σ against genotype.** If the
   cutoff removes 1 control and 6 mutants, it is no longer selecting on
   recording quality but on something tracking the biology — you have kept only
   the healthiest mutants. No floor and no cutoff rescues that; it is a
   confound in the recordings, and must be reported as one.
3. **Show a sensitivity analysis** across a range of cutoffs. If the effect
   appears at σ < 2.0 and vanishes at σ < 2.5, there is no result.

### The cutoff and the floor are the same knob

`docs/sigma_cutoff.py`, deconvolution, built-in defaults, untruncated:

| cells kept | excluded | σ_max | floor needed | events before → after | retained | floor vs. mode |
|---|---|---|---|---|---|---|
| 3 | — | 2.78 | 20.0 pA | 2247 → 1064 | **47%** | above |
| 2 | 0010 | 1.96 | 14.1 pA | 1439 → 970 | **67%** | above |
| 1 | 0023, 0010 | 1.40 | 10.1 pA | 864 → 804 | **93%** | below |

Choose the retention you can afford and read the cutoff off the table, rather
than picking a σ number in the abstract.

### A criterion with meaning: resolve the mode

| cell | σ | its own 6σ threshold | modal \|amplitude\| | median \|amplitude\| | rate (Hz) |
|---|---|---|---|---|---|
| 0006 | 1.40 | 8.4 pA | 12.8 pA | 25.0 | 7.20 |
| 0023 | 1.96 | 11.8 pA | 11.2 pA | 14.9 | 4.79 |
| 0010 | 2.78 | 16.7 pA | 15.6 pA | 19.0 | 6.73 |

In 0023 and 0010 the modal amplitude sits **at** the detection threshold: those
cells never resolve the peak of their own amplitude distribution, only its
tail. 0006 does (mode 12.8 above threshold 8.4). That gives a criterion with a
rationale instead of an arbitrary number:

> **keep cells whose own detection threshold falls below the modal event
> amplitude** — cells that resolve the population rather than its tail.

On these data that is roughly **σ ≲ 1.5 pA** for deconvolution at 6σ, which
would exclude two of the three cells. Treat this as provisional: the mode is
itself estimated from a truncated distribution and is therefore biased upward,
and three cells cannot support a cut point.

### Prefer a covariate for borderline cells

A hard cutoff discards information and is arbitrary at the boundary. Including
σ as a fixed effect in a mixed model keeps every cell, adjusts explicitly, and
supports the stronger statement that the genotype effect survives adjustment.
The best presentation is both: the hard cutoff as the primary analysis because
it is simple to report, the covariate model as confirmation.

### Do not use σ alone

Pair it with series resistance and holding current — a cell can be quiet and
still badly clamped. And σ is filter-dependent, so it is comparable across
cells only when preprocessing is identical, which the pipeline enforces via the
output stem (`_filt3000Hz10000Hz_hum60Hz`). Mains hum inflates σ too, which is
one reason it is removed by default; a recording that needed it says how many
seconds were affected.

---

## 5. Which detector, and what the other four are for

Declare **one primary method before looking at the data**. Deconvolution is the
reasonable default: it estimates the event waveform from each recording, so it
has nothing to tune per cell.

Running five methods and reporting the one that reaches significance is
p-hacking. What the multi-method design actually buys is a robustness claim
most of this literature cannot make: *the genotype effect holds under five
independent detectors, including two machine-learning ones that cannot share a
template bias.* That is cheap here only because every method's events pass
through the same `measure_at`.

One split to respect:

- **Kinetics** — use unanimous consensus events. The bias toward clean,
  isolated, large events is acceptable when measuring shape.
- **Frequency** — do **not** use unanimous consensus. That bias can differ by
  genotype if one group has more overlapping events, which would manufacture a
  rate difference. Use the single primary method, or ≥2-method agreement.

This is the same asymmetry recorded in `calibrate.py`: pooling unanimous events
is right for measuring kinetics and wrong for anything touching the detection
limit.

---

## 6. Calibrate once, across both genotypes

Running `calibrate --write` separately per genotype bakes each group's own
kinetics into its own template and deconvolution kernel — it builds the
difference under test into the detector. Derive one parameter set from a
genotype-balanced pool and apply it to everything.

`calibrate` is also the right place to get the floor: it reports the noisiest
cell's σ across a folder. Multiply by the primary detector's σ multiplier
(not 4) and add the margin.

---

## 7. Blinding and review

Review while blind. Code the filenames, review, then unblind at analysis. The
review window writes `<stem>_multimethod_reviewed.csv`; reviewing a subset per
genotype also yields a **per-group false-positive rate**, which is worth
knowing precisely when noise differs between groups — a differing FP rate is by
itself a frequency difference.

---

## 8. Know which parameters were in effect

Every command prints the config file in effect and every value that differs
from the built-in defaults, and records the same under `config_source` in
`<stem>_multimethod_run.json`. Check it. A config written weeks earlier
silently changing an amplitude floor is exactly how the 766-versus-864
discrepancy in §3 arose. See [the README](../README.md#the-config-file-and-why-every-command-announces-it).

---

## 9. Suggested workflow

```bash
# 1. One run per cell, all methods, identical parameters.
#    For a whole folder, select the right protocol rather than the right
#    filenames -- and look before you choose the substring:
python -m multimethod batch FOLDER --list
python -m multimethod batch FOLDER --protocol gap_free
#    multimethod_batch_summary.csv is then the per-cell table steps 3-5 need,
#    keyed on the absolute file path, with noise_sd already in it.

# 2. Confirm nothing unexpected was in effect
#    (each run prints this, and stores it under config_source)

# 3. Collect sigma per cell from every _multimethod_run.json
#    -> test sigma against genotype BEFORE anything else
#    -> if it correlates with genotype, stop and report the confound

# 4. Pre-specify the sigma cutoff and the derived floor together
#    floor = 1.2 * primary_detector_sigma_multiplier * max(sigma of kept cells)

# 5. Per cell: apply the floor post hoc, then reduce to one value per measure
#    frequency = (events with |amplitude| >= floor) / duration_s
#    amplitude = median |amplitude| of those events

# 6. Compare with n = cells (nested on animal if applicable)

# 7. Robustness: repeat under each of the other four detectors
#    Sensitivity: repeat across a range of sigma cutoffs
```

---

## 10. Not yet implemented

The package is strictly per-recording — there is no concept of cell, animal or
genotype anywhere in it, so steps 3–7 above are currently manual. A `compare`
command should encode them rather than leave them to discipline:

- a manifest mapping recording → genotype / cell / animal
- σ per cell tabulated, **with an explicit σ-versus-genotype test that refuses
  to be quiet about a confound**
- the floor computed per detector from σ_max and stated, not assumed
- exclusions listed per genotype, with the retention cost shown
- a cell-level table as the analysis unit
- box-and-whisker with individual cell points, and amplitude CDFs, by genotype
- the same effect under all five detectors, side by side
- a sensitivity panel across σ cutoffs

---

## References

- Greger & Watson (2025), *J Physiol* — detection limit 6.2 pA ≈ 3.9σ; modal
  CA1 mEPSC below it; amplitude changes misread as frequency changes.
- Pernía-Andrade et al. (2012), *Biophys J* — deconvolution-based detection.
- Clements & Bekkers (1997), *Biophys J* — optimally scaled template.
- O'Neill et al. (2025), *eLife* — miniML.
- Pircher et al. (2022), *PLOS ONE* 17(9): e0273501 — unsupervised state
  clustering. [doi:10.1371/journal.pone.0273501](https://doi.org/10.1371/journal.pone.0273501)
