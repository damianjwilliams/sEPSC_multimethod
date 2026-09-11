"""Run a whole folder, selecting recordings by their acquisition protocol.

A day's folder holds every protocol you ran -- gap-free sEPSC sweeps alongside
steps, IV curves, seal tests. Only some of them are the ones you want to detect
events in, and the recording itself knows which it is: pCLAMP writes the
protocol name into the ABF header. So the selector here is the protocol, matched
by a case-insensitive substring, rather than a filename convention that has to
be maintained by hand.

Two deliberate choices:

**The survey is cheap and separate from the run.** `survey` reads headers only
(`loadData=False`), so `--list` over a large folder is near-instant and you can
see what protocols exist before committing to an analysis. Picking a substring
blind is how you end up analysing seal tests.

**One recording's failure never ends the batch.** A folder will contain a
truncated file, a multi-sweep protocol, a recording on a channel that does not
exist. Each becomes a row with a reason, exactly as a failed detector becomes a
DetectorResult rather than an exception, and the summary is rewritten after
every recording so a long batch interrupted at file 40 still has 39 rows.

The summary's first column is the absolute file path. Everything downstream --
the per-cell table a group comparison needs, a manifest mapping recording to
genotype -- keys on it, and a bare filename is not unique across a folder tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from .runner import RunOutcome, run_all
from .settings import Settings

#: Written into the folder that was analysed.
BATCH_SUMMARY_NAME = "multimethod_batch_summary.csv"

#: The package reads sweep 0 of a gap-free recording. A protocol with many
#: sweeps is a different kind of experiment, and analysing its first sweep
#: silently would be a wrong answer rather than a missing one.
MAX_SWEEPS = 1


@dataclass(frozen=True)
class Recording:
    """One .abf as its header describes it, before any data is read."""

    path: str
    """Absolute."""

    protocol: str = ""
    sweeps: int = 0
    channels: int = 0
    rate_hz: float = 0.0
    duration_s: float = 0.0
    error: str = ""
    """Non-empty if the header could not be read. Such a file is reported, not
    dropped: a truncated recording you meant to analyse must not vanish."""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def usable(self) -> str:
        """Empty if this can be run, otherwise why not."""
        if self.error:
            return self.error
        if self.sweeps > MAX_SWEEPS:
            return (f"{self.sweeps} sweeps; this analyses gap-free recordings "
                    f"and would only see the first")
        if self.channels < 1:
            return "no ADC channels"
        return ""


def survey(folder: str, recursive: bool = False) -> list[Recording]:
    """Every .abf under `folder`, read from headers only.

    Deliberately tolerant: a file that cannot be parsed becomes a Recording
    carrying the error, so `--list` shows the whole folder rather than dying on
    the first bad file.
    """
    import pyabf

    paths = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            paths += [os.path.join(root, f) for f in files
                      if f.lower().endswith(".abf")]
    else:
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(".abf")]

    out = []
    for p in sorted(paths):
        p = os.path.abspath(p)
        try:
            abf = pyabf.ABF(p, loadData=False)
            out.append(Recording(path=p, protocol=str(abf.protocol or ""),
                                 sweeps=int(abf.sweepCount),
                                 channels=int(abf.channelCount),
                                 rate_hz=float(abf.dataRate),
                                 duration_s=float(abf.dataLengthSec)))
        except Exception as exc:
            out.append(Recording(path=p, error=f"{type(exc).__name__}: {exc}"))
    return out


def protocols(recs: list[Recording]) -> dict[str, int]:
    """protocol name -> how many recordings use it, most common first."""
    counts: dict[str, int] = {}
    for r in recs:
        key = r.protocol if r.protocol else ("<unreadable>" if r.error
                                             else "<no protocol in header>")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def match(recs: list[Recording], pattern: str) -> list[Recording]:
    """Recordings whose protocol contains `pattern`, case-insensitively.

    Substring rather than glob or regex: the point is that you can type part of
    a protocol name you half-remember. An empty pattern matches everything,
    which is what `--protocol ""` means -- analyse the folder regardless.
    """
    needle = (pattern or "").strip().lower()
    if not needle:
        return list(recs)
    return [r for r in recs if needle in (r.protocol or "").lower()]


def describe_selection(recs: list[Recording], selected: list[Recording],
                       pattern: str) -> str:
    """What the pattern picked, and what it passed over."""
    lines = []
    distinct = sorted({r.protocol for r in selected if r.protocol})
    if pattern:
        lines.append(f'Protocol filter "{pattern}" matched {len(selected)} of '
                     f'{len(recs)} recording(s) across {len(distinct)} '
                     f'protocol(s):')
    else:
        lines.append(f"No protocol filter: all {len(selected)} recording(s).")
    for name in distinct:
        n = sum(1 for r in selected if r.protocol == name)
        lines.append(f"    {name}  ({n})")
    if len(distinct) > 1:
        lines.append("  NOTE: that substring spans more than one protocol. If "
                     "they are different")
        lines.append("  experiments, narrow it -- their events are not "
                     "comparable.")

    skipped = [r for r in selected if r.usable]
    if skipped:
        lines.append("")
        lines.append(f"Skipping {len(skipped)}:")
        for r in skipped:
            lines.append(f"    {r.name}: {r.usable}")
    return "\n".join(lines)


def summary_row(rec: Recording, outcome: RunOutcome | None,
                settings: Settings, error: str = "") -> dict:
    """One row of the batch table. Absolute path first, always."""
    row = {
        "file_path": rec.path,
        "file_name": rec.name,
        "protocol": rec.protocol,
        "channel": 0,
        "duration_s": round(rec.duration_s, 3),
        "acquired_hz": rec.rate_hz,
        "status": "ok" if outcome is not None else "skipped",
        "message": error,
    }
    if outcome is None:
        return row

    row["channel"] = outcome.trace.channel
    row["duration_s"] = round(outcome.trace.duration_s, 3)
    row["analysed_hz"] = outcome.trace.fs
    row["unit"] = outcome.trace.unit
    # sigma per cell: the single most important column for comparing
    # recordings, because a detection threshold is relative to it.
    row["noise_sd"] = round(outcome.noise_sd, 4)
    row["detection_limit_4sd"] = round(4 * outcome.noise_sd, 4)

    dur = outcome.trace.duration_s or 1.0
    for method, res in outcome.results.items():
        row[f"n_{method}"] = res.n_events if res.ok else None
        row[f"rate_{method}_hz"] = round(res.n_events / dur, 4) if res.ok else None

    n_cons = len(outcome.consensus)
    ran = outcome.ran
    row["n_consensus"] = n_cons
    row["rate_consensus_hz"] = round(n_cons / dur, 4)
    row["n_unanimous"] = sum(1 for ce in outcome.consensus
                             if ce.n_methods == len(ran)) if ran else 0
    row["methods_ran"] = ";".join(ran)
    row["methods_failed"] = ";".join(m for m, r in outcome.results.items()
                                     if not r.ok)
    row["stem"] = outcome.stem
    return row


def frame(rows: list[dict]) -> pd.DataFrame:
    """The rows as a table, with the path columns pinned to the front."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    lead = [c for c in ("file_path", "file_name", "protocol", "channel",
                        "duration_s", "noise_sd", "detection_limit_4sd")
            if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


def run_folder(folder: str, pattern: str = "", settings: Settings | None = None,
               channel: int = 0, methods: list | None = None,
               recursive: bool = False, plots: bool = False,
               out_path: str | None = None, config_source=None,
               progress=None) -> pd.DataFrame:
    """Analyse every recording whose protocol matches, and tabulate them.

    The summary is rewritten to disk after each recording, so an interrupted
    batch -- or one whose output drive misbehaves -- keeps what it already did.
    """
    s = settings or Settings()
    recs = match(survey(folder, recursive), pattern)
    out_path = out_path or os.path.join(folder, BATCH_SUMMARY_NAME)
    rows: list[dict] = []

    for i, rec in enumerate(recs, 1):
        why = rec.usable
        if why:
            rows.append(summary_row(rec, None, s, why))
            frame(rows).to_csv(out_path, index=False)
            continue
        if progress:
            progress(f"[{i}/{len(recs)}] {rec.name}")
        try:
            outcome = run_all(rec.path, channel, s, methods=methods,
                              config_source=config_source)
            outcome.write()
            if plots and outcome.ran:
                from .plots import write_all
                write_all(outcome)
            rows.append(summary_row(rec, outcome, s))
            if progress:
                progress(f"      sigma {outcome.noise_sd:.2f}, "
                         f"{len(outcome.consensus)} consensus events, "
                         f"{len(outcome.ran)}/{len(outcome.results)} methods")
        except Exception as exc:
            # One unreadable recording must not cost the other thirty-nine.
            rows.append(summary_row(rec, None, s, f"{type(exc).__name__}: {exc}"))
            if progress:
                progress(f"      FAILED {type(exc).__name__}: {exc}")
        frame(rows).to_csv(out_path, index=False)

    return frame(rows)


def noise_report(df: pd.DataFrame) -> str:
    """The spread of sigma across the batch, and what it implies.

    Printed because it is the first thing that decides whether recordings can
    be compared at all: a detection threshold is a multiple of sigma, so a
    two-fold spread means the quiet cells resolve events the noisy ones never
    see. See docs/comparing-groups.md.
    """
    if df.empty or "noise_sd" not in df.columns:
        return ""
    sd = df["noise_sd"].dropna()
    if sd.empty:
        return ""
    lo, hi = float(sd.min()), float(sd.max())
    spread = f" ({hi / lo:.2f}x)" if lo else ""
    lines = [f"noise sigma across {len(sd)} recording(s): "
             f"{lo:.2f} to {hi:.2f}{spread}"]
    if lo and hi / lo > 1.5:
        lines.append(f"  A shared amplitude floor for this set has to clear "
                     f"the noisiest cell's own")
        lines.append(f"  threshold, not 4 sigma of it. See "
                     f"docs/comparing-groups.md section 3.")
    return "\n".join(lines)
