"""`python -m multimethod <command>` -- one dispatcher, four commands.

Manual dispatch rather than argparse sub-parsers, so each command keeps its
own parser and `python -m multimethod run --help` shows run's real options
instead of a flattened summary of everything.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from . import detectors

COMMANDS = {
    "run": "Run every enabled detector and write the consensus table",
    "review": "Open the review window: which method found which event",
    "launch": "Parameter window first, then run, then review",
    "benchmark": "Score the in-repo detectors against planted events",
    "methods": "List the detectors and whether each one can run here",
    "calibrate": "Set the defaults from your own recordings instead of a fixture",
}


def _usage() -> str:
    lines = [
        "usage: python -m multimethod <command> [args...]", "",
        textwrap.fill(
            "Run several synaptic-event detectors on one recording, measure "
            "every event the same way, and review which method found which "
            "event.", 78),
        "",
    ]
    width = max(len(c) for c in COMMANDS)
    for name, help_text in COMMANDS.items():
        lines.append(f"    {name.ljust(width)}   {help_text}")
    lines += ["", "Each command has its own --help.",
              "The same commands exist as scripts in the repo root:",
              "    run_all.py    launch.py    review_methods.py"]
    return "\n".join(lines)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("abf", nargs="?", default="", help="path to a gap-free .abf")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--config", default=None,
                   help="settings JSON (default: multimethod.config.json "
                        "beside the package, if it exists)")


def _load_settings(args):
    from .settings import Settings

    s = Settings.load(args.config)
    if getattr(args, "no_filter", False):
        s.preprocessing = type(s.preprocessing)(**{**s.preprocessing.as_dict(),
                                                   "enabled": False})
    if getattr(args, "tolerance_ms", None):
        s.tolerance_ms = float(args.tolerance_ms)
    if getattr(args, "methods", None):
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in detectors.REGISTRY]
        if unknown:
            raise SystemExit(f"unknown method(s): {unknown}; "
                             f"known: {detectors.names()}")
        for name in detectors.names():
            s.enabled[name] = name in wanted
    return s


def cmd_run(argv) -> int:
    from .runner import run_all

    p = argparse.ArgumentParser(prog="python -m multimethod run",
                                description=COMMANDS["run"])
    _add_common(p)
    p.add_argument("--methods", default=None,
                   help="comma-separated subset, e.g. threshold,deconvolution")
    p.add_argument("--no-filter", action="store_true",
                   help="skip the Bessel low-pass + resample")
    p.add_argument("--tolerance-ms", type=float, default=None,
                   help="how close two methods' peaks must be to agree")
    p.add_argument("--gui", action="store_true",
                   help="open the parameter window before running")
    p.add_argument("--review", action="store_true",
                   help="open the review window when the run finishes")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="skip the analysis figures (they are written by default)")
    p.set_defaults(plots=True)
    args = p.parse_args(argv)

    s = _load_settings(args)
    abf, channel = args.abf, args.channel
    if args.gui or not abf:
        from .paramwindow import edit_settings
        picked = edit_settings(s, abf, channel, args.config)
        if picked is None:
            print("Cancelled — nothing run.")
            return 0
        s, abf, channel = picked
    if not abf:
        p.error("an .abf path is required (or use --gui)")
    if not os.path.isfile(abf):
        p.error(f"no such file: {abf}")

    print(f"Running on {os.path.basename(abf)} — {s.preprocessing.describe()}")
    outcome = run_all(abf, channel, s, progress=lambda r: print(r.status_line(),
                                                               flush=True))
    # The per-method lines were already streamed by `progress`; print only the
    # tail of the report so they do not appear twice.
    print()
    print(f"noise sigma = {outcome.noise_sd:.2f} {outcome.trace.unit}  "
          f"(4 sigma detection limit = {4 * outcome.noise_sd:.2f})")
    print(f"{len(outcome.ran)} of {len(outcome.results)} method(s) ran "
          f"-> {len(outcome.consensus)} consensus events")
    for w in outcome.warnings:
        print(f"NOTE: {w}")
    written = outcome.write()
    if args.plots and outcome.ran:
        from .plots import write_all
        written += write_all(outcome)
    print()
    for path in written:
        print(f"wrote {os.path.basename(path)}")

    if outcome.ran:
        print()
        print(_agreement_report(outcome))
    if args.review:
        from .review import open_review
        open_review(abf, channel, s)
    elif outcome.ran:
        print()
        print(f"review with: python review_methods.py {abf}"
              f"{'' if s.preprocessing.enabled else ' --no-filter'}")
    return 0


def _agreement_report(outcome) -> str:
    from . import consensus as c

    methods = outcome.ran
    lines = ["Agreement:"]
    by = c.by_agreement(outcome.consensus, len(methods))
    for _, row in by.iterrows():
        n = int(row["n_methods"])
        lines.append(f"  found by {n} of {len(methods)}: "
                     f"{int(row['n_events']):6d} events "
                     f"({row['fraction'] * 100:5.1f}%)")
    lines.append("")
    lines.append("Per method:")
    summ = c.summarise(outcome.consensus, methods)
    for _, row in summ.iterrows():
        lines.append(f"  {row['method']:<14} {int(row['n_events']):6d} events, "
                     f"{int(row['unique_to_it']):5d} found by it alone, "
                     f"median amplitude {row['median_amplitude']:.1f}")
    return "\n".join(lines)


def cmd_review(argv) -> int:
    p = argparse.ArgumentParser(prog="python -m multimethod review",
                                description=COMMANDS["review"])
    _add_common(p)
    p.add_argument("--no-filter", action="store_true",
                   help="the run being reviewed was made without filtering")
    p.add_argument("--only-unique", action="store_true",
                   help="start filtered to events only one method found")
    args = p.parse_args(argv)
    if not args.abf:
        p.error("an .abf path is required")

    from .review import open_review
    open_review(args.abf, args.channel, _load_settings(args), args.only_unique)
    return 0


def cmd_launch(argv) -> int:
    """Parameter window -> run -> review, in one go."""
    p = argparse.ArgumentParser(prog="python -m multimethod launch",
                                description=COMMANDS["launch"])
    _add_common(p)
    args = p.parse_args(argv)
    return cmd_run(([args.abf] if args.abf else [])
                   + ["--channel", str(args.channel), "--gui", "--review"]
                   + (["--config", args.config] if args.config else []))


def cmd_benchmark(argv) -> int:
    from .benchmark import run_benchmark

    p = argparse.ArgumentParser(prog="python -m multimethod benchmark",
                                description=COMMANDS["benchmark"])
    p.add_argument("--methods", default=None,
                   help="comma-separated subset (default: the in-repo ones)")
    p.add_argument("--duration-s", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    methods = ([m.strip() for m in args.methods.split(",")] if args.methods
               else None)
    run_benchmark(methods=methods, duration_s=args.duration_s, seed=args.seed)
    return 0


def cmd_methods(argv) -> int:
    p = argparse.ArgumentParser(prog="python -m multimethod methods",
                                description=COMMANDS["methods"])
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    from .settings import Settings
    s = Settings.load(args.config)
    print(f"{'method':<14} {'kind':<9} {'status':<9} description")
    print("-" * 92)
    for name in detectors.names():
        det = detectors.get(name)
        ok, why = det.available(s.params_for(name))
        kind = "external" if det.external else "in-repo"
        status = "ready" if ok else "unavailable"
        print(f"{name:<14} {kind:<9} {status:<9} {det.blurb}")
        if not ok:
            print(f"{'':<34} -> {why}")
    return 0


def cmd_calibrate(argv) -> int:
    from .calibrate import apply, collect, report

    p = argparse.ArgumentParser(prog="python -m multimethod calibrate",
                                description=COMMANDS["calibrate"])
    p.add_argument("folder", help="folder of .abf recordings")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--min-methods", type=int, default=None,
                   help="how many methods must agree for an event to be pooled "
                        "(default: all that ran)")
    p.add_argument("--rerun", action="store_true",
                   help="re-run detection instead of reading existing results")
    p.add_argument("--write", action="store_true",
                   help="save the proposed values as the new defaults")
    args = p.parse_args(argv)
    if not os.path.isdir(args.folder):
        p.error(f"not a folder: {args.folder}")

    from .settings import Settings
    s = Settings.load(args.config)
    cal = collect(args.folder, s, args.channel, args.min_methods, args.rerun,
                  progress=lambda m: print(m, flush=True))
    print()
    print(report(cal, s))
    if args.write and cal.proposed:
        path = apply(cal, s).save(args.config)
        print()
        print(f"wrote {path}")
    elif cal.proposed:
        print()
        print("re-run with --write to save these as the defaults")
    return 0


HANDLERS = {"run": cmd_run, "review": cmd_review, "launch": cmd_launch,
            "benchmark": cmd_benchmark, "methods": cmd_methods,
            "calibrate": cmd_calibrate}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in HANDLERS:
        print(f"unknown command {cmd!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    return HANDLERS[cmd](rest)
