"""Tests for running a folder, selected by protocol.

The risks here are not arithmetic. They are:

  - selecting the wrong recordings, silently (a substring that spans two
    protocols, or a multi-sweep protocol analysed as though it were gap-free);
  - losing a whole batch to one bad file;
  - producing a table that cannot be joined to anything, because it identifies
    recordings by a filename that is not unique across a folder tree.

So most of these check reporting and resilience rather than numbers. The ABF
reading itself is stubbed: `survey` is the only part that touches pyabf, and
what matters downstream is what it returns, not that pyabf works.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from multimethod import batch
from multimethod.settings import Settings


def _rec(name, protocol="DW_Gap_free_-70_mV", sweeps=1, channels=1, error=""):
    return batch.Recording(path=os.path.join(os.sep, "data", name),
                           protocol=protocol, sweeps=sweeps, channels=channels,
                           rate_hz=50_000.0, duration_s=120.0, error=error)


# ------------------------------------------------------------- the selection

def test_matching_is_a_case_insensitive_substring():
    recs = [_rec("a.abf", "DW_Gap_free_-70_mV"), _rec("b.abf", "DW_IC_IV")]
    assert [r.name for r in batch.match(recs, "gap")] == ["a.abf"]
    assert [r.name for r in batch.match(recs, "GAP_FREE")] == ["a.abf"]
    assert [r.name for r in batch.match(recs, "  gap  ")] == ["a.abf"], (
        "a pasted protocol name often carries whitespace")


def test_an_empty_pattern_means_every_recording():
    recs = [_rec("a.abf", "X"), _rec("b.abf", "Y")]
    assert len(batch.match(recs, "")) == 2
    assert len(batch.match(recs, None)) == 2


def test_a_substring_spanning_two_protocols_is_flagged():
    """`IC_IV` matches both `IC_IV` and `DW_IC_IV` in real data. Running them
    together silently would pool two different experiments."""
    recs = [_rec("a.abf", "IC_IV"), _rec("b.abf", "DW_IC_IV"),
            _rec("c.abf", "DW_Gap_free_-70_mV")]
    text = batch.describe_selection(recs, batch.match(recs, "ic_iv"), "ic_iv")
    assert "more than one protocol" in text
    assert "IC_IV" in text and "DW_IC_IV" in text


def test_a_single_protocol_match_is_not_flagged():
    recs = [_rec("a.abf"), _rec("b.abf")]
    text = batch.describe_selection(recs, batch.match(recs, "gap"), "gap")
    assert "more than one protocol" not in text
    assert "matched 2 of 2" in text


def test_protocols_are_counted_most_common_first():
    recs = [_rec("a.abf", "B"), _rec("b.abf", "A"), _rec("c.abf", "A")]
    assert list(batch.protocols(recs)) == ["A", "B"]


def test_a_file_with_no_protocol_is_still_listed():
    """Dropping it would make the counts lie about the folder."""
    counts = batch.protocols([_rec("a.abf", "")])
    assert counts == {"<no protocol in header>": 1}


def test_an_unreadable_file_is_listed_separately():
    counts = batch.protocols([_rec("a.abf", "", error="ValueError: truncated")])
    assert counts == {"<unreadable>": 1}


# ----------------------------------------------------------- what to refuse

def test_a_multi_sweep_recording_is_refused_with_a_reason():
    """The package reads sweep 0. Analysing the first sweep of a 50-sweep IV
    protocol is a wrong answer, which is worse than a missing one."""
    why = _rec("iv.abf", "DW_IC_IV", sweeps=50).usable
    assert why
    assert "50 sweeps" in why and "gap-free" in why


def test_a_gap_free_recording_is_accepted():
    assert _rec("a.abf").usable == ""


def test_a_header_error_makes_the_recording_unusable_but_visible():
    r = _rec("bad.abf", error="ValueError: truncated")
    assert "truncated" in r.usable
    assert r in batch.match([r], "")


def test_refused_recordings_are_named_in_the_selection_report():
    recs = [_rec("ok.abf"), _rec("iv.abf", sweeps=20)]
    text = batch.describe_selection(recs, recs, "")
    assert "Skipping 1" in text
    assert "iv.abf" in text


# -------------------------------------------------------------- the table

def test_the_first_column_is_the_absolute_path():
    """Everything downstream joins on it, and a bare filename is not unique
    across a folder tree."""
    df = batch.frame([batch.summary_row(_rec("a.abf"), None, Settings(), "why")])
    assert list(df.columns)[0] == "file_path"
    assert os.path.isabs(df.at[0, "file_path"])


def test_a_skipped_recording_still_gets_a_row_with_its_reason():
    row = batch.summary_row(_rec("iv.abf", sweeps=20), None, Settings(),
                            "20 sweeps")
    assert row["status"] == "skipped"
    assert row["message"] == "20 sweeps"
    assert row["file_path"].endswith("iv.abf")


def test_the_table_survives_rows_with_different_columns():
    """A skipped row has no per-method columns; a successful one does. The
    frame must not lose either."""
    rows = [batch.summary_row(_rec("skip.abf"), None, Settings(), "no"),
            {"file_path": "/data/ok.abf", "file_name": "ok.abf",
             "protocol": "X", "status": "ok", "noise_sd": 1.4,
             "n_threshold": 10}]
    df = batch.frame(rows)
    assert len(df) == 2
    assert "n_threshold" in df.columns
    assert pd.isna(df.at[0, "n_threshold"])


def test_an_empty_selection_produces_an_empty_frame_not_a_crash():
    assert batch.frame([]).empty


# ------------------------------------------------- resilience of the run

def test_one_bad_recording_does_not_end_the_batch(tmp_path, monkeypatch):
    """The whole point of a batch: file 3 of 40 being unreadable must cost one
    row, not the other thirty-nine."""
    recs = [_rec("a.abf"), _rec("boom.abf"), _rec("c.abf")]
    monkeypatch.setattr(batch, "survey", lambda *a, **k: recs)

    calls = []

    def fake_run_all(path, channel, s, methods=None, config_source=None):
        calls.append(path)
        if "boom" in path:
            raise RuntimeError("unreadable sweep")
        from test_runner_and_settings import synth_trace
        from multimethod import runner
        trace, _ = synth_trace(duration_s=6.0, seed=1)
        monkeypatch.setattr(runner, "load", lambda *a, **k: trace)
        out = runner.run_all(str(tmp_path / os.path.basename(path)), 0, s,
                             methods=["threshold"])
        return out

    monkeypatch.setattr(batch, "run_all", fake_run_all)
    df = batch.run_folder(str(tmp_path), "", Settings(),
                          out_path=str(tmp_path / "sum.csv"))

    assert len(calls) == 3, "the batch must keep going past the failure"
    assert len(df) == 3
    assert list(df["status"]) == ["ok", "skipped", "ok"]
    assert "unreadable sweep" in df.at[1, "message"]


def test_the_summary_is_written_as_it_goes(tmp_path, monkeypatch):
    """An interrupted batch, or one whose output drive misbehaves, should keep
    the rows it already produced."""
    seen_lengths = []
    recs = [_rec("a.abf"), _rec("b.abf")]
    monkeypatch.setattr(batch, "survey", lambda *a, **k: recs)

    out = tmp_path / "sum.csv"

    def fake_run_all(path, channel, s, methods=None, config_source=None):
        if out.exists():
            seen_lengths.append(len(pd.read_csv(out)))
        raise RuntimeError("stop")

    monkeypatch.setattr(batch, "run_all", fake_run_all)
    batch.run_folder(str(tmp_path), "", Settings(), out_path=str(out))

    assert seen_lengths == [1], (
        "the second recording should have seen the first one's row on disk")
    assert len(pd.read_csv(out)) == 2


# ------------------------------------------------------------ the warning

def test_a_wide_noise_spread_is_called_out():
    # The real sigmas of 0006 and 0010, unrounded: the ratio is reported from
    # the stored values, so rounding them first would change the last digit.
    df = pd.DataFrame({"noise_sd": [1.4047, 2.7826]})
    text = batch.noise_report(df)
    assert "1.98x" in text
    assert "comparing-groups" in text, "point at the reasoning, not just a number"


def test_a_narrow_noise_spread_gets_no_warning():
    text = batch.noise_report(pd.DataFrame({"noise_sd": [1.40, 1.50]}))
    assert "1.07x" in text
    assert "shared amplitude floor" not in text


def test_noise_report_handles_a_table_with_no_successful_runs():
    assert batch.noise_report(pd.DataFrame({"status": ["skipped"]})) == ""
    assert batch.noise_report(pd.DataFrame()) == ""


# ----------------------------------------------------------------- the CLI

def test_batch_is_registered_and_offers_the_protocol_flag():
    import contextlib
    import io as _io

    from multimethod import cli

    assert "batch" in cli.HANDLERS
    buf = _io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        cli.HANDLERS["batch"](["--help"])
    text = buf.getvalue()
    for flag in ("--protocol", "--list", "--dry-run", "--recursive", "--out"):
        assert flag in text, f"batch is missing {flag}"


def test_batch_rejects_a_path_that_is_not_a_folder(tmp_path, capsys):
    from multimethod import cli

    f = tmp_path / "a.abf"
    f.write_bytes(b"x")
    with pytest.raises(SystemExit):
        cli.HANDLERS["batch"]([str(f)])
