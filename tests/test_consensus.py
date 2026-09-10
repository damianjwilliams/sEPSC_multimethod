"""Tests for cross-method event matching.

This is the package's own logic rather than a wrapper around a detector, and
the two rules that are easy to get wrong are both tested explicitly: clusters
must not chain, and one method must not appear twice in one cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from multimethod import consensus
from multimethod.events import Event


def ev(t: float, amp: float = -20.0, method: str = "") -> Event:
    """An Event at time t. Only peak_time_s and amplitude matter here."""
    return Event(peak_idx=int(t * 10_000), peak_time_s=t, baseline=0.0,
                 amplitude=amp, rise_time_ms=1.0, decay_time_ms=3.0,
                 area=-50.0, method=method)


DT = 1e-4


# ------------------------------------------------------------- basic matching

def test_two_methods_on_the_same_event_make_one_consensus_event():
    got = consensus.build({"a": [ev(1.0)], "b": [ev(1.0005)]}, DT, tolerance_ms=3.0)
    assert len(got) == 1
    assert sorted(got[0].methods) == ["a", "b"]
    assert got[0].n_methods == 2


def test_detections_further_apart_than_the_tolerance_stay_separate():
    got = consensus.build({"a": [ev(1.0)], "b": [ev(1.010)]}, DT, tolerance_ms=3.0)
    assert len(got) == 2
    assert all(c.n_methods == 1 for c in got)


def test_a_method_that_finds_nothing_still_counts_as_having_run():
    got = consensus.build({"a": [ev(1.0)], "b": []}, DT)
    assert len(got) == 1
    assert got[0].methods == ["a"]
    df = consensus.frame(got, ["a", "b"])
    assert df["found_b"].tolist() == [0]      # a real zero, not a missing column


def test_empty_input_gives_an_empty_frame_with_every_column():
    df = consensus.frame([], ["a", "b"])
    assert df.empty
    for col in ("event_id", "found_a", "amp_a", "found_b", "amp_b",
                "amplitude_median", "amplitude_spread"):
        assert col in df.columns


# ------------------------------------------------ the two rules that matter

def test_clusters_do_not_chain():
    """Single-linkage would merge 0, 2.5, 5, 7.5 ms into one 7.5 ms 'event'.

    Each detection must be within tolerance of the cluster's MEAN, which
    bounds how wide a cluster can get.
    """
    per_method = {f"m{i}": [ev(t)] for i, t in
                  enumerate([1.0000, 1.0025, 1.0050, 1.0075])}
    got = consensus.build(per_method, DT, tolerance_ms=3.0)
    assert len(got) > 1, "a 3 ms tolerance must not absorb a 7.5 ms span"
    spans = [max(c.per_method_time_s.values()) - min(c.per_method_time_s.values())
             for c in got]
    assert max(spans) <= 2 * 3.0 / 1000.0


def test_one_method_cannot_occupy_a_cluster_twice():
    """A method that fragments one event into three must not look like three
    methods agreeing."""
    got = consensus.build({"a": [ev(1.0), ev(1.001), ev(1.002)]}, DT,
                          tolerance_ms=3.0)
    assert len(got) == 3, "each of a method's own detections needs its own cluster"
    assert all(c.methods == ["a"] for c in got)
    assert all(c.n_methods == 1 for c in got)


def test_the_nearer_of_two_detections_joins_the_cluster():
    got = consensus.build({"a": [ev(1.0)], "b": [ev(1.0002), ev(1.0028)]}, DT,
                          tolerance_ms=3.0)
    joined = [c for c in got if c.n_methods == 2]
    assert len(joined) == 1
    assert abs(joined[0].per_method_time_s["b"] - 1.0002) < 1e-9


# ---------------------------------------------------------------- the table

def test_frame_columns_are_ordered_and_typed():
    got = consensus.build({"a": [ev(1.0, -10.0)], "b": [ev(1.0005, -14.0)]}, DT)
    df = consensus.frame(got, ["a", "b"])
    assert list(df.columns) == [
        "event_id", "peak_time_s", "peak_idx", "n_methods", "methods",
        "found_a", "amp_a", "found_b", "amp_b",
        "amplitude_median", "amplitude_spread"]
    assert df["peak_idx"].dtype == "int64"
    assert df["found_a"].dtype == "int64"
    assert df.at[0, "methods"] == "a;b"


def test_amplitude_median_and_spread_use_only_the_methods_that_found_it():
    got = consensus.build({"a": [ev(1.0, -10.0)], "b": [ev(1.0005, -20.0)],
                           "c": []}, DT)
    df = consensus.frame(got, ["a", "b", "c"])
    assert df.at[0, "amplitude_median"] == pytest.approx(-15.0)
    assert df.at[0, "amplitude_spread"] == pytest.approx(10.0)
    assert np.isnan(df.at[0, "amp_c"])


def test_spread_is_zero_for_a_single_method_event():
    df = consensus.frame(consensus.build({"a": [ev(1.0, -9.0)]}, DT), ["a"])
    assert df.at[0, "amplitude_spread"] == 0.0


# -------------------------------------------------------------- the summaries

def test_summarise_counts_events_unique_to_each_method():
    per = {"a": [ev(1.0), ev(2.0), ev(3.0)],
           "b": [ev(1.0005), ev(5.0)]}
    got = consensus.build(per, DT)
    summ = consensus.summarise(got, ["a", "b"]).set_index("method")
    assert summ.at["a", "n_events"] == 3
    assert summ.at["a", "unique_to_it"] == 2        # 2.0 and 3.0
    assert summ.at["b", "n_events"] == 2
    assert summ.at["b", "unique_to_it"] == 1        # 5.0


def test_agreement_matrix_diagonal_is_the_event_count():
    per = {"a": [ev(1.0), ev(2.0)], "b": [ev(1.0005)]}
    got = consensus.build(per, DT)
    m = consensus.agreement_matrix(got, ["a", "b"])
    assert m.loc["a", "a"] == 2.0
    assert m.loc["b", "b"] == 1.0
    assert m.loc["a", "b"] == pytest.approx(1 / 2)   # 1 shared of 2 total
    assert m.loc["a", "b"] == m.loc["b", "a"]


def test_by_agreement_partitions_every_event_exactly_once():
    per = {"a": [ev(1.0), ev(2.0), ev(3.0)],
           "b": [ev(1.0005), ev(2.0005)],
           "c": [ev(1.0008)]}
    got = consensus.build(per, DT)
    by = consensus.by_agreement(got, 3)
    assert int(by["n_events"].sum()) == len(got)
    assert by["fraction"].sum() == pytest.approx(1.0)


def test_tolerance_changes_the_answer_and_that_is_the_point():
    per = {"a": [ev(1.0)], "b": [ev(1.004)]}
    tight = consensus.build(per, DT, tolerance_ms=2.0)
    loose = consensus.build(per, DT, tolerance_ms=8.0)
    assert len(tight) == 2 and len(loose) == 1
