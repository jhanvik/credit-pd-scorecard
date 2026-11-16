"""Tests for the fairness metrics, with answers worked out by hand."""
import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard.fairness import MIN_GROUP, age_group, bootstrap_gap, gap_table, group_metrics

# group A: 4 customers (2 defaulters, 2 not), flagged = [defaulter1, defaulter2, non1, none]
Y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
FLAG = np.array([True, True, True, False, True, False, False, False])
P = np.array([0.9, 0.8, 0.6, 0.1, 0.7, 0.2, 0.2, 0.1])
G = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])


def test_group_metrics_hand_calculation():
    t = group_metrics(Y, P, FLAG, G).set_index("group")
    a, b = t.loc["A"], t.loc["B"]
    assert a["n"] == 4 and a["default_rate"] == 0.5
    assert a["flag_rate"] == 0.75 and a["tpr"] == 1.0 and a["fpr"] == 0.5 and a["precision"] == pytest.approx(2 / 3)
    assert b["flag_rate"] == 0.25 and b["tpr"] == 1.0 and b["fpr"] == 0.0 and b["precision"] == 1.0
    assert a["calib_ratio"] == pytest.approx(np.mean(P[:4]) / 0.5)


def test_small_groups_are_marked_unreliable():
    t = group_metrics(Y, P, FLAG, G)
    assert set(t["reliable"]) == {"too small"} and MIN_GROUP > 4


def test_bootstrap_gap_point_estimate_matches_the_difference_of_rates():
    mask_a, mask_b = (G == "A").to_numpy(), (G == "B").to_numpy()
    d, lo, hi = bootstrap_gap(Y, FLAG, mask_a, mask_b, "flag_rate", n_boot=200)
    assert d == pytest.approx(0.5)
    assert lo <= d <= hi


def test_two_identically_behaving_groups_have_a_gap_interval_around_zero():
    rng = np.random.default_rng(0)
    n = 6_000
    y = (rng.uniform(size=n) < 0.2).astype(int)
    p = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(scale=0.1, size=n), 0.01, 0.99)
    flag = p >= 0.3
    groups = pd.Series(rng.choice(["big", "small"], n, p=[0.6, 0.4]))
    t = gap_table(y, flag, groups)
    assert len(t) == 1 and t.loc[0, "versus"] == "big"
    lo, hi = (float(v) for v in t.loc[0, "flag_rate_gap"].split("[")[1].rstrip("]").split(","))
    assert lo < 0 < hi


def test_gap_table_skips_groups_that_are_too_small():
    groups = pd.Series(["big"] * 500 + ["tiny"] * 10)
    y = np.array([0, 1] * 255)
    assert len(gap_table(y, y == 1, groups)) == 0


def test_age_groups_cover_every_adult_age():
    g = age_group(pd.Series([21, 29, 30, 39, 40, 49, 50, 79]))
    assert g.astype(str).tolist() == ["<30", "<30", "30-39", "30-39", "40-49", "40-49", "50+", "50+"]
