"""Tests for WoE binning. All data is synthetic, with answers worked out by hand."""
import math

import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard.woe import WoeBinner, fit_all, iv_strength, transform_all


def make_data(n: int = 20_000, slope: float = 0.5, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """x uniform in [0, 1]; default probability rises (slope > 0) or falls with x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    p = 0.25 + slope * (x - 0.5)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.Series(x), pd.Series(y)


def test_iv_and_woe_match_hand_calculation():
    # x=0: 80 good, 20 bad.  x=1: 40 good, 60 bad.  Totals: 120 good, 80 bad.
    x = pd.Series([0] * 100 + [1] * 100)
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 40 + [1] * 60)
    b = WoeBinner(monotonic=None, min_bad_count=1, min_bin_frac=0.01, smoothing=0.0).fit(x, y)
    woe0 = math.log((80 / 120) / (20 / 80))
    woe1 = math.log((40 / 120) / (60 / 80))
    iv = (80 / 120 - 20 / 80) * woe0 + (40 / 120 - 60 / 80) * woe1
    assert b.bin_table_["woe"].tolist() == pytest.approx([woe0, woe1])
    assert b.iv_ == pytest.approx(iv)
    assert b.iv_ == pytest.approx(0.7466, abs=1e-3)


def test_safer_bin_has_positive_woe():
    x = pd.Series([0] * 100 + [1] * 100)
    y = pd.Series([0] * 95 + [1] * 5 + [0] * 50 + [1] * 50)
    b = WoeBinner(monotonic=None, min_bad_count=1, min_bin_frac=0.01).fit(x, y)
    assert b.bin_table_.loc[0, "woe"] > 0 > b.bin_table_.loc[1, "woe"]


def test_monotonic_auto_follows_the_data_direction():
    x, y = make_data(slope=-0.5)
    b = WoeBinner(monotonic="auto").fit(x, y)
    rates = b.bin_table_["default_rate"].to_numpy()
    assert (np.diff(rates) <= 1e-12).all()  # default rate falls as x rises


def test_monotonic_is_enforced_on_a_u_shaped_relationship():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.uniform(0, 1, 30_000))
    p = 0.1 + 0.6 * (x - 0.5) ** 2 * 4
    y = pd.Series((rng.uniform(0, 1, 30_000) < p).astype(int))
    b = WoeBinner(monotonic="ascending").fit(x, y)
    assert (np.diff(b.bin_table_["default_rate"].to_numpy()) >= -1e-12).all()


def test_bins_respect_minimum_size_and_maximum_count():
    x, y = make_data()
    b = WoeBinner(max_bins=4, min_bin_frac=0.10).fit(x, y)
    table = b.bin_table_
    assert len(table) <= 4
    assert (table["count"] >= math.ceil(0.10 * len(x))).all()
    assert table["count_share"].sum() == pytest.approx(1.0)


def test_large_missing_group_gets_its_own_bin():
    rng = np.random.default_rng(2)
    n = 20_000
    x = pd.Series(rng.uniform(0, 1, n))
    miss = rng.uniform(0, 1, n) < 0.2
    y = pd.Series((rng.uniform(0, 1, n) < np.where(miss, 0.6, 0.15)).astype(int))
    x[miss] = np.nan
    b = WoeBinner().fit(x, y)
    assert b.bin_table_["bin"].iloc[-1] == "missing"
    assert b.nan_woe_ < 0  # missing is riskier than average here
    assert b.transform(pd.Series([np.nan]))[0] == pytest.approx(b.nan_woe_)


def test_missing_unseen_in_training_gets_neutral_woe():
    x, y = make_data(5_000)
    b = WoeBinner().fit(x, y)
    assert b.transform(pd.Series([np.nan]))[0] == 0.0


def test_transform_agrees_with_the_bin_table():
    x, y = make_data()
    b = WoeBinner().fit(x, y)
    values, counts = np.unique(b.transform(x), return_counts=True)
    table = b.bin_table_.sort_values("woe")
    assert values == pytest.approx(table["woe"].to_numpy())
    assert counts.tolist() == table["count"].tolist()


def test_constant_feature_has_zero_iv_and_one_bin():
    x = pd.Series([5.0] * 1_000)
    y = pd.Series([0, 1] * 500)
    b = WoeBinner().fit(x, y)
    assert len(b.bin_table_) == 1
    assert b.iv_ == pytest.approx(0.0, abs=1e-9)


def test_a_pile_of_zeros_gets_its_own_bin():
    rng = np.random.default_rng(3)
    zeros = np.zeros(8_000)
    positive = rng.uniform(1, 100, 12_000)
    x = pd.Series(np.concatenate([zeros, positive]))
    y = pd.Series((rng.uniform(0, 1, 20_000) < np.where(x == 0, 0.5, 0.15)).astype(int))
    b = WoeBinner(monotonic=None).fit(x, y)
    assert 0 < b.cuts_[0] <= positive.min()  # the first cut separates zeros from everything else
    assert b.bin_table_.loc[0, "count"] == 8_000  # the first bin is exactly the zeros
    assert b.transform(pd.Series([0.0]))[0] != b.transform(pd.Series([positive.min()]))[0]


def test_a_pile_of_zeros_is_isolated_even_when_quantile_cuts_are_far_above_zero():
    # 6% zeros: the 5th-percentile cut would otherwise land on a large positive value.
    rng = np.random.default_rng(4)
    x = pd.Series(np.concatenate([np.zeros(1_200), rng.uniform(50, 100, 18_800)]))
    y = pd.Series((rng.uniform(0, 1, 20_000) < np.where(x == 0, 0.6, 0.15)).astype(int))
    b = WoeBinner(monotonic=None, min_bad_count=1).fit(x, y)
    assert b.bin_table_.loc[0, "count"] == 1_200


def test_neighbouring_bins_with_the_same_default_rate_are_merged():
    # The outcome does not depend on x at all, so extra bins carry no information.
    rng = np.random.default_rng(5)
    x = pd.Series(rng.uniform(0, 1, 20_000))
    y = pd.Series((rng.uniform(0, 1, 20_000) < 0.2).astype(int))
    b = WoeBinner(monotonic="ascending", max_bins=10).fit(x, y)
    assert len(b.bin_table_) <= 3
    assert b.iv_ < 0.02


def test_target_must_be_binary():
    with pytest.raises(ValueError):
        WoeBinner().fit(pd.Series([1.0, 2.0, 3.0]), pd.Series([0, 1, 2]))


def test_fit_all_and_transform_all_shapes():
    x, y = make_data(5_000)
    df = pd.DataFrame({"a": x, "b": x**2, "DEFAULT": y})
    binners, summary = fit_all(df, ["a", "b"], "DEFAULT", overrides={"b": {"monotonic": None}})
    out = transform_all(df, binners)
    assert list(out.columns) == ["a_woe", "b_woe"]
    assert len(out) == len(df) and not out.isna().any().any()
    assert summary["iv"].is_monotonic_decreasing


def test_iv_on_same_rows_equals_training_iv_and_is_close_on_fresh_rows():
    x, y = make_data()
    b = WoeBinner().fit(x, y)
    assert b.iv_on(x, y) == pytest.approx(b.iv_, rel=1e-6)
    x2, y2 = make_data(seed=99)  # fresh rows, same process: slightly lower (fitted to noise), not collapsed
    assert 0.9 * b.iv_ < b.iv_on(x2, y2) <= 1.02 * b.iv_


def test_iv_strength_labels():
    assert iv_strength(0.01) == "not useful"
    assert iv_strength(0.05) == "weak"
    assert iv_strength(0.2) == "medium"
    assert iv_strength(0.4) == "strong"
    assert "leakage" in iv_strength(0.9)
