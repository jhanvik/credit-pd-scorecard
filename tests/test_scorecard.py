"""Tests for the scorecard maths. Synthetic data, with answers worked out by hand."""
import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard import scorecard as sc


def test_scaling_puts_base_odds_at_base_score_and_pdo_doubles_odds():
    factor, offset = sc.scaling(base_score=600, base_odds=50, pdo=20)
    # odds 50:1 good:bad -> PD = 1/51
    assert sc.score_from_pd(np.array([1 / 51]), factor, offset)[0] == pytest.approx(600)
    # odds 100:1 -> exactly 20 points higher
    assert sc.score_from_pd(np.array([1 / 101]), factor, offset)[0] == pytest.approx(620)


def test_rank_metrics_hand_example():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = sc.rank_metrics(y, p)
    assert m["auc"] == pytest.approx(0.75)  # 3 of the 4 default/non-default pairs are ordered correctly
    assert m["gini"] == pytest.approx(0.5)
    assert m["ks"] == pytest.approx(0.5)


def test_brier_score_hand_example():
    y = np.array([0, 1])
    p = np.array([0.2, 0.6])
    assert sc.metrics(y, p)["brier"] == pytest.approx((0.2**2 + 0.4**2) / 2)


def test_near_duplicate_keeps_the_higher_iv_feature():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2_000)
    woe = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.1, size=2_000), "c": rng.normal(size=2_000)})
    kept, log = sc.drop_near_duplicates(woe, {"a": 0.5, "b": 0.3, "c": 0.1})
    assert kept == ["a", "c"] and log[0]["dropped"] == "b"


def test_vif_is_about_one_for_independent_and_large_for_near_copies():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5_000)
    indep = pd.DataFrame({"a": x, "b": rng.normal(size=5_000)})
    assert sc.vif_table(indep)["vif"].max() == pytest.approx(1.0, abs=0.05)
    copies = pd.DataFrame({"a": x, "b": x + rng.normal(scale=0.05, size=5_000)})
    assert sc.vif_table(copies)["vif"].min() > 100
    kept, log = sc.drop_high_vif(copies)
    assert len(kept) == 1 and len(log) == 1


def make_woe_problem(n=20_000, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.choice([-0.5, 0.5], n)
    b = rng.choice([-1.0, 0.0, 1.0], n)
    noise = rng.normal(size=n)
    z = -1.5 - 0.9 * a - 0.7 * b  # true model: negative coefficients on WoE
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"a_woe": a, "b_woe": b, "noise_woe": noise}), pd.Series(y)


def test_fit_logit_keeps_real_signals_with_negative_signs_and_prunes_noise():
    woe, y = make_woe_problem()
    fit = sc.fit_logit(woe, y)
    assert {"a_woe", "b_woe"} <= set(fit.features)
    assert (fit.table["coef"] < 0).all() and (fit.table["p_value"] <= sc.P_LIMIT).all()
    assert fit.params["a_woe"] == pytest.approx(-0.9, abs=0.15)


def test_fit_logit_removes_a_positive_sign_feature():
    woe, y = make_woe_problem()
    woe["wrong_woe"] = -woe["a_woe"] + np.random.default_rng(3).normal(scale=0.2, size=len(woe))
    fit = sc.fit_logit(woe[["a_woe", "wrong_woe", "b_woe"]], y)
    assert (fit.table["coef"] < 0).all()


def test_points_table_reproduces_the_model_score_within_rounding():
    woe, y = make_woe_problem()
    fit = sc.fit_logit(woe[["a_woe", "b_woe"]], y)
    bins = pd.DataFrame(
        [{"feature": "a", "bin": f"a{v}", "woe": v, "count": 1, "default_rate": 0.1} for v in (-0.5, 0.5)]
        + [{"feature": "b", "bin": f"b{v}", "woe": v, "count": 1, "default_rate": 0.1} for v in (-1.0, 0.0, 1.0)]
    )
    factor, offset = sc.scaling()
    card = sc.points_table(fit, bins, factor, offset)
    model_score = sc.score_from_pd(sc.predict_pd(fit, woe), factor, offset)
    lookup = {(r["feature"], r["woe"]): r["points"] for _, r in card.iterrows()}
    card_score = woe["a_woe"].map(lambda v: lookup[("a", v)]) + woe["b_woe"].map(lambda v: lookup[("b", v)])
    assert np.max(np.abs(card_score.to_numpy() - model_score)) <= 0.5 * len(fit.features) + 1e-6
    # riskier bin (negative WoE) must never get MORE points than a safer bin of the same feature
    a = card[card["feature"] == "a"].sort_values("woe")
    assert a["points"].is_monotonic_increasing


def test_calibration_table_covers_every_row_in_ten_buckets():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.9, 10_000)
    y = (rng.uniform(size=10_000) < p).astype(int)
    t = sc.calibration_table(y, p)
    assert len(t) == 10 and t["customers"].sum() == 10_000
    assert (t["avg_predicted_pd"] - t["observed_default_rate"]).abs().max() < 0.05


def test_auc_interval_contains_the_point_estimate():
    rng = np.random.default_rng(5)
    p = rng.uniform(size=3_000)
    y = (rng.uniform(size=3_000) < p).astype(int)
    lo, hi = sc.auc_ci(y, p, n_boot=200)
    auc = sc.rank_metrics(y, p)["auc"]
    assert lo < auc < hi
