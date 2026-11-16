"""Tests for PSI, reason codes and SHAP additivity. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard.challenger import make_model
from credit_pd_scorecard.explain import scorecard_reason_codes, shap_matrix
from credit_pd_scorecard.stability import psi, psi_label, psi_table


def test_psi_of_identical_samples_is_zero_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    same_dist = psi(ref, rng.normal(size=20_000))
    shifted = psi(ref, rng.normal(loc=0.5, size=20_000))
    assert same_dist < 0.01 < shifted and psi_label(same_dist) == "stable"


def test_psi_matches_a_hand_calculation():
    # reference: 50% in each of two bins; new sample: 80% / 20%.
    ref = np.array([0] * 50 + [1] * 50)
    new = np.array([0] * 80 + [1] * 20)
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(ref, new, n_bins=2) == pytest.approx(expected)


def test_psi_labels_and_table_ordering():
    assert [psi_label(v) for v in (0.05, 0.15, 0.4)] == ["stable", "moderate shift", "significant shift"]
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=5_000), "b": rng.normal(size=5_000)})
    new = pd.DataFrame({"a": rng.normal(size=5_000), "b": rng.normal(loc=1.0, size=5_000)})
    t = psi_table(ref, new, ["a", "b"])
    assert t["feature"].tolist() == ["b", "a"] and t["psi"].is_monotonic_decreasing


def test_psi_ignores_missing_values():
    ref = np.array([1.0, 2.0, 3.0, np.nan] * 100)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-12)


def test_reason_codes_list_only_features_below_average_worst_first():
    coefs = {"a_woe": -1.0, "b_woe": -0.5, "c_woe": -0.2}
    mean = pd.Series({"a_woe": 0.0, "b_woe": 0.0, "c_woe": 0.0})
    row = pd.Series({"a_woe": -1.0, "b_woe": -1.0, "c_woe": 0.5})  # a, b riskier than average; c safer
    t = scorecard_reason_codes(row, coefs, mean, factor=10.0, top_k=3)
    assert t["feature"].tolist() == ["a", "b"]  # c is above average, so it is not a reason
    assert t["points_vs_average"].tolist() == pytest.approx([-10.0, -5.0])


def test_shap_values_add_up_to_the_model_output():
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(800, 4)), columns=list("abcd"))
    y = (rng.uniform(size=800) < 1 / (1 + np.exp(-(1.5 * x["a"] - x["b"])))).astype(int)
    model = make_model(n_estimators=40, num_leaves=7).fit(x, y)
    values, base = shap_matrix(model.booster_, x)
    raw = model.booster_.predict(x, raw_score=True)
    assert np.max(np.abs(base + values.sum(axis=1) - raw)) < 1e-6
    assert np.abs(values[:, :2]).mean() > np.abs(values[:, 2:]).mean()  # the real signals matter most
