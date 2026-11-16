"""Tests for the challenger helpers. Synthetic data only."""
import numpy as np
import pytest

from credit_pd_scorecard.challenger import make_model, paired_auc_difference


def test_identical_models_have_zero_auc_difference():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2_000)
    y = (rng.uniform(size=2_000) < p).astype(int)
    diff, lo, hi = paired_auc_difference(y, p, p, n_boot=100)
    assert diff == 0 and lo == 0 and hi == 0


def test_a_better_model_gets_a_positive_difference_and_interval():
    rng = np.random.default_rng(1)
    signal = rng.uniform(size=5_000)
    y = (rng.uniform(size=5_000) < signal).astype(int)
    good = signal
    weak = 0.3 * signal + 0.7 * rng.uniform(size=5_000)
    diff, lo, hi = paired_auc_difference(y, good, weak, n_boot=200)
    assert diff > 0 and lo > 0 and lo < diff < hi


def test_same_seed_gives_identical_predictions():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1_000, 4))
    y = (rng.uniform(size=1_000) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    a = make_model(n_estimators=30, num_leaves=7).fit(x, y).predict_proba(x)[:, 1]
    b = make_model(n_estimators=30, num_leaves=7).fit(x, y).predict_proba(x)[:, 1]
    assert a == pytest.approx(b)
    assert a.min() >= 0 and a.max() <= 1
