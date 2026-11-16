"""Tests for the cutoff / net-benefit maths, with answers worked out by hand."""
import numpy as np
import pytest

from credit_pd_scorecard.business import (
    Assumptions,
    best_cutoff,
    bootstrap_net,
    contributions,
    evaluate,
    paired_net_difference,
    sweep,
)

A = Assumptions(lgd=0.5, effect=0.2, outreach_cost=10.0, false_alarm_cost=100.0)
Y = np.array([1, 0])
P = np.array([0.8, 0.4])
EAD = np.array([1_000.0, 1_000.0])


def test_net_benefit_hand_calculation_at_three_cutoffs():
    # treat only the defaulter: 0.2 * 0.5 * 1000 - 10 = 90
    assert contributions(Y, P, EAD, 0.5, A).sum() == pytest.approx(90.0)
    # treat both: 90 + (-10 - 100) = -20
    assert contributions(Y, P, EAD, 0.3, A).sum() == pytest.approx(-20.0)
    # treat nobody: 0
    assert contributions(Y, P, EAD, 0.9, A).sum() == 0.0


def test_evaluate_reports_treated_share_precision_and_recall():
    r = evaluate(Y, P, EAD, 0.5, A)
    assert r["treated_share"] == 0.5 and r["precision"] == 1.0 and r["recall"] == 1.0
    assert evaluate(Y, P, EAD, 0.3, A)["precision"] == 0.5


def test_a_customer_exactly_at_the_cutoff_is_treated():
    assert evaluate(Y, np.array([0.5, 0.1]), EAD, 0.5, A)["treated_share"] == 0.5


def test_best_cutoff_picks_the_highest_net_benefit():
    t = sweep(Y, P, EAD, A, cutoffs=[0.3, 0.5, 0.9])
    assert best_cutoff(t) == 0.5


def test_exposure_scales_the_benefit_of_catching_a_defaulter():
    big = contributions(Y, P, np.array([5_000.0, 1_000.0]), 0.5, A).sum()
    assert big == pytest.approx(0.2 * 0.5 * 5_000 - 10)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=4_000)
    y = (rng.uniform(size=4_000) < p).astype(int)
    ead = rng.uniform(5_000, 80_000, 4_000)
    point = contributions(y, p, ead, 0.4, Assumptions()).sum()
    lo, hi = bootstrap_net(y, p, ead, 0.4, Assumptions(), n_boot=300)
    assert lo < point < hi


def test_paired_difference_of_identical_models_is_zero():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=1_000)
    y = (rng.uniform(size=1_000) < p).astype(int)
    ead = np.full(1_000, 20_000.0)
    assert paired_net_difference(y, p, p, ead, 0.4, 0.4, Assumptions(), n_boot=50) == (0.0, 0.0, 0.0)
