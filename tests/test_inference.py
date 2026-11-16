"""Tests for single-customer scoring."""
import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard import config, inference

ARTIFACTS_PRESENT = (config.ARTIFACTS / "scorecard_model.json").exists()
needs_artifacts = pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="run export_artifacts first")


def good_customer(**overrides) -> dict:
    raw = {
        "LIMIT_BAL": 200_000,
        **dict(zip(config.PAY_COLS, [0, 0, 0, 0, 0, 0], strict=True)),
        **dict(zip(config.BILL_COLS, [30_000, 28_000, 27_000, 25_000, 24_000, 22_000], strict=True)),
        **dict(zip(config.PAYAMT_COLS, [5_000, 5_000, 5_000, 5_000, 5_000, 5_000], strict=True)),
    }
    raw.update(overrides)
    return raw


def test_woe_lookup_matches_binner_logic():
    spec = {"cuts": [10.0, 20.0], "woe": [0.5, 0.0, -0.5], "nan_woe": -0.1}
    assert [inference.woe_value(v, spec) for v in (5, 10, 15, 20, 99)] == [0.5, 0.0, 0.0, -0.5, -0.5]
    assert inference.woe_value(float("nan"), spec) == -0.1


@pytest.mark.parametrize(
    "bad",
    [{"LIMIT_BAL": 0}, {"LIMIT_BAL": float("nan")}, {"PAY_1": 12}, {"PAY_1": 1.5}, {"PAY_AMT1": -1}, {"PAY_2": "late"}],
)
def test_bad_inputs_are_rejected_with_a_clear_error(bad):
    with pytest.raises(ValueError):
        inference.validate(good_customer(**bad))


def test_missing_fields_are_rejected():
    raw = good_customer()
    del raw["BILL_AMT3"]
    with pytest.raises(ValueError, match="BILL_AMT3"):
        inference.validate(raw)


@needs_artifacts
def test_worse_repayment_history_raises_risk_and_lowers_score():
    ok = inference.score_customer(good_customer())
    late = inference.score_customer(good_customer(PAY_1=3, PAY_2=2))
    assert late["pd_scorecard"] > ok["pd_scorecard"] and late["score"] < ok["score"]
    assert 0 < ok["pd_scorecard"] < 1 and 0 < ok["pd_lgbm"] < 1


@needs_artifacts
def test_score_matches_the_pd_formula():
    r = inference.score_customer(good_customer(PAY_1=2))
    m = inference.load_artifacts().model
    expected = m["offset"] + m["factor"] * np.log((1 - r["pd_scorecard"]) / r["pd_scorecard"])
    assert r["score"] == pytest.approx(expected)


@needs_artifacts
def test_a_late_customer_gets_reasons_and_a_good_one_may_not():
    late = inference.score_customer(good_customer(PAY_1=3, PAY_2=3, PAY_3=2))
    assert len(late["reasons"]) >= 1 and all(r["points_vs_average"] < 0 for r in late["reasons"])


@needs_artifacts
@pytest.mark.skipif(not (config.DATA_PROCESSED / "valid_predictions.parquet").exists(), reason="needs local data")
def test_single_customer_scoring_reproduces_the_batch_predictions():
    from credit_pd_scorecard.split import load_split

    valid = load_split("valid")
    preds = pd.read_parquet(config.DATA_PROCESSED / "valid_predictions.parquet")
    for i in (0, 1, 2, 500, 3000):
        raw = {f: valid.iloc[i][f].item() for f in inference.RAW_FIELDS}
        r = inference.score_customer(raw)
        assert r["pd_scorecard"] == pytest.approx(preds["pd_scorecard"].iloc[i], abs=1e-9)
        assert r["pd_lgbm"] == pytest.approx(preds["pd_lgbm"].iloc[i], abs=1e-9)
