"""Tests for cleaning and feature engineering, using one hand-checked customer."""
import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard import config
from credit_pd_scorecard.features import (
    ENGINEERED_COLS,
    add_features,
    clean_codes,
    prepare,
)


def one_customer(**overrides) -> pd.DataFrame:
    row = {
        "LIMIT_BAL": 100_000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        **dict(zip(config.PAY_COLS, [2, 0, -1, -2, 1, 0])),
        **dict(zip(config.BILL_COLS, [50_000, 40_000, 30_000, 20_000, 10_000, 0])),
        **dict(zip(config.PAYAMT_COLS, [20_000, 30_000, 10_000, 5_000, 0, 0])),
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_hand_checked_values():
    out = add_features(one_customer()).iloc[0]
    assert out["util_1"] == pytest.approx(0.5)
    assert out["util_avg"] == pytest.approx(0.25)
    assert out["util_max"] == pytest.approx(0.5)
    assert out["pay_ratio_1"] == pytest.approx(20_000 / 40_000)
    # paid 65,000 against positive prior bills 40k+30k+20k+10k = 100k
    assert out["pay_ratio_avg"] == pytest.approx(0.65)
    assert out["max_delay"] == 2
    assert out["n_delay_months"] == 2
    assert out["delay_sum_recent3"] == 2
    assert out["bill_trend"] == pytest.approx(0.5)
    assert out["pay_to_limit"] == pytest.approx(0.65)


def test_adds_exactly_the_documented_columns_and_does_not_change_the_input():
    df = one_customer()
    before = df.copy()
    out = add_features(df)
    assert [c for c in out.columns if c not in df.columns] == ENGINEERED_COLS
    pd.testing.assert_frame_equal(df, before)


def test_features_never_need_the_target():
    # No DEFAULT column present: if a feature touched the target this would raise.
    assert config.TARGET not in one_customer().columns
    add_features(one_customer())


def test_negative_bill_counts_as_zero_utilization():
    out = add_features(one_customer(BILL_AMT1=-5_000)).iloc[0]
    assert out["util_1"] == 0.0


def test_utilization_and_payment_ratio_are_capped():
    out = add_features(one_customer(BILL_AMT1=900_000, PAY_AMT1=900_000)).iloc[0]
    assert out["util_1"] == pytest.approx(1.5)
    assert out["pay_ratio_1"] == pytest.approx(1.0)


def test_payment_ratio_is_missing_when_there_was_no_bill_to_pay():
    out = add_features(one_customer(BILL_AMT2=0, BILL_AMT3=-10, BILL_AMT4=0, BILL_AMT5=0, BILL_AMT6=0)).iloc[0]
    assert np.isnan(out["pay_ratio_1"]) and np.isnan(out["pay_ratio_avg"])


def test_codes_minus2_minus1_and_0_all_mean_no_delay():
    quiet = dict(zip(config.PAY_COLS, [-2, -1, 0, 0, -1, -2]))
    out = add_features(one_customer(**quiet)).iloc[0]
    assert out["max_delay"] == 0 and out["n_delay_months"] == 0 and out["delay_sum_recent3"] == 0


def test_zero_credit_limit_is_rejected():
    with pytest.raises(ValueError):
        add_features(one_customer(LIMIT_BAL=0))


def test_clean_codes_folds_undocumented_values_into_other():
    df = pd.DataFrame({"EDUCATION": [0, 1, 4, 5, 6], "MARRIAGE": [0, 1, 2, 3, 0]})
    out = clean_codes(df)
    assert out["EDUCATION"].tolist() == [4, 1, 4, 4, 4]
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3, 3]


def test_prepare_runs_both_steps():
    out = prepare(one_customer(EDUCATION=5, MARRIAGE=0))
    assert out.loc[0, "EDUCATION"] == 4 and out.loc[0, "MARRIAGE"] == 3
    assert "util_avg" in out.columns
