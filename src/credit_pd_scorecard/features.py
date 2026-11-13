"""Row-wise cleaning and feature engineering.

Every feature here is computed from the SAME customer's own columns, using
fixed rules (no averages learned from other rows). That is why it is safe to
apply before or after the split: nothing here can leak information from the
validation or test rows into training.

Assumption (not stated in the UCI description, so verify before relying on it):
the payment made in month k settles the bill of month k+1 in the raw numbering
(PAY_AMT1 = September payment pays the BILL_AMT2 = August statement).
"""
import numpy as np
import pandas as pd

from credit_pd_scorecard import config

# Fixed caps chosen from the data's own ranges (not from the target).
UTIL_CAP = 1.5  # bill / limit; 95% of values are at or below about 1.0
PAY_RATIO_CAP = 1.0  # "paid in full or more"
PAY_TO_LIMIT_CAP = 2.0
BILL_TREND_RANGE = (-1.0, 1.0)

ENGINEERED_COLS = [
    "util_1",  # latest month's bill as a share of the credit limit
    "util_avg",  # average utilization over the six months
    "util_max",  # worst month's utilization
    "pay_ratio_1",  # latest payment as a share of the bill it settles
    "pay_ratio_avg",  # total paid / total billed over the five settled bills
    "max_delay",  # worst months-late status in the window (0 if never late)
    "n_delay_months",  # number of months with a delay of 1+ months
    "delay_sum_recent3",  # months-late summed over the latest 3 months
    "bill_trend",  # (latest bill - oldest bill) / limit
    "pay_to_limit",  # total paid over six months / limit
]
MODEL_INPUT_COLS = [*config.RAW_BEHAVIOR_COLS, *ENGINEERED_COLS]


def clean_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented category codes into the documented 'other' code."""
    out = df.copy()
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the engineered columns added."""
    if (df["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL must be positive to compute utilization")
    out = df.copy()
    limit = out["LIMIT_BAL"].astype(float)
    bills = out[config.BILL_COLS].astype(float)
    pays = out[config.PAYAMT_COLS].astype(float)
    status = out[config.PAY_COLS].astype(float)

    util = bills.div(limit, axis=0).clip(lower=0.0, upper=UTIL_CAP)
    out["util_1"] = util["BILL_AMT1"]
    out["util_avg"] = util.mean(axis=1)
    out["util_max"] = util.max(axis=1)

    # Payment k settles the bill of month k+1; only meaningful if that bill was positive.
    prior_bill = bills.iloc[:, 1:].to_numpy()
    paid = pays.iloc[:, :5].to_numpy()
    positive = prior_bill > 0
    ratio = np.divide(paid, prior_bill, out=np.full(prior_bill.shape, np.nan), where=positive)
    out["pay_ratio_1"] = np.clip(ratio[:, 0], 0.0, PAY_RATIO_CAP)
    total_billed = np.where(positive, prior_bill, 0.0).sum(axis=1)
    total_paid = np.where(positive, paid, 0.0).sum(axis=1)
    avg_ratio = np.divide(
        total_paid, total_billed, out=np.full(total_billed.shape, np.nan), where=total_billed > 0
    )
    out["pay_ratio_avg"] = np.clip(avg_ratio, 0.0, PAY_RATIO_CAP)

    delay = status.clip(lower=0.0)  # -2, -1 and 0 all mean "no delay recorded"
    out["max_delay"] = delay.max(axis=1)
    out["n_delay_months"] = (status >= 1).sum(axis=1)
    out["delay_sum_recent3"] = delay[["PAY_1", "PAY_2", "PAY_3"]].sum(axis=1)

    out["bill_trend"] = ((bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit).clip(*BILL_TREND_RANGE)
    out["pay_to_limit"] = (pays.sum(axis=1) / limit).clip(lower=0.0, upper=PAY_TO_LIMIT_CAP)
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean category codes, then add engineered features."""
    return add_features(clean_codes(df))
