"""Project-wide constants: paths, random seed, column groups and documented codes.

Keeping these in one place means every module (audit, split, features, WoE)
agrees on the same names, and a change is made once.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = REPORTS / "figures"

SEED = 42
ID_COL = "ID"
TARGET = "DEFAULT"

# Stratified train / validation / test split (the test set is locked until the end).
SPLIT_FRACTIONS = {"train": 0.6, "valid": 0.2, "test": 0.2}

# Column groups. In the raw file the "1" columns are September 2005, "2" is
# August, ... and "6" is April 2005 (see README, section 4).
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]  # repayment status
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]  # bill statement amount
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]  # amount paid
RAW_BEHAVIOR_COLS = ["LIMIT_BAL", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]

# Demographic columns: NOT used as model inputs (fair-lending caution).
# They are kept only for the fairness check later.
DEMOGRAPHIC_COLS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

# Codes that the UCI variable description actually defines.
DOCUMENTED_CODES = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    "PAY": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},  # -2 and 0 are NOT documented
}
