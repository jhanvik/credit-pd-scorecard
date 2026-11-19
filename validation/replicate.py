"""Independent re-computation of the frozen scorecard (validator script).

Written for the validation report. It deliberately imports NOTHING from the
credit_pd_scorecard package: it reads only the files the pipeline produced and
re-does the fitting, scoring and metric arithmetic with its own code.

SCOPE LIMIT (stated in the report): feature engineering and WoE binning are NOT
re-derived here. The WoE-transformed tables are taken as given, so this script
tests the fitting step, the scoring step, the points formula and the metrics.

Uses TRAIN and VALIDATION data only. The test split is never read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ARTIFACT = ROOT / "artifacts" / "scorecard_model.json"
OUT = ROOT / "validation" / "replication_results.md"

# Reported by the developer (reports/scorecard_report.md), validation split.
REPORTED = {"auc": 0.7648, "ks": 0.3906, "brier": 0.1398}
TOL_COEF = 1e-4  # refit coefficient tolerance
TOL_PD = 1e-9  # scoring tolerance against stored predictions


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    """AUC as the Mann-Whitney statistic (average ranks handle ties)."""
    ranks = pd.Series(score).rank(method="average").to_numpy()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks_stat(y: np.ndarray, score: np.ndarray) -> float:
    """Largest gap between cumulative score distributions of defaulters and non-defaulters."""
    order = np.argsort(score, kind="mergesort")
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def main() -> int:
    spec = json.loads(ARTIFACT.read_text())
    coef = spec["coefficients"]
    feats = list(coef)
    train = pd.read_parquet(PROC / "woe_train.parquet")
    valid = pd.read_parquet(PROC / "woe_valid.parquet")
    stored = pd.read_parquet(PROC / "valid_predictions.parquet")
    rows: list[tuple[str, str, str]] = []

    # R1. Refit the logistic regression on training data and compare coefficients.
    fit = sm.Logit(train["DEFAULT"], sm.add_constant(train[feats])).fit(disp=0)
    diffs = {f: float(fit.params[f] - coef[f]) for f in feats}
    diffs["intercept"] = float(fit.params["const"] - spec["intercept"])
    worst = max(diffs, key=lambda k: abs(diffs[k]))
    r1 = abs(diffs[worst]) <= TOL_COEF
    rows.append(("R1 refit coefficients", f"max abs diff {abs(diffs[worst]):.2e} ({worst})", "PASS" if r1 else "FAIL"))

    # R2. Score validation with the frozen coefficients and compare with stored PDs.
    z = spec["intercept"] + valid[feats].to_numpy() @ np.array([coef[f] for f in feats])
    pd_new = sigmoid(z)
    merged = valid[["ID"]].assign(pd_new=pd_new).merge(stored[["ID", "pd_scorecard"]], on="ID", validate="one_to_one")
    d2 = float(np.max(np.abs(merged["pd_new"] - merged["pd_scorecard"])))
    rows.append(("R2 scoring vs stored PDs", f"max abs diff {d2:.2e} on {len(merged):,} rows", "PASS" if d2 <= TOL_PD else "FAIL"))

    # R3. Metrics, with own implementations, against the developer's reported values.
    y = valid["DEFAULT"].to_numpy()
    auc_own, auc_skl = auc_rank(y, pd_new), float(roc_auc_score(y, pd_new))
    ks = ks_stat(y, pd_new)
    brier = float(np.mean((pd_new - y) ** 2))
    rows.append(("R3a AUC (own rank formula vs sklearn)", f"{auc_own:.6f} vs {auc_skl:.6f}", "PASS" if abs(auc_own - auc_skl) < 1e-9 else "FAIL"))
    for name, val in (("auc", auc_own), ("ks", ks), ("brier", brier)):
        ok = abs(round(val, 4) - REPORTED[name]) < 1e-9
        rows.append((f"R3 {name} vs reported {REPORTED[name]}", f"{val:.4f}", "PASS" if ok else "FAIL"))

    # R4. Points formula: base 600 at 50:1 good:bad odds, 20 points to double the odds.
    def points(p: float) -> float:
        return spec["offset"] + spec["factor"] * np.log((1 - p) / p)

    p50 = 1 / 51  # odds of 50 good per 1 bad
    ok_base = abs(points(p50) - 600) < 1e-6
    ok_pdo = abs(points(1 / 101) - points(1 / 51) - 20) < 1e-6  # odds 100:1 vs 50:1
    rows.append(("R4 points at 50:1 odds", f"{points(p50):.4f}", "PASS" if ok_base else "FAIL"))
    rows.append(("R4 +20 points per doubling of odds", f"{points(1 / 101) - points(1 / 51):.4f}", "PASS" if ok_pdo else "FAIL"))

    table = pd.DataFrame(rows, columns=["check", "result", "status"])
    lines = ["# Replication results (validator script)", "", "Train and validation data only; test split not read.", ""]
    lines += ["| check | result | status |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    OUT.write_text("\n".join(lines) + "\n")
    print(table.to_string(index=False))
    print(f"Wrote {OUT}")
    return 0 if (table["status"] == "PASS").all() else 1


if __name__ == "__main__":
    sys.exit(main())
