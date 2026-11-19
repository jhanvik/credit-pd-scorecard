"""Outcomes analysis of the frozen scorecard on the VALIDATION split (validator script).

Independent of the model package: reads stored predictions and the raw validation
rows only. The test split is not read (it was already consumed by the developer).
All data here has been seen by the developer, so this is a re-examination, not a
fresh out-of-sample test (see finding F2).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "validation" / "outcomes_results.md"
SEED = 42
N_BOOT = 500


def md(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join(
        "| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in row) + " |\n"
        for row in df.itertuples(index=False)
    )
    return head + body


def boot_auc(y: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    pred = pd.read_parquet(PROC / "valid_predictions.parquet")
    raw = pd.read_parquet(PROC / "valid.parquet")
    df = pred.merge(raw[["ID", "LIMIT_BAL", "PAY_1"]], on="ID", validate="one_to_one")
    y, p = df["DEFAULT"].to_numpy(), df["pd_scorecard"].to_numpy()

    # 1. Calibration in the large and calibration slope/intercept.
    eps = 1e-9
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    fit = sm.Logit(y, sm.add_constant(logit)).fit(disp=0)
    ci = fit.conf_int()
    citl = pd.DataFrame(
        [
            {"measure": "mean predicted PD", "value": float(p.mean())},
            {"measure": "observed default rate", "value": float(y.mean())},
            {"measure": "calibration intercept (ideal 0)", "value": float(fit.params[0])},
            {"measure": "calibration slope (ideal 1)", "value": float(fit.params[1])},
            {"measure": "slope 95% CI low", "value": float(ci[1][0])},
            {"measure": "slope 95% CI high", "value": float(ci[1][1])},
        ]
    )

    # 2. Decile calibration with a normal-approximation band on the predicted rate.
    df["decile"] = pd.qcut(df["pd_scorecard"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    g = df.groupby("decile").agg(n=("DEFAULT", "size"), avg_pd=("pd_scorecard", "mean"), observed=("DEFAULT", "mean")).reset_index()
    g["gap"] = g["observed"] - g["avg_pd"]
    se = np.sqrt(g["avg_pd"] * (1 - g["avg_pd"]) / g["n"])
    g["outside_95"] = np.where(np.abs(g["gap"]) > 1.96 * se, "YES", "")
    reversals = int((np.diff(g["observed"].to_numpy()) < 0).sum())

    # 3. Segment AUC with bootstrap intervals.
    rng = np.random.default_rng(SEED)
    df["limit_band"] = pd.qcut(df["LIMIT_BAL"], 3, labels=["low limit", "mid limit", "high limit"])
    df["pay1_group"] = np.where(df["PAY_1"] >= 1, "PAY_1 >= 1 (late)", "PAY_1 <= 0 (not late)")
    rows = []
    for col in ("limit_band", "pay1_group"):
        for name, sub in df.groupby(col, observed=True):
            ys, ss = sub["DEFAULT"].to_numpy(), sub["pd_scorecard"].to_numpy()
            lo, hi = boot_auc(ys, ss, rng)
            rows.append({"segment": str(name), "n": len(sub), "default_rate": float(ys.mean()), "auc": float(roc_auc_score(ys, ss)), "ci_low": lo, "ci_high": hi})
    seg = pd.DataFrame(rows)

    lines = ["# Outcomes analysis (validator script, validation split)", ""]
    lines += ["Data already seen by the developer; a re-examination, not a fresh test.", ""]
    lines += ["## Calibration summary", md(citl), ""]
    lines += ["## Decile calibration", md(g), f"Rank-ordering reversals in observed rate across deciles: {reversals}.", ""]
    lines += ["## AUC by segment (500 bootstrap resamples, seed 42)", md(seg)]
    OUT.write_text("\n".join(lines) + "\n")
    print(md(citl))
    print(md(g))
    print(f"reversals={reversals}")
    print(md(seg))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
