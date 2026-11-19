"""Sensitivity and stress tests of the frozen scorecard on the VALIDATION split (validator script).

Uses the developer's frozen artifacts and scoring function as a black box (this is NOT independent code;
see the report). Reads valid.parquet only. The consumed test split is never read.
Perturbations are illustrative shocks chosen by the validator, not forecasts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_pd_scorecard import config
from credit_pd_scorecard.final_test import frozen_predictions
from credit_pd_scorecard.inference import load_artifacts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "stress_results.md"
CUTOFF = 0.30  # frozen scorecard cutoff (README)


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join(
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |\n"
        for r in df.itertuples(index=False)
    )
    return head + body


def shock(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    d = raw.copy()
    if name == "LIMIT_BAL x0.8":
        d["LIMIT_BAL"] = d["LIMIT_BAL"] * 0.8
    elif name == "BILL_AMT1-6 x1.2":
        d[config.BILL_COLS] = d[config.BILL_COLS] * 1.2
    elif name == "PAY_AMT1-6 x0.8":
        d[config.PAYAMT_COLS] = d[config.PAYAMT_COLS] * 0.8
    elif name == "PAY_1 +1 (capped at 9)":
        d["PAY_1"] = (d["PAY_1"] + 1).clip(upper=9)
    elif name == "All PAY_x +1 (capped at 9)":
        d[config.PAY_COLS] = (d[config.PAY_COLS] + 1).clip(upper=9)
    elif name == "Combined: limit x0.8, bill x1.2, pay_amt x0.8":
        d["LIMIT_BAL"] = d["LIMIT_BAL"] * 0.8
        d[config.BILL_COLS] = d[config.BILL_COLS] * 1.2
        d[config.PAYAMT_COLS] = d[config.PAYAMT_COLS] * 0.8
    return d


def main() -> int:
    raw = pd.read_parquet(config.DATA_PROCESSED / "valid.parquet")
    art = load_artifacts()
    y = raw[config.TARGET].to_numpy()
    base, _ = frozen_predictions(raw, art)
    base_flag = base >= CUTOFF
    rows = [{"scenario": "Baseline", "mean_pd": float(base.mean()), "auc": float(roc_auc_score(y, base)),
             "flag_rate": float(base_flag.mean()), "pct_flag_changed": 0.0, "mean_abs_pd_change": 0.0}]
    for name in ["LIMIT_BAL x0.8", "BILL_AMT1-6 x1.2", "PAY_AMT1-6 x0.8", "PAY_1 +1 (capped at 9)",
                 "All PAY_x +1 (capped at 9)", "Combined: limit x0.8, bill x1.2, pay_amt x0.8"]:
        p, _ = frozen_predictions(shock(raw, name), art)
        rows.append({"scenario": name, "mean_pd": float(p.mean()), "auc": float(roc_auc_score(y, p)),
                     "flag_rate": float((p >= CUTOFF).mean()),
                     "pct_flag_changed": float(((p >= CUTOFF) != base_flag).mean()),
                     "mean_abs_pd_change": float(np.abs(p - base).mean())})
    res = pd.DataFrame(rows)

    # Missing-data behaviour: blank one input at a time (NaN) and see whether scoring still returns valid PDs.
    miss = []
    for col in ["LIMIT_BAL", "PAY_1", "BILL_AMT1", "PAY_AMT1"]:
        d = raw.copy()
        d[col] = np.nan
        try:
            p, _ = frozen_predictions(d, art)
            ok = bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
            miss.append({"blanked_column": col, "valid_pds": "yes" if ok else "NO",
                         "mean_pd": float(np.nanmean(p)), "auc": float(roc_auc_score(y, np.nan_to_num(p)))})
        except Exception as err:  # record failure mode rather than hide it
            miss.append({"blanked_column": col, "valid_pds": f"ERROR: {type(err).__name__}", "mean_pd": float("nan"),
                         "auc": float("nan")})
    miss = pd.DataFrame(miss)

    lines = ["# Sensitivity and stress tests (validator script, validation split)", "",
             f"Frozen scorecard cutoff {CUTOFF}. Shocks are illustrative, chosen by the validator.", "",
             "## Input shocks", md(res), "", "## Missing-data behaviour (one column blanked at a time)", md(miss)]
    OUT.write_text("\n".join(lines) + "\n")
    print(md(res))
    print(md(miss))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
