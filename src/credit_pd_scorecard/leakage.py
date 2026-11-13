"""Leakage check: which columns may a model see, and does any look "too good"?

Leakage means a model input contains information that would not exist at the
moment of the decision (or that is the answer in disguise). It makes results look
great in testing and fail in real life. Two kinds of evidence are collected:

1. A written decision for every column group (was it known at prediction time?).
2. An automatic screen: the AUC of each feature used ALONE, on the training split.
   In consumer credit, a single behavioral feature scoring above ~0.75 AUC is
   unusual (a rule of thumb, not a law) and deserves a second look.
"""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_pd_scorecard import config
from credit_pd_scorecard.features import ENGINEERED_COLS, MODEL_INPUT_COLS, prepare
from credit_pd_scorecard.report_utils import df_to_markdown
from credit_pd_scorecard.split import load_split

SUSPICIOUS_AUC = 0.75

DECISIONS = [
    ("ID", "Row identifier", "Exclude", "No behavioral meaning; could leak row order."),
    (
        "LIMIT_BAL",
        "Credit limit given to the customer",
        "Use",
        "A property of the account, set before the observation window (assumed from the UCI text).",
    ),
    (
        "PAY_1 .. PAY_6",
        "Repayment status, Sept to Apr 2005",
        "Use",
        "Observed in the six months before the outcome month. Residual risk: if the Sept status "
        "were recorded after the outcome window opened it would leak; the UCI text does not say.",
    ),
    ("BILL_AMT1 .. 6", "Bill statement amounts, Sept to Apr", "Use", "Observed before the outcome month."),
    ("PAY_AMT1 .. 6", "Amounts paid, Sept to Apr", "Use", "Observed before the outcome month."),
    (
        "SEX, EDUCATION, MARRIAGE, AGE",
        "Demographics",
        "Exclude from the model",
        "Fair-lending caution (verify the rules that apply to you). Kept only for the later fairness check.",
    ),
    (
        "Engineered features (10)",
        "Ratios and delinquency counts",
        "Use",
        "Built from the same customer's own columns above with fixed rules; they never read the "
        "target or other customers' rows.",
    ),
    (config.TARGET, "default payment next month", "Target", "The outcome to predict."),
]


def single_feature_auc(df: pd.DataFrame, cols: list[str], target: str = config.TARGET) -> pd.DataFrame:
    """AUC of each column used alone (missing rows skipped, orientation reported)."""
    rows = []
    for col in cols:
        ok = df[col].notna()
        auc = roc_auc_score(df.loc[ok, target], df.loc[ok, col])
        rows.append(
            {
                "feature": col,
                "auc_alone": max(auc, 1 - auc),
                "higher_value_means": "more default" if auc >= 0.5 else "less default",
                "rows_used": int(ok.sum()),
                "flag": "INVESTIGATE" if max(auc, 1 - auc) > SUSPICIOUS_AUC else "",
            }
        )
    return pd.DataFrame(rows).sort_values("auc_alone", ascending=False).reset_index(drop=True)


def render_markdown(screen: pd.DataFrame, demo: pd.DataFrame, manifest: dict | None) -> str:
    decisions = pd.DataFrame(DECISIONS, columns=["columns", "meaning", "decision", "reason"])
    lines = [
        "# Leakage check",
        "",
        "## 1. Decision for every column",
        "",
        df_to_markdown(decisions, index=False),
        "",
        "## 2. Automatic screen: each model input used alone (training split)",
        f"Flag threshold: AUC > {SUSPICIOUS_AUC} (rule of thumb). "
        f"Features flagged: {int((screen['flag'] != '').sum())}.",
        "",
        df_to_markdown(screen, index=False, float_fmt="{:.3f}"),
        "",
        "## 3. Demographic columns (excluded from the model), for reference",
        "",
        df_to_markdown(demo, index=False, float_fmt="{:.3f}"),
        "",
        "## 4. Other checks",
    ]
    if manifest:
        overlap = manifest["rows_matching_train_features"]
        lines.append(
            f"- Rows whose features exactly match a training row: validation {overlap['valid']}, "
            f"test {overlap['test']} (negligible, but reported)."
        )
    lines += [
        "- All fitting (bin edges, IV, selection) uses the training split only.",
        "- Assumption to verify: every input is measured before the outcome month. The UCI text lists "
        "April to September 2005 for inputs and does not name the outcome month.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    train = prepare(load_split("train"))
    screen = single_feature_auc(train, MODEL_INPUT_COLS)
    demo = single_feature_auc(train, config.DEMOGRAPHIC_COLS)
    manifest_path = config.DATA_PROCESSED / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "leakage_table.md").write_text(render_markdown(screen, demo, manifest))
    print(f"Screened {len(screen)} model inputs ({len(ENGINEERED_COLS)} engineered).")
    print(screen.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Flagged above AUC {SUSPICIOUS_AUC}: {int((screen['flag'] != '').sum())}")


if __name__ == "__main__":
    main()
