"""Population Stability Index (PSI): has the population the model sees changed?

PSI compares the distribution of a variable (or the score) in a reference sample with a new sample:
    PSI = sum over bins of (share_new - share_ref) * ln(share_new / share_ref)
Bin edges come from the REFERENCE sample only. Rule-of-thumb bands (conventions, not laws):
below 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 significant shift.

Honest caveat for this project: our splits are random, so train-vs-validation PSI should be near zero
by construction. It shows the machinery works; it cannot show real-world drift, which needs later data.
"""
import numpy as np
import pandas as pd

from credit_pd_scorecard import config
from credit_pd_scorecard.report_utils import df_to_markdown

EPS = 1e-4  # floor for empty bins so ln() never sees zero


def psi_label(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"


def psi(reference, new, n_bins: int = 10) -> float:
    ref = np.asarray(reference, float)
    new = np.asarray(new, float)
    ref, new = ref[~np.isnan(ref)], new[~np.isnan(new)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    k = len(edges) + 1
    p_ref = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=k) / len(ref)
    p_new = np.bincount(np.searchsorted(edges, new, side="right"), minlength=k) / len(new)
    p_ref, p_new = np.clip(p_ref, EPS, None), np.clip(p_new, EPS, None)
    return float(np.sum((p_new - p_ref) * np.log(p_new / p_ref)))


def psi_table(ref_df: pd.DataFrame, new_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(ref_df[c], new_df[c])} for c in cols]
    t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    t["reading"] = t["psi"].map(psi_label)
    return t


def main() -> None:
    import json

    from credit_pd_scorecard.challenger import scorecard_probabilities
    from credit_pd_scorecard.features import prepare
    from credit_pd_scorecard.split import load_split

    train, valid = prepare(load_split("train")), prepare(load_split("valid"))
    woe_tr = pd.read_parquet(config.DATA_PROCESSED / "woe_train.parquet")
    woe_va = pd.read_parquet(config.DATA_PROCESSED / "woe_valid.parquet")
    model = json.loads((config.DATA_PROCESSED / "scorecard_model.json").read_text())
    feats = [f.removesuffix("_woe") for f in model["coefficients"]]

    p_tr, p_va = scorecard_probabilities(woe_tr), scorecard_probabilities(woe_va)
    score_psi = psi(p_tr, p_va)
    table = psi_table(train, valid, feats)

    # Illustrative stress: shift two features in a COPY of validation to show PSI reacts. Not real data.
    stressed = valid.copy()
    stressed["util_avg"] = (stressed["util_avg"] * 1.3).clip(upper=1.5)
    stressed["PAY_1"] = stressed["PAY_1"] + (stressed["PAY_1"] >= 0).astype(int)
    stress = psi_table(train, stressed, ["util_avg", "PAY_1"])

    lines = [
        "# Stability report (PSI)", "",
        "Reference = training split. Bands: <0.10 stable, 0.10-0.25 moderate, >0.25 significant (conventions).", "",
        f"Score PSI, train vs validation: {score_psi:.4f} ({psi_label(score_psi)}).", "",
        "Splits are random, so near-zero PSI here is expected by construction and does NOT prove stability over time. "
        "Real monitoring needs later data (the monitoring plan in the validation report).", "",
        "## Selected features, train vs validation", df_to_markdown(table, index=False, float_fmt="{:.4f}"), "",
        "## Illustrative stress test (synthetic shift applied to a copy of validation)",
        "util_avg multiplied by 1.3 (capped at 1.5); PAY_1 raised by one step for customers with PAY_1 >= 0.", "",
        df_to_markdown(stress, index=False, float_fmt="{:.4f}"), "",
    ]
    (config.REPORTS / "stability_report.md").write_text("\n".join(lines))
    print(f"score PSI {score_psi:.4f} ({psi_label(score_psi)}); max feature PSI {table['psi'].max():.4f}")
    print(stress.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
