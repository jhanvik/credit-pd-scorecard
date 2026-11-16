"""Explainability: SHAP for the LightGBM challenger, points-based reason codes for the scorecard.

Reason codes answer "why was this customer scored as risky?" in plain words. For the scorecard each
feature's points are compared with the AVERAGE training customer's points; the features where this
customer falls furthest below average are the reasons. For LightGBM the SHAP values play that role.
Everything is computed on validation customers; the test split is not loaded.
"""
import json
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_pd_scorecard import config
from credit_pd_scorecard.features import MODEL_INPUT_COLS, prepare
from credit_pd_scorecard.reasons import REASON_TEXT, scorecard_reason_codes  # noqa: F401
from credit_pd_scorecard.report_utils import df_to_markdown
from credit_pd_scorecard.split import load_split

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"

def shap_matrix(booster: lgb.Booster, x: pd.DataFrame):
    """SHAP values on the log-odds scale, plus the base value."""
    explainer = shap.TreeExplainer(booster)
    with warnings.catch_warnings():  # shap warns that binary LightGBM output is a list; handled below
        warnings.simplefilter("ignore", UserWarning)
        values = explainer.shap_values(x)
    if isinstance(values, list):  # older shap versions return one array per class
        values = values[1]
    base = np.atleast_1d(explainer.expected_value)[-1]
    return np.asarray(values), float(base)


def bar_chart(table: pd.DataFrame, path) -> None:
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(t["feature"], t["mean_abs_shap"], color=BLUE, height=0.7)
    ax.set_xlabel("mean |SHAP| (log-odds impact per customer)", color=MUTED)
    ax.set_title("LightGBM: what drives predicted risk (validation)", loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    valid = prepare(load_split("valid"))
    woe_tr = pd.read_parquet(config.DATA_PROCESSED / "woe_train.parquet")
    woe_va = pd.read_parquet(config.DATA_PROCESSED / "woe_valid.parquet")
    preds = pd.read_parquet(config.DATA_PROCESSED / "valid_predictions.parquet")
    model = json.loads((config.DATA_PROCESSED / "scorecard_model.json").read_text())
    booster = lgb.Booster(model_file=str(config.DATA_PROCESSED / "lgbm_model.txt"))

    x = valid[MODEL_INPUT_COLS]
    values, base = shap_matrix(booster, x)
    raw = booster.predict(x, raw_score=True)
    additivity_gap = float(np.max(np.abs(base + values.sum(axis=1) - raw)))

    glob = pd.DataFrame({"feature": MODEL_INPUT_COLS, "mean_abs_shap": np.abs(values).mean(axis=0)})
    glob = glob.sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    bar_chart(glob, config.FIGURES / "shap_global.png")

    # One illustrative "declined" customer: median-PD customer among those the challenger rates above 50%.
    high = preds[preds["pd_lgbm"] > 0.5].sort_values("pd_lgbm")
    pick = high.index[len(high) // 2]
    cust = valid.loc[pick]
    local = pd.DataFrame({"feature": MODEL_INPUT_COLS, "value": x.loc[pick].to_numpy(), "shap_log_odds": values[pick]})
    local = local.sort_values("shap_log_odds", ascending=False).head(5).reset_index(drop=True)
    local["reason"] = local["feature"].map(lambda c: REASON_TEXT.get(c, c))

    mean_woe = woe_tr[list(model["coefficients"])].mean()
    codes = scorecard_reason_codes(woe_va.loc[pick], model["coefficients"], mean_woe, model["factor"])
    codes["value"] = codes["feature"].map(lambda c: cust[c])

    lines = [
        "# Explainability report", "",
        "SHAP values describe the LightGBM challenger; reason codes describe the scorecard. "
        "Both are computed on validation customers.", "",
        f"SHAP additivity check (base value + sum of SHAP = model output): largest gap {additivity_gap:.2e}.", "",
        "## Global drivers (mean |SHAP|, top 15)", df_to_markdown(glob, index=False, float_fmt="{:.4f}"), "",
        "![shap](figures/shap_global.png)", "",
        f"## Example declined customer (ID {int(cust[config.ID_COL])}; "
        f"scorecard PD {preds.loc[pick, 'pd_scorecard']:.1%}, LightGBM PD {preds.loc[pick, 'pd_lgbm']:.1%})",
        "Chosen mechanically as the median-PD customer among those above 50% PD; NOT hand-picked.", "",
        "### Scorecard reason codes (points below the average customer)", df_to_markdown(codes, index=False, float_fmt="{:.2f}"), "",
        "### LightGBM top risk-raising SHAP contributions", df_to_markdown(local, index=False, float_fmt="{:.3f}"), "",
        "Caveats: SHAP shows how the model uses features, not causes. Reason-code wording and the comparison "
        "baseline (average customer) are design choices; adverse-action rules vary by jurisdiction (verify).", "",
    ]
    (config.REPORTS / "explain_report.md").write_text("\n".join(lines))
    print(f"SHAP additivity gap {additivity_gap:.2e}")
    print(glob.head(6).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Example customer {int(cust[config.ID_COL])}")
    print(codes[["reason", "value", "points_vs_average"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(local[["reason", "value", "shap_log_odds"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
