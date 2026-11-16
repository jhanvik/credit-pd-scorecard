"""Challenger model: LightGBM on the engineered (non-WoE) features, compared honestly with the scorecard.

Rules followed:
  * Hyperparameters are tuned with 5-fold stratified cross-validation INSIDE the training split only.
  * Validation is used once, to compare models. The test split is never loaded.
  * Both models are scored on the same validation customers; the AUC difference gets a paired bootstrap interval.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from credit_pd_scorecard import config
from credit_pd_scorecard import scorecard as sc
from credit_pd_scorecard.features import MODEL_INPUT_COLS, prepare
from credit_pd_scorecard.report_utils import df_to_markdown
from credit_pd_scorecard.split import load_split

N_ITER = 20
CV_FOLDS = 5
PARAM_SPACE = {
    "num_leaves": [7, 15, 31],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 400],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0],
}


def paired_auc_difference(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = config.SEED):
    """AUC(a) - AUC(b) with a 95% interval, resampling the SAME customers for both models."""
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def make_model(**params) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", subsample_freq=1, random_state=config.SEED, deterministic=True,
        force_row_wise=True, n_jobs=2, verbose=-1, **params,
    )


def scorecard_probabilities(woe_df: pd.DataFrame) -> np.ndarray:
    """Re-score with the saved scorecard coefficients (no refitting)."""
    m = json.loads((config.DATA_PROCESSED / "scorecard_model.json").read_text())
    z = m["intercept"] + sum(coef * woe_df[f].to_numpy(float) for f, coef in m["coefficients"].items())
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    train = prepare(load_split("train"))
    valid = prepare(load_split("valid"))
    woe_valid = pd.read_parquet(config.DATA_PROCESSED / "woe_valid.parquet")
    assert (woe_valid[config.ID_COL].to_numpy() == valid[config.ID_COL].to_numpy()).all(), "row order mismatch"
    xtr, ytr = train[MODEL_INPUT_COLS], train[config.TARGET].to_numpy()
    xva, yva = valid[MODEL_INPUT_COLS], valid[config.TARGET].to_numpy()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.SEED)
    search = RandomizedSearchCV(make_model(), PARAM_SPACE, n_iter=N_ITER, scoring="roc_auc", cv=cv,
                                random_state=config.SEED, n_jobs=1, refit=True)
    search.fit(xtr, ytr)
    best = search.best_estimator_
    cvres = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_top = cvres[["rank_test_score", "mean_test_score", "std_test_score", *[f"param_{k}" for k in PARAM_SPACE]]].head(5)

    best.booster_.save_model(str(config.DATA_PROCESSED / "lgbm_model.txt"))
    p_lgb_tr, p_lgb_va = best.predict_proba(xtr)[:, 1], best.predict_proba(xva)[:, 1]
    p_sc_va = scorecard_probabilities(woe_valid)
    rows = [
        {"model": "scorecard (logistic, WoE)", "split": "validation", **sc.metrics(yva, p_sc_va)},
        {"model": "LightGBM challenger", "split": "validation", **sc.metrics(yva, p_lgb_va)},
        {"model": "LightGBM challenger", "split": "train", **sc.metrics(ytr, p_lgb_tr)},
    ]
    comp = pd.DataFrame(rows)
    diff, lo, hi = paired_auc_difference(yva, p_lgb_va, p_sc_va)
    cal = sc.calibration_table(yva, p_lgb_va)
    imp = pd.DataFrame({"feature": MODEL_INPUT_COLS, "gain": best.booster_.feature_importance("gain")})
    imp["share_of_gain"] = imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False).head(10)[["feature", "share_of_gain"]]

    pd.DataFrame({config.ID_COL: valid[config.ID_COL], config.TARGET: yva, "pd_scorecard": p_sc_va, "pd_lgbm": p_lgb_va}
                 ).to_parquet(config.DATA_PROCESSED / "valid_predictions.parquet", index=False)

    verdict = (
        "The interval excludes zero: the challenger is better on validation by a statistically visible margin."
        if lo > 0 else
        "The interval includes zero: on this validation sample the two models are not clearly different."
    )
    lines = [
        "# Challenger report (LightGBM vs logistic scorecard)", "",
        "Tuning: 20 random draws from a small grid, 5-fold stratified CV inside the training split, scored by AUC. "
        "Validation is used only to compare models. The test split is untouched.", "",
        "## Best hyperparameters", "```", json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}, indent=1), "```",
        f"Cross-validated AUC (train folds): {search.best_score_:.4f}", "",
        "## Top 5 candidates by CV AUC", df_to_markdown(cv_top, index=False, float_fmt="{:.4f}"), "",
        "## Comparison", df_to_markdown(comp, index=False, float_fmt="{:.4f}"), "",
        f"Validation AUC difference (LightGBM minus scorecard): {diff:+.4f}, 95% paired-bootstrap interval {lo:+.4f} to {hi:+.4f}. {verdict}", "",
        "## Challenger calibration by risk bucket (validation)", df_to_markdown(cal, index=False, float_fmt="{:.4f}"), "",
        "## Top 10 features by gain (global importance only; not an explanation of any one decision, SHAP comes later)",
        df_to_markdown(imp, index=False, float_fmt="{:.3f}"), "",
        "Results can differ in the last decimals across machines and library versions even with fixed seeds.", "",
    ]
    (config.REPORTS / "challenger_report.md").write_text("\n".join(lines))
    print(df_to_markdown(comp, index=False, float_fmt="{:.4f}"))
    print(f"AUC diff {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] | best CV AUC {search.best_score_:.4f}")
    print(search.best_params_)


if __name__ == "__main__":
    main()
