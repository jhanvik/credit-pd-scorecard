"""Logistic-regression scorecard on WoE features.

Pipeline (all choices use the TRAINING split only):
  1. Candidate features: IV >= 0.02 (convention: below that a feature adds little).
  2. Drop near-duplicates: walk features from highest IV down and skip any whose WoE
     correlation with an already-kept feature exceeds 0.85.
  3. Drop high multicollinearity: iteratively remove the feature with the highest VIF
     while it is above 5 (a common rule of thumb, not a law).
  4. Fit an unregularised logistic regression of DEFAULT on the WoE values. Because
     WoE = ln(good/bad), every coefficient should be NEGATIVE; remove positive-sign
     features, then insignificant ones (p > 0.05), one at a time.
  5. Convert to points: Score = Offset - Factor * ln(odds), where odds = good:bad.
Validation is used to REPORT performance only. The test split is never loaded here.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

from credit_pd_scorecard import config

IV_MIN = 0.02
CORR_LIMIT = 0.85
VIF_LIMIT = 5.0
P_LIMIT = 0.05


def drop_near_duplicates(woe: pd.DataFrame, iv: dict[str, float], limit: float = CORR_LIMIT):
    """Keep the higher-IV feature of any pair correlated above `limit`. Returns (kept, log)."""
    corr = woe.corr().abs()
    kept: list[str] = []
    log: list[dict] = []
    for col in sorted(woe.columns, key=lambda c: -iv[c]):
        clash = [k for k in kept if corr.loc[col, k] > limit]
        if clash:
            log.append({"dropped": col, "reason": f"corr {corr.loc[col, clash[0]]:.2f} with higher-IV {clash[0]}"})
        else:
            kept.append(col)
    return kept, log


def vif_table(woe: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor per feature (the intercept column is added, not reported)."""
    x = sm.add_constant(woe, has_constant="add")
    rows = [(c, variance_inflation_factor(x.to_numpy(float), i)) for i, c in enumerate(x.columns) if c != "const"]
    return pd.DataFrame(rows, columns=["feature", "vif"]).sort_values("vif", ascending=False).reset_index(drop=True)


def drop_high_vif(woe: pd.DataFrame, limit: float = VIF_LIMIT):
    cols, log = list(woe.columns), []
    while len(cols) > 1:
        t = vif_table(woe[cols])
        if t["vif"].iloc[0] <= limit:
            break
        log.append({"dropped": t["feature"].iloc[0], "reason": f"VIF {t['vif'].iloc[0]:.1f} > {limit}"})
        cols.remove(t["feature"].iloc[0])
    return cols, log


@dataclass
class FitResult:
    features: list[str]
    params: pd.Series  # includes "const"
    table: pd.DataFrame
    log: list[dict] = field(default_factory=list)


def fit_logit(woe: pd.DataFrame, y: pd.Series, p_limit: float = P_LIMIT) -> FitResult:
    """Fit, then prune features with a wrong (positive) sign or p > p_limit, one at a time."""
    feats, log = list(woe.columns), []
    while True:
        res = sm.Logit(y.to_numpy(float), sm.add_constant(woe[feats], has_constant="add")).fit(disp=0, maxiter=200)
        coef, pval = res.params.drop("const"), res.pvalues.drop("const")
        wrong_sign = coef[coef > 0]
        if len(wrong_sign) and len(feats) > 1:
            worst = pval[wrong_sign.index].idxmax()
            log.append({"dropped": worst, "reason": f"positive coefficient {coef[worst]:+.3f} (expected negative)"})
        elif pval.max() > p_limit and len(feats) > 1:
            worst = pval.idxmax()
            log.append({"dropped": worst, "reason": f"p-value {pval[worst]:.3f} > {p_limit}"})
        else:
            break
        feats.remove(worst)
    table = pd.DataFrame({"coef": coef, "std_err": res.bse.drop("const"), "p_value": pval})
    table.index.name = "feature"
    return FitResult(feats, res.params, table.reset_index(), log)


def scaling(base_score: int = 600, base_odds: float = 50.0, pdo: int = 20) -> tuple[float, float]:
    """Factor and offset so that `base_odds`:1 good:bad scores `base_score` and +pdo doubles the odds."""
    factor = pdo / np.log(2)
    return factor, base_score - factor * np.log(base_odds)


def points_table(fit: FitResult, bins: pd.DataFrame, factor: float, offset: float) -> pd.DataFrame:
    """Integer points for every bin of every selected feature.

    ln(odds good:bad) = -(b0 + sum b_i * woe_i); points_i = -(b_i * woe + b0/n) * factor + offset/n.
    """
    n = len(fit.features)
    b0 = fit.params["const"]
    rows = []
    for f in fit.features:
        sub = bins[bins["feature"] == f.removesuffix("_woe")]
        b = fit.params[f]
        for _, r in sub.iterrows():
            pts = -(b * r["woe"] + b0 / n) * factor + offset / n
            rows.append({"feature": r["feature"], "bin": r["bin"], "woe": r["woe"], "count": r["count"],
                         "default_rate": r["default_rate"], "points": int(round(pts))})
    return pd.DataFrame(rows)


def predict_pd(fit: FitResult, woe: pd.DataFrame) -> np.ndarray:
    z = fit.params["const"] + woe[fit.features].to_numpy(float) @ fit.params[fit.features].to_numpy(float)
    return 1.0 / (1.0 + np.exp(-z))


def score_from_pd(pd_hat: np.ndarray, factor: float, offset: float) -> np.ndarray:
    pd_hat = np.clip(pd_hat, 1e-9, 1 - 1e-9)
    return offset + factor * np.log((1 - pd_hat) / pd_hat)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def rank_metrics(y: np.ndarray, risk: np.ndarray) -> dict:
    """AUC, Gini (2*AUC-1) and KS. `risk` must be HIGHER for riskier customers."""
    auc = roc_auc_score(y, risk)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y, risk)}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Ranking metrics plus the Brier score (mean squared error of the probabilities)."""
    return {**rank_metrics(y, p), "brier": float(brier_score_loss(y, p))}


def auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = config.SEED) -> tuple[float, float]:
    """95% bootstrap interval for AUC (resample rows with replacement)."""
    rng = np.random.default_rng(seed)
    n, out = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def calibration_table(y: np.ndarray, p: np.ndarray, q: int = 10) -> pd.DataFrame:
    """Average predicted PD vs observed default rate in PD buckets (bucket 1 = lowest risk)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), q=q, labels=False) + 1
    t = df.groupby("bucket").agg(customers=("y", "size"), avg_predicted_pd=("p", "mean"), observed_default_rate=("y", "mean"))
    return t.reset_index()
