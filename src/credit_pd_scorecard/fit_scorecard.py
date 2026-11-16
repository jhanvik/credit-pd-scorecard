"""Phase 4 runner: select features, fit the logistic scorecard, report validation performance.

Reads the WoE files written by build_features. The TEST split is never loaded.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_pd_scorecard import config
from credit_pd_scorecard import scorecard as sc
from credit_pd_scorecard.report_utils import df_to_markdown

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def calibration_figure(cal: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    top = float(max(cal["avg_predicted_pd"].max(), cal["observed_default_rate"].max())) * 1.1
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--", label="perfect calibration")
    ax.plot(cal["avg_predicted_pd"], cal["observed_default_rate"], color=BLUE, lw=2, marker="o", ms=6, label="validation deciles")
    ax.set_xlabel("average predicted PD", color=MUTED)
    ax.set_ylabel("observed default rate", color=MUTED)
    ax.set_title("Calibration on validation (10 risk buckets)", loc="left", color=INK, fontsize=11)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    train = pd.read_parquet(config.DATA_PROCESSED / "woe_train.parquet")
    valid = pd.read_parquet(config.DATA_PROCESSED / "woe_valid.parquet")
    bins = pd.read_parquet(config.DATA_PROCESSED / "woe_bin_tables.parquet")
    woe_cols = [c for c in train.columns if c.endswith("_woe")]
    ytr, yva = train[config.TARGET].to_numpy(), valid[config.TARGET].to_numpy()

    # IV per feature, computed on train (same definition as the IV report).
    from credit_pd_scorecard.woe import _woe_iv  # noqa: PLC0415

    iv = {}
    for c in woe_cols:
        g = pd.crosstab(train[c].round(12), train[config.TARGET]).reindex(columns=[0, 1], fill_value=0)
        iv[c] = _woe_iv(g[0].to_numpy(float), g[1].to_numpy(float), 0.0)[1]

    cand = [c for c in woe_cols if iv[c] >= sc.IV_MIN]
    steps = [{"dropped": c, "reason": f"IV {iv[c]:.3f} < {sc.IV_MIN}"} for c in woe_cols if iv[c] < sc.IV_MIN]
    kept, log = sc.drop_near_duplicates(train[cand], iv)
    steps += log
    vif_before = sc.vif_table(train[kept])
    kept, log = sc.drop_high_vif(train[kept])
    steps += log
    fit = sc.fit_logit(train[kept], train[config.TARGET])
    steps += fit.log
    vif_final = sc.vif_table(train[fit.features])

    p_tr, p_va = sc.predict_pd(fit, train), sc.predict_pd(fit, valid)
    m_tr, m_va = sc.metrics(ytr, p_tr), sc.metrics(yva, p_va)
    lo, hi = sc.auc_ci(yva, p_va)
    cal = sc.calibration_table(yva, p_va)
    factor, offset = sc.scaling()
    card = sc.points_table(fit, bins, factor, offset)

    # The points table must reproduce the model's score (up to integer rounding).
    def card_score(df: pd.DataFrame) -> np.ndarray:
        total = np.zeros(len(df))
        for f in fit.features:
            b = bins[bins["feature"] == f.removesuffix("_woe")]
            lookup = dict(zip(b["woe"].round(12), card[card["feature"] == f.removesuffix("_woe")]["points"]))
            total += df[f].round(12).map(lookup).to_numpy(float)
        return total

    s_card, s_model = card_score(valid), sc.score_from_pd(p_va, factor, offset)
    gap = float(np.max(np.abs(s_card - s_model)))
    m_score = sc.rank_metrics(yva, -s_card)  # lower score = riskier, so negate for AUC/KS

    comp = pd.DataFrame([{"split": "train", **m_tr}, {"split": "validation", **m_va}])
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    calibration_figure(cal, config.FIGURES / "calibration_valid.png")
    card.to_parquet(config.DATA_PROCESSED / "scorecard_points.parquet", index=False)
    (config.DATA_PROCESSED / "scorecard_model.json").write_text(json.dumps(
        {"intercept": float(fit.params["const"]), "coefficients": {f: float(fit.params[f]) for f in fit.features},
         "factor": factor, "offset": offset, "base_score": 600, "base_odds_good_to_bad": 50, "pdo": 20}, indent=2))

    lines = [
        "# Scorecard report (logistic regression on WoE features)", "",
        "Validation is used to REPORT only; the test split has not been touched. Splits are random and stratified "
        "(the data has no usable time dimension), so performance may be optimistic relative to a true out-of-time test.", "",
        "## Feature selection log (all decisions use training data)",
        df_to_markdown(pd.DataFrame(steps), index=False) if steps else "No features removed.", "",
        "## VIF before the sign/significance pruning", df_to_markdown(vif_before, index=False, float_fmt="{:.2f}"), "",
        "## Final model coefficients (expected negative on WoE features)",
        df_to_markdown(fit.table, index=False, float_fmt="{:.4f}"), "",
        f"Intercept: {fit.params['const']:.4f}. Features kept: {len(fit.features)}. "
        f"Max VIF in final model: {vif_final['vif'].max():.2f}.", "",
        "## Performance", df_to_markdown(comp, index=False, float_fmt="{:.4f}"), "",
        f"Validation AUC 95% bootstrap interval: {lo:.4f} to {hi:.4f} (1,000 resamples, seed {config.SEED}).",
        f"Points scorecard (rounded integers) validation AUC {m_score['auc']:.4f}, KS {m_score['ks']:.4f}; "
        f"largest gap to the unrounded model score: {gap:.2f} points.", "",
        "Rules of thumb for KS and Gini vary by portfolio; compare against a challenger, not a fixed bar.", "",
        "## Calibration by risk bucket (validation)", df_to_markdown(cal, index=False, float_fmt="{:.4f}"), "",
        "![calibration](figures/calibration_valid.png)", "",
        "## Scorecard scaling (illustrative convention)",
        f"Base score 600 at 50:1 good:bad odds, 20 points to double the odds (Factor {factor:.3f}, Offset {offset:.3f}). "
        "These are common textbook choices, not requirements.", "",
        "## Points table", df_to_markdown(card, index=False, float_fmt="{:.4f}"), "",
    ]
    (config.REPORTS / "scorecard_report.md").write_text("\n".join(lines))
    print(df_to_markdown(comp, index=False, float_fmt="{:.4f}"))
    print(f"valid AUC CI {lo:.4f}-{hi:.4f} | features kept {len(fit.features)} | card-vs-model gap {gap:.2f} pts")
    for s in steps:
        print(f"  dropped {s['dropped']}: {s['reason']}")


if __name__ == "__main__":
    main()
