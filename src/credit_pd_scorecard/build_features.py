"""Phase 3 build: engineered features -> WoE bins (fit on TRAIN only) -> IV report.

Order matters, and it is the heart of avoiding leakage:
  1. prepare() every split (row-wise rules, never reads the target).
  2. Learn bin edges and WoE values from the TRAIN rows only.
  3. Apply those frozen bins to validation. The test split is not touched here.
IV is also measured on validation with the same bins; a big drop signals overfit bins.
"""
import json

import pandas as pd

from credit_pd_scorecard import config
from credit_pd_scorecard.features import MODEL_INPUT_COLS, prepare
from credit_pd_scorecard.report_utils import df_to_markdown
from credit_pd_scorecard.split import load_split
from credit_pd_scorecard.woe import fit_all, iv_strength, transform_all

# PAY_x codes are ordered categories, not a smooth scale: keep their natural shape.
OVERRIDES = {c: {"monotonic": None} for c in config.PAY_COLS}
CORR_LIMIT = 0.85


def high_corr_pairs(woe_train: pd.DataFrame, summary: pd.DataFrame, limit: float = CORR_LIMIT) -> pd.DataFrame:
    """Pairs of WoE features that are nearly duplicates (matters for logistic regression)."""
    iv = dict(zip(summary["feature"], summary["iv"]))
    corr = woe_train.corr().abs()
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] > limit:
                rows.append({"feature_a": a.removesuffix("_woe"), "feature_b": b.removesuffix("_woe"), "abs_corr": corr.loc[a, b],
                             "higher_iv": max((a, b), key=lambda c: iv[c.removesuffix("_woe")]).removesuffix("_woe")})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "abs_corr", "higher_iv"]).sort_values("abs_corr", ascending=False)


def main() -> None:
    train = prepare(load_split("train"))
    valid = prepare(load_split("valid"))

    binners, summary = fit_all(train, MODEL_INPUT_COLS, config.TARGET, overrides=OVERRIDES)
    summary["iv_valid"] = [binners[f].iv_on(valid[f], valid[config.TARGET]) for f in summary["feature"]]
    summary["iv_drop"] = summary["iv"] - summary["iv_valid"]
    summary["valid_strength"] = summary["iv_valid"].map(iv_strength)

    woe_train = transform_all(train, binners)
    woe_valid = transform_all(valid, binners)
    pairs = high_corr_pairs(woe_train, summary)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, base, woe in (("train", train, woe_train), ("valid", valid, woe_valid)):
        out = pd.concat([base[[config.ID_COL, config.TARGET]], woe], axis=1)
        out.to_parquet(config.DATA_PROCESSED / f"woe_{name}.parquet", index=False)

    bins = {f: b.bin_table_.assign(feature=f) for f, b in binners.items()}
    pd.concat(bins.values(), ignore_index=True)[["feature", "bin", "count", "default_rate", "woe"]].to_parquet(
        config.DATA_PROCESSED / "woe_bin_tables.parquet", index=False
    )
    (config.DATA_PROCESSED / "woe_bins.json").write_text(json.dumps(
        {f: {"cuts": [float(c) for c in b.cuts_], "woe": [float(w) for w in b.woe_], "nan_woe": float(b.nan_woe_)}
         for f, b in binners.items()}, indent=2))

    lines = [
        "# WoE / IV report (bins fitted on the training split only)", "",
        "IV bands (<0.02 not useful, <0.1 weak, <0.3 medium, <0.5 strong, above = suspicious) are "
        "conventions from credit-scoring practice, not laws.", "",
        "`iv_valid` re-measures IV on the validation split using the SAME bins. A large `iv_drop` means the bins "
        "were fitted to noise.", "",
        "## How to read the 'suspiciously strong' flags",
        "The delinquency features score above IV 0.5. The leakage screen (leakage_table.md) found no single feature "
        "above AUC 0.75, and the strength comes from a small group of customers who are already 2+ months late and "
        "default at a very high rate (see the PAY_1 bin table). That is a plausible behavioral signal, so the flag is "
        "kept as a WARNING, not a removal. The residual risk is timing: the UCI text does not say how the latest "
        "status (PAY_1) relates to the outcome month. This goes in the README limitations.", "",
        "## Information Value by feature", df_to_markdown(summary, index=False, float_fmt="{:.4f}"), "",
        f"## Near-duplicate features (|corr| > {CORR_LIMIT} on WoE values, train)",
        df_to_markdown(pairs, index=False, float_fmt="{:.3f}") if len(pairs) else "None.", "",
        "Keep the higher-IV feature of each pair, or the more business-sensible one; decide in the model phase.", "",
        "## Bin tables (top 8 features by IV)",
    ]
    for f in summary["feature"].head(8):
        lines += [f"### {f}", df_to_markdown(bins[f].drop(columns=["feature"]), index=False, float_fmt="{:.4f}"), ""]
    (config.REPORTS / "iv_report.md").write_text("\n".join(lines))

    print(summary.head(10)[["feature", "iv", "iv_valid", "strength"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Features with IV >= 0.02: {int((summary['iv'] >= 0.02).sum())} of {len(summary)}; max IV {summary['iv'].max():.3f}")
    print(f"Near-duplicate pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
