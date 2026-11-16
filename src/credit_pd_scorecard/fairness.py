"""Fairness sanity check: how do the models and the cutoff rule behave across demographic groups?

SEX, EDUCATION, MARRIAGE and AGE are NOT model inputs. They are used here only to audit outcomes.
Not using a protected attribute does NOT guarantee fair outcomes (other features can act as proxies),
which is exactly why we audit.

For each group, at the cutoff chosen in business.py, we report:
  default_rate  observed share who defaulted        mean_pd / calib_ratio  predicted vs observed
  flag_rate     share who receive the risk action   tpr  share of true defaulters flagged
  fpr           share of non-defaulters flagged     precision  share of flagged who default
Gaps compare each group with the largest group in the same attribute, with bootstrap intervals.

CAVEATS: base default rates genuinely differ between groups, so equal flag rates are not expected.
Which fairness metric matters, and what legal rules apply (for example to age or sex, and in which
country), is a compliance question: verify with the applicable regulations. No legal threshold is claimed.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_pd_scorecard import config
from credit_pd_scorecard.report_utils import df_to_markdown

MIN_GROUP = 200  # below this the estimates are too noisy to read much into
LABELS = {
    "SEX": {1: "male", 2: "female"},
    "EDUCATION": {1: "graduate school", 2: "university", 3: "high school", 4: "other"},
    "MARRIAGE": {1: "married", 2: "single", 3: "other"},
}


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"])


def _rates(y: np.ndarray, flag: np.ndarray) -> dict:
    pos, neg = y == 1, y == 0
    return {
        "flag_rate": float(flag.mean()),
        "tpr": float(flag[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(flag[neg].mean()) if neg.any() else float("nan"),
        "precision": float(y[flag].mean()) if flag.any() else float("nan"),
    }


def group_metrics(y: np.ndarray, p: np.ndarray, flag: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for g in groups.dropna().unique():
        m = (groups == g).to_numpy()
        auc = roc_auc_score(y[m], p[m]) if len(np.unique(y[m])) == 2 else float("nan")
        rows.append({
            "group": str(g), "n": int(m.sum()), "default_rate": float(y[m].mean()), "mean_pd": float(p[m].mean()),
            "calib_ratio": float(p[m].mean() / y[m].mean()) if y[m].mean() > 0 else float("nan"),
            "auc": float(auc), **_rates(y[m], flag[m]),
        })
    t = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    t["reliable"] = np.where(t["n"] >= MIN_GROUP, "yes", "too small")
    return t


def bootstrap_gap(y: np.ndarray, flag: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, metric: str, n_boot: int = 500, seed: int = config.SEED):
    """metric(group A) - metric(group B), with a 95% interval; each group is resampled on its own."""
    rng = np.random.default_rng(seed)
    ya, fa, yb, fb = y[mask_a], flag[mask_a], y[mask_b], flag[mask_b]
    point = _rates(ya, fa)[metric] - _rates(yb, fb)[metric]
    diffs = []
    for _ in range(n_boot):
        ia, ib = rng.integers(0, len(ya), len(ya)), rng.integers(0, len(yb), len(yb))
        diffs.append(_rates(ya[ia], fa[ia])[metric] - _rates(yb[ib], fb[ib])[metric])
    return float(point), float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def gap_table(y: np.ndarray, flag: np.ndarray, groups: pd.Series, metrics=("flag_rate", "tpr", "fpr")) -> pd.DataFrame:
    """Each reliable group versus the largest group of the same attribute."""
    sizes = groups.value_counts()
    ref = sizes.index[0]
    rows = []
    for g in sizes.index[1:]:
        if sizes[g] < MIN_GROUP:
            continue
        row = {"group": str(g), "versus": str(ref)}
        for metric in metrics:
            d, lo, hi = bootstrap_gap(y, flag, (groups == g).to_numpy(), (groups == ref).to_numpy(), metric)
            row[f"{metric}_gap"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    from credit_pd_scorecard.features import prepare
    from credit_pd_scorecard.split import load_split

    valid = prepare(load_split("valid"))
    preds = pd.read_parquet(config.DATA_PROCESSED / "valid_predictions.parquet")
    assert (valid[config.ID_COL].to_numpy() == preds[config.ID_COL].to_numpy()).all(), "row order mismatch"
    cutoffs = json.loads((config.DATA_PROCESSED / "cutoff.json").read_text())["cutoffs"]
    y = preds[config.TARGET].to_numpy()
    models = {"scorecard": preds["pd_scorecard"].to_numpy(), "LightGBM": preds["pd_lgbm"].to_numpy()}
    attrs = {
        "SEX": valid["SEX"].map(LABELS["SEX"]), "EDUCATION": valid["EDUCATION"].map(LABELS["EDUCATION"]),
        "MARRIAGE": valid["MARRIAGE"].map(LABELS["MARRIAGE"]), "AGE": age_group(valid["AGE"]),
    }

    lines = [
        "# Fairness sanity check (validation split)", "",
        "Protected-type attributes are NOT model inputs; they are used only to audit outcomes. Read the caveats in "
        "`fairness.py`: base rates differ by group, no legal threshold is claimed, and the rules that apply are a compliance "
        f"question to verify. Groups with fewer than {MIN_GROUP} customers are marked too small to interpret.", "",
    ]
    for model, p in models.items():
        flag = p >= cutoffs[model]
        lines += [f"## {model} (cutoff {cutoffs[model]:.2f}, flags {flag.mean():.1%} of customers)", ""]
        for name, groups in attrs.items():
            lines += [f"### {name}", df_to_markdown(group_metrics(y, p, flag, groups), index=False, float_fmt="{:.3f}"), ""]
            gaps = gap_table(y, flag, groups)
            if len(gaps):
                lines += ["Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:", df_to_markdown(gaps, index=False), ""]
    (config.REPORTS / "fairness_report.md").write_text("\n".join(lines))

    for model, p in models.items():
        flag = p >= cutoffs[model]
        print(f"== {model} (cutoff {cutoffs[model]:.2f}) ==")
        for name, groups in attrs.items():
            t = group_metrics(y, p, flag, groups)
            print(f"-- {name}")
            print(t[["group", "n", "default_rate", "flag_rate", "tpr", "fpr", "calib_ratio", "reliable"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
