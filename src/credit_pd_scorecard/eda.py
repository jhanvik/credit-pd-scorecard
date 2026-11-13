"""Exploratory data analysis on the TRAINING split only.

Why train only: if you study validation/test rows to decide what to build, those rows
have quietly influenced the model and are no longer a clean exam. Every number and chart
here comes from `load_split("train")`.
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_pd_scorecard import config
from credit_pd_scorecard.features import prepare
from credit_pd_scorecard.report_utils import df_to_markdown
from credit_pd_scorecard.split import load_split

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"


def rate_table(df: pd.DataFrame, group: pd.Series, name: str) -> pd.DataFrame:
    """Customers and default rate per group."""
    g = df.groupby(group, observed=True)[config.TARGET].agg(customers="size", default_rate="mean")
    g.index.name = name
    return g.reset_index()


def decile_group(s: pd.Series, q: int = 10) -> pd.Series:
    """Quantile buckets; duplicates='drop' copes with piles of identical values."""
    return pd.qcut(s, q=q, duplicates="drop")


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 29, 39, 49, 59, 200], labels=["<30", "30-39", "40-49", "50-59", "60+"])


def _bar(ax, labels, rates, overall, title, xlabel):
    ax.bar(range(len(rates)), rates, color=BLUE, width=0.7)
    ax.axhline(overall, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(rates)), labels, fontsize=8, rotation=0)
    ax.set_title(f"{title}\n(dashed line = overall {overall:.1%})", loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel("default rate", fontsize=9, color=MUTED)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_rate_chart(tbl: pd.DataFrame, col: str, overall: float, title: str, xlabel: str, fname: str, wide=False):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9 if wide else 7, 4))
    labels = [f"D{i + 1}" for i in range(len(tbl))] if "decile" in col else [str(v) for v in tbl[col]]
    _bar(ax, labels, tbl["default_rate"].to_numpy(), overall, title, xlabel)
    for i, (n, r) in enumerate(zip(tbl["customers"], tbl["default_rate"])):
        ax.text(i, r, f"n={n:,}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylim(0, float(tbl["default_rate"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(config.FIGURES / fname, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    train = prepare(load_split("train"))
    overall = float(train[config.TARGET].mean())
    config.REPORTS.mkdir(parents=True, exist_ok=True)

    pay1 = rate_table(train, train["PAY_1"], "PAY_1")
    limit = rate_table(train, decile_group(train["LIMIT_BAL"]), "LIMIT_BAL decile")
    util = rate_table(train, decile_group(train["util_avg"]), "util_avg decile")
    ndelay = rate_table(train, train["n_delay_months"], "months delayed (of 6)")
    demo = {
        "SEX": rate_table(train, train["SEX"], "SEX"),
        "EDUCATION": rate_table(train, train["EDUCATION"], "EDUCATION"),
        "MARRIAGE": rate_table(train, train["MARRIAGE"], "MARRIAGE"),
        "AGE band": rate_table(train, age_band(train["AGE"]), "AGE band"),
    }

    save_rate_chart(pay1, "PAY_1", overall, "Default rate by latest repayment status (PAY_1)",
                    "PAY_1: -2/-1/0 = paid or revolving, 1+ = months late", "eda_pay1.png")
    save_rate_chart(limit, "LIMIT_BAL decile", overall, "Default rate by credit limit decile",
                    "LIMIT_BAL decile (NT$, low to high)", "eda_limit.png", wide=True)
    save_rate_chart(util, "util_avg decile", overall, "Default rate by 6-month average utilization decile",
                    "util_avg decile (bill / limit, low to high)", "eda_util.png", wide=True)
    save_rate_chart(ndelay, "months delayed (of 6)", overall, "Default rate by number of late months",
                    "months with PAY >= 1", "eda_delays.png")

    # Missingness and outliers on the engineered columns (train only).
    num = train.drop(columns=[config.TARGET, config.ID_COL])
    desc = num.describe(percentiles=[0.01, 0.5, 0.99]).T[["mean", "min", "1%", "50%", "99%", "max"]]
    desc.insert(0, "missing", num.isna().sum())
    corr = train[[*config.RAW_BEHAVIOR_COLS, "util_avg", "pay_ratio_avg", "max_delay"]].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, bool), k=1)).stack().sort_values(ascending=False)
    pairs = upper[upper > 0.9].rename("abs_corr").reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_corr"]

    lines = [
        "# EDA summary (training split only)", "",
        f"Rows: {len(train):,}. Overall default rate: {overall:.2%}.", "",
        "Use these default rates as a baseline: a segment is only interesting if it differs from the overall rate "
        "AND has enough customers (see the `customers` column).", "",
        "## Default rate by latest repayment status", df_to_markdown(pay1, index=False, float_fmt="{:.3f}"), "",
        "Decile charts label buckets D1 (lowest) to D10 (highest); exact ranges are in the tables.", "",
        "## By credit-limit decile", df_to_markdown(limit, index=False, float_fmt="{:.3f}"), "",
        "## By average utilization decile", df_to_markdown(util, index=False, float_fmt="{:.3f}"), "",
        "## By number of late months", df_to_markdown(ndelay, index=False, float_fmt="{:.3f}"), "",
        "## Demographic baselines (NOT model inputs; for the later fairness check)",
    ]
    for name, t in demo.items():
        lines += [f"### {name}", df_to_markdown(t, index=False, float_fmt="{:.3f}"), ""]
    lines += [
        "## Distribution and outliers", df_to_markdown(desc, float_fmt="{:,.2f}"), "",
        "## Highly correlated pairs (|corr| > 0.9)",
        df_to_markdown(pairs, index=False, float_fmt="{:.3f}") if len(pairs) else "None.", "",
        "Correlated bill columns are expected (a balance carries over month to month). "
        "A logistic regression needs these handled later (VIF check, keep one per group).", "",
    ]
    (config.REPORTS / "eda_summary.md").write_text("\n".join(lines))
    print(f"EDA on {len(train):,} train rows; default rate {overall:.2%}. Figures in {config.FIGURES}")


if __name__ == "__main__":
    main()
