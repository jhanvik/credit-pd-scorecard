"""Choose a PD cutoff by expected business value, with every assumption stated.

THE ACTION: customers whose predicted PD is at or above the cutoff get a proactive risk action
(outreach plus a credit-limit review). Customers below it are left alone.

THE ECONOMICS (ALL NUMBERS ARE ILLUSTRATIVE ASSUMPTIONS, not facts about any real bank):
  * A treated customer who really would default: the action avoids `effect` of the loss.
        loss = LGD * exposure, exposure = the latest bill (BILL_AMT1, floored at 0).
  * Every treated customer costs `outreach_cost`.
  * A treated customer who would NOT have defaulted costs an extra `false_alarm_cost`
    (friction, lost spending). Amounts are in NT dollars like the data.
Net benefit = loss avoided - costs, versus doing nothing (= 0). It uses REALIZED outcomes on the
sample, so it needs no assumption that the PDs are well calibrated.

The cutoff is chosen on VALIDATION and then frozen. Reporting the same validation number is
optimistic (we picked its best point); the honest number comes from the one-time test run later.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_pd_scorecard import config
from credit_pd_scorecard.report_utils import df_to_markdown


@dataclass(frozen=True)
class Assumptions:
    lgd: float = 0.60
    effect: float = 0.20
    outreach_cost: float = 100.0
    false_alarm_cost: float = 1_500.0


def contributions(y: np.ndarray, p: np.ndarray, ead: np.ndarray, cutoff: float, a: Assumptions) -> np.ndarray:
    """Net benefit contributed by each customer at this cutoff (0 for customers not treated)."""
    treat = p >= cutoff
    value = y * a.effect * a.lgd * ead - a.outreach_cost - (1 - y) * a.false_alarm_cost
    return np.where(treat, value, 0.0)


def evaluate(y: np.ndarray, p: np.ndarray, ead: np.ndarray, cutoff: float, a: Assumptions) -> dict:
    treat = p >= cutoff
    tp = int((treat & (y == 1)).sum())
    return {
        "cutoff": cutoff,
        "net_benefit": float(contributions(y, p, ead, cutoff, a).sum()),
        "treated_share": float(treat.mean()),
        "precision": tp / treat.sum() if treat.sum() else float("nan"),
        "recall": tp / (y == 1).sum(),
    }


def sweep(y, p, ead, a: Assumptions, cutoffs=None) -> pd.DataFrame:
    cutoffs = np.round(np.arange(0.05, 0.96, 0.01), 2) if cutoffs is None else cutoffs
    return pd.DataFrame([evaluate(y, p, ead, c, a) for c in cutoffs])


def best_cutoff(table: pd.DataFrame) -> float:
    """Cutoff with the highest net benefit (the lowest one if there is a tie)."""
    return float(table.loc[table["net_benefit"].idxmax(), "cutoff"])


def bootstrap_net(y, p, ead, cutoff, a, n_boot=1000, seed=config.SEED) -> tuple[float, float]:
    """95% interval for total net benefit when the validation customers are resampled."""
    c = contributions(y, p, ead, cutoff, a)
    rng = np.random.default_rng(seed)
    sums = [c[rng.integers(0, len(c), len(c))].sum() for _ in range(n_boot)]
    return float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))


def paired_net_difference(y, p_a, p_b, ead, cut_a, cut_b, a, n_boot=1000, seed=config.SEED):
    """Net benefit of model A minus model B, resampling the same customers for both."""
    ca, cb = contributions(y, p_a, ead, cut_a, a), contributions(y, p_b, ead, cut_b, a)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ca), len(ca))
        d.append(ca[idx].sum() - cb[idx].sum())
    return float((ca - cb).sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from credit_pd_scorecard.split import load_split

    preds = pd.read_parquet(config.DATA_PROCESSED / "valid_predictions.parquet")
    valid = load_split("valid")
    assert (valid[config.ID_COL].to_numpy() == preds[config.ID_COL].to_numpy()).all(), "row order mismatch"
    y = preds[config.TARGET].to_numpy()
    ead = valid["BILL_AMT1"].clip(lower=0).to_numpy(float)
    models = {"scorecard": preds["pd_scorecard"].to_numpy(), "LightGBM": preds["pd_lgbm"].to_numpy()}
    a = Assumptions()

    sweeps = {m: sweep(y, p, ead, a) for m, p in models.items()}
    chosen = {m: best_cutoff(t) for m, t in sweeps.items()}
    rows = []
    for m, p in models.items():
        r = evaluate(y, p, ead, chosen[m], a)
        lo, hi = bootstrap_net(y, p, ead, chosen[m], a)
        rows.append({"model": m, **r, "net_ci_low": lo, "net_ci_high": hi})
    summary = pd.DataFrame(rows)
    diff, dlo, dhi = paired_net_difference(y, models["LightGBM"], models["scorecard"], ead, chosen["LightGBM"], chosen["scorecard"], a)

    base = {"treat nobody": evaluate(y, models["scorecard"], ead, 1.01, a), "treat everybody": evaluate(y, models["scorecard"], ead, 0.0, a)}
    base_t = pd.DataFrame([{"policy": k, **v} for k, v in base.items()]).drop(columns=["cutoff", "precision"])

    sens = []
    for eff in (0.10, 0.20, 0.30):
        for fa in (500.0, 1_500.0, 3_000.0):
            sa = Assumptions(effect=eff, false_alarm_cost=fa)
            t = sweep(y, models["scorecard"], ead, sa)
            c = best_cutoff(t)
            sens.append({"effect": eff, "false_alarm_cost": fa, "best_cutoff": c, "net_benefit": float(t.loc[t["cutoff"] == c, "net_benefit"].iloc[0])})
    sens = pd.DataFrame(sens)

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (m, t), color in zip(sweeps.items(), ("#2a78d6", "#eb6834")):
        ax.plot(t["cutoff"], t["net_benefit"] / 1e6, color=color, lw=2, label=m)
        ax.plot([chosen[m]], [t.loc[t["cutoff"] == chosen[m], "net_benefit"].iloc[0] / 1e6], "o", color=color, ms=7, mec="#fcfcfb", mew=2)
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xlabel("PD cutoff (treat customers at or above)", color="#52514e")
    ax.set_ylabel("net benefit, NT$ million (validation)", color="#52514e")
    ax.set_title("Net benefit by PD cutoff (illustrative assumptions)", loc="left", fontsize=11, color="#0b0b0b")
    ax.grid(color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "net_benefit_curve.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    (config.DATA_PROCESSED / "cutoff.json").write_text(json.dumps({"cutoffs": chosen, "assumptions": asdict(a), "chosen_on": "validation"}, indent=2))
    lines = [
        "# Cutoff and business impact", "",
        "## Assumptions (ILLUSTRATIVE, chosen by the analyst, not facts about any real bank)",
        df_to_markdown(pd.DataFrame([asdict(a)]), index=False, float_fmt="{:,.2f}"), "",
        "Exposure = latest bill (BILL_AMT1, floored at 0). Amounts in NT$. Net benefit uses REALIZED outcomes on the validation sample, "
        "versus doing nothing. See the module docstring in business.py for the full logic.", "",
        "## Chosen cutoffs (selected on validation, then frozen for the one-time test run)",
        df_to_markdown(summary, index=False, float_fmt="{:,.3f}"), "",
        f"Paired difference in net benefit (LightGBM minus scorecard), NT$: {diff:,.0f}, 95% bootstrap interval {dlo:,.0f} to {dhi:,.0f}.", "",
        "These validation figures are optimistic: the cutoff was chosen where validation looks best.", "",
        "## Simple baselines on the same customers", df_to_markdown(base_t, index=False, float_fmt="{:,.3f}"), "",
        "## Sensitivity of the scorecard's best cutoff to the assumptions", df_to_markdown(sens, index=False, float_fmt="{:,.2f}"), "",
        "![net benefit](figures/net_benefit_curve.png)", "",
    ]
    (config.REPORTS / "business_impact.md").write_text("\n".join(lines))
    print(df_to_markdown(summary, index=False, float_fmt="{:,.3f}"))
    print(df_to_markdown(base_t, index=False, float_fmt="{:,.3f}"))
    print(f"LightGBM minus scorecard net: {diff:,.0f} [{dlo:,.0f}, {dhi:,.0f}]")
    print(df_to_markdown(sens, index=False, float_fmt="{:,.2f}"))


if __name__ == "__main__":
    main()
