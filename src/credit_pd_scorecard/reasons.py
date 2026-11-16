"""Plain-English reason codes for the scorecard (light module: no SHAP/matplotlib import)."""
import pandas as pd

REASON_TEXT = {
    "PAY_1": "Latest repayment status (Sept)", "PAY_2": "Repayment status 2 months ago", "PAY_3": "Repayment status 3 months ago",
    "PAY_4": "Repayment status 4 months ago", "PAY_5": "Repayment status 5 months ago", "PAY_6": "Repayment status 6 months ago",
    "max_delay": "Longest payment delay in 6 months", "n_delay_months": "Number of late months", "delay_sum_recent3": "Recent (3-month) delinquency",
    "LIMIT_BAL": "Low credit limit", "util_avg": "High average credit utilization", "util_1": "High latest utilization",
    "util_max": "High peak utilization", "pay_ratio_1": "Low latest payment relative to bill", "pay_ratio_avg": "Low average payment relative to bill",
    "pay_to_limit": "Low payments relative to limit", "bill_trend": "Rising bill balance",
    **{f"PAY_AMT{i}": f"Low payment amount {i} month(s) ago" for i in range(1, 7)},
    **{f"BILL_AMT{i}": f"Bill amount {i} month(s) ago" for i in range(1, 7)},
}


def scorecard_reason_codes(woe_row: pd.Series, coefs: dict[str, float], mean_woe: pd.Series, factor: float, top_k: int = 4) -> pd.DataFrame:
    """Points this customer is BELOW the average customer, per feature (largest shortfall first).

    points_i - average_points_i = -factor * b_i * (woe_i - mean_woe_i); coefficients are negative, so a
    below-average WoE (riskier than average) gives a negative difference.
    """
    rows = []
    for f, b in coefs.items():
        diff = -factor * b * (woe_row[f] - mean_woe[f])
        rows.append({"feature": f.removesuffix("_woe"), "points_vs_average": float(diff)})
    t = pd.DataFrame(rows).sort_values("points_vs_average")
    t = t[t["points_vs_average"] < 0].head(top_k).reset_index(drop=True)
    t["reason"] = t["feature"].map(lambda c: REASON_TEXT.get(c, c))
    return t
