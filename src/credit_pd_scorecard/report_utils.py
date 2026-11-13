"""Tiny helper to write DataFrames as markdown tables without extra dependencies."""
import numpy as np
import pandas as pd


def _fmt(value: object, float_fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return float_fmt.format(float(value))
    return str(value).replace("|", "/")


def df_to_markdown(df: pd.DataFrame, index: bool = True, float_fmt: str = "{:,.3f}") -> str:
    """Return a GitHub-style markdown table for `df`."""
    frame = df.reset_index() if index else df
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(_fmt(v, float_fmt) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])
