"""Score ONE customer from the saved artifacts (what the demo app calls).

Inputs are the 19 raw behavioural fields. Demographic fields are not needed and are never used.
"""
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from credit_pd_scorecard import config
from credit_pd_scorecard.features import MODEL_INPUT_COLS, prepare
from credit_pd_scorecard.reasons import scorecard_reason_codes

RAW_FIELDS = config.RAW_BEHAVIOR_COLS
# Placeholders required by clean_codes(); the models do not use them.
_UNUSED_DEFAULTS = {"SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35}


@dataclass(frozen=True)
class Artifacts:
    model: dict
    bins: dict
    mean_woe: dict
    cutoffs: dict
    booster: lgb.Booster


@lru_cache(maxsize=2)
def load_artifacts(directory: Path = config.ARTIFACTS) -> Artifacts:
    read = lambda n: json.loads((directory / n).read_text())  # noqa: E731
    return Artifacts(
        model=read("scorecard_model.json"),
        bins=read("woe_bins.json"),
        mean_woe=read("mean_woe.json"),
        cutoffs=read("cutoff.json")["cutoffs"],
        booster=lgb.Booster(model_file=str(directory / "lgbm_model.txt")),
    )


def validate(raw: dict) -> None:
    """Raise ValueError with a clear message for missing, non-numeric or impossible inputs."""
    missing = [f for f in RAW_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for f in RAW_FIELDS:
        v = raw[f]
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v):
            raise ValueError(f"{f} must be a finite number, got {v!r}")
    if raw["LIMIT_BAL"] <= 0:
        raise ValueError("LIMIT_BAL must be positive")
    for f in config.PAY_COLS:
        if raw[f] != int(raw[f]) or not -2 <= raw[f] <= 9:
            raise ValueError(f"{f} must be a whole number from -2 to 9, got {raw[f]}")
    for f in config.PAYAMT_COLS:
        if raw[f] < 0:
            raise ValueError(f"{f} cannot be negative")


def woe_value(value: float, spec: dict) -> float:
    """Same lookup as WoeBinner.transform: first bin whose upper cut exceeds the value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return spec["nan_woe"]
    return spec["woe"][int(np.searchsorted(spec["cuts"], value, side="right"))]


def score_customer(raw: dict, art: Artifacts | None = None) -> dict:
    art = art or load_artifacts()
    validate(raw)
    row = prepare(pd.DataFrame([{**_UNUSED_DEFAULTS, **{f: raw[f] for f in RAW_FIELDS}}]))
    m = art.model

    woe = {f: woe_value(float(row[f.removesuffix("_woe")].iloc[0]), art.bins[f.removesuffix("_woe")]) for f in m["coefficients"]}
    z = m["intercept"] + sum(coef * woe[f] for f, coef in m["coefficients"].items())
    pd_card = 1.0 / (1.0 + math.exp(-z))
    pd_clip = min(max(pd_card, 1e-9), 1 - 1e-9)
    score = m["offset"] + m["factor"] * math.log((1 - pd_clip) / pd_clip)

    codes = scorecard_reason_codes(pd.Series(woe), m["coefficients"], pd.Series(art.mean_woe), m["factor"])
    pd_lgb = float(art.booster.predict(row[MODEL_INPUT_COLS])[0])
    return {
        "pd_scorecard": pd_card,
        "score": score,
        "flag_scorecard": pd_card >= art.cutoffs["scorecard"],
        "pd_lgbm": pd_lgb,
        "flag_lgbm": pd_lgb >= art.cutoffs["LightGBM"],
        "reasons": codes[["reason", "points_vs_average"]].to_dict("records"),
    }
