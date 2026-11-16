"""Copy the small files the demo needs into artifacts/ (committed to git).

data/ is git-ignored (it holds the raw dataset), so a hosted demo or a Docker image could not find the
model files there. These artifacts contain model parameters only, no customer rows.
"""
import json
import shutil

import pandas as pd

from credit_pd_scorecard import config

FILES = ["scorecard_model.json", "woe_bins.json", "lgbm_model.txt", "cutoff.json"]


def main() -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy(config.DATA_PROCESSED / name, config.ARTIFACTS / name)
    model = json.loads((config.DATA_PROCESSED / "scorecard_model.json").read_text())
    woe_tr = pd.read_parquet(config.DATA_PROCESSED / "woe_train.parquet")
    mean_woe = {f: float(woe_tr[f].mean()) for f in model["coefficients"]}
    (config.ARTIFACTS / "mean_woe.json").write_text(json.dumps(mean_woe, indent=2))
    print(f"Wrote {len(FILES) + 1} files to {config.ARTIFACTS}")


if __name__ == "__main__":
    main()
