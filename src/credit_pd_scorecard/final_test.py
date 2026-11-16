"""One-time final evaluation on the locked TEST split, using only FROZEN artifacts.

Nothing is refitted or retuned here. The scorecard, LightGBM model and both cutoffs come from artifacts/
(created from training and validation only). Two modes:

  --rehearsal            run the identical code on the VALIDATION split (writes to data/processed, no marker).
                         Use it to prove the code path works; its numbers must match the earlier validation report.
  --confirm-final-run    run on the TEST split ONCE. Writes reports/final_test_report.md and a marker file
                         reports/TEST_SET_USED.json. Refuses to run again while the marker exists.

Why a marker: every extra look at the test set while making decisions turns it into a second validation
set. After this run, changing the model in response to these numbers would invalidate them.
"""
import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from credit_pd_scorecard import config
from credit_pd_scorecard import scorecard as sc
from credit_pd_scorecard.business import Assumptions, bootstrap_net, evaluate, paired_net_difference
from credit_pd_scorecard.challenger import paired_auc_difference
from credit_pd_scorecard.features import MODEL_INPUT_COLS, prepare
from credit_pd_scorecard.inference import Artifacts, load_artifacts
from credit_pd_scorecard.report_utils import df_to_markdown
from credit_pd_scorecard.split import ids_hash, load_split
from credit_pd_scorecard.stability import psi, psi_label

ARTIFACT_FILES = ["scorecard_model.json", "woe_bins.json", "lgbm_model.txt", "cutoff.json", "mean_woe.json"]
MARKER = config.REPORTS / "TEST_SET_USED.json"


def woe_vector(values: np.ndarray, spec: dict) -> np.ndarray:
    """Vectorised version of inference.woe_value (same bins, same missing-value rule)."""
    values = np.asarray(values, float)
    idx = np.searchsorted(spec["cuts"], np.nan_to_num(values), side="right")
    return np.where(np.isnan(values), spec["nan_woe"], np.asarray(spec["woe"])[idx])


def frozen_predictions(raw: pd.DataFrame, art: Artifacts) -> tuple[np.ndarray, np.ndarray]:
    """PDs from the frozen scorecard and the frozen LightGBM model. No fitting happens here."""
    df = prepare(raw)
    m = art.model
    z = np.full(len(df), m["intercept"], float)
    for f, coef in m["coefficients"].items():
        name = f.removesuffix("_woe")
        z += coef * woe_vector(df[name].to_numpy(), art.bins[name])
    p_card = 1.0 / (1.0 + np.exp(-z))
    p_lgb = art.booster.predict(df[MODEL_INPUT_COLS])
    return p_card, np.asarray(p_lgb)


def evaluate_frozen(raw: pd.DataFrame, art: Artifacts, a: Assumptions | None = None) -> dict:
    a = a or Assumptions()
    y = raw[config.TARGET].to_numpy()
    ead = raw["BILL_AMT1"].clip(lower=0).to_numpy(float)
    p_card, p_lgb = frozen_predictions(raw, art)
    models = {"scorecard": p_card, "LightGBM": p_lgb}
    cut = art.cutoffs
    rows, biz = [], []
    for name, p in models.items():
        lo, hi = sc.auc_ci(y, p)
        rows.append({"model": name, **sc.metrics(y, p), "auc_ci_low": lo, "auc_ci_high": hi})
        r = evaluate(y, p, ead, cut[name], a)
        nlo, nhi = bootstrap_net(y, p, ead, cut[name], a)
        biz.append({"model": name, **r, "net_ci_low": nlo, "net_ci_high": nhi})
    d_auc = paired_auc_difference(y, p_lgb, p_card)
    d_net = paired_net_difference(y, p_lgb, p_card, ead, cut["LightGBM"], cut["scorecard"], a)
    baselines = pd.DataFrame(
        [{"policy": k, **{x: v for x, v in evaluate(y, p_card, ead, c, a).items() if x in ("net_benefit", "treated_share")}}
         for k, c in (("treat nobody", 1.01), ("treat everybody", 0.0))]
    )
    return {
        "rows": len(raw), "defaults": int(y.sum()), "metrics": pd.DataFrame(rows), "business": pd.DataFrame(biz),
        "auc_diff": d_auc, "net_diff": d_net, "baselines": baselines,
        "calibration": {n: sc.calibration_table(y, p) for n, p in models.items()},
        "p_card": p_card, "assumptions": a,
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(test: dict, valid: dict, art: Artifacts) -> str:
    a = test["assumptions"]
    both = pd.concat([valid["metrics"].assign(split="validation"), test["metrics"].assign(split="TEST")]).sort_values(["model", "split"], ascending=[True, False])
    biz = pd.concat([valid["business"].assign(split="validation"), test["business"].assign(split="TEST")]).sort_values(["model", "split"], ascending=[True, False])
    score_psi = psi(valid["p_card"], test["p_card"])
    ad, alo, ahi = test["auc_diff"]
    nd, nlo, nhi = test["net_diff"]
    lines = [
        "# Final test-set report (one-time evaluation)", "",
        f"Test rows: {test['rows']:,} (defaults {test['defaults']:,}). Models, bins, features and cutoffs were FROZEN before this run "
        f"(cutoffs chosen on validation: {json.dumps(art.cutoffs)}). Nothing was refitted or retuned. "
        "The marker file TEST_SET_USED.json records this run; changing the model in response to these numbers would invalidate them.", "",
        "## Discrimination and calibration (validation vs test, same code)",
        df_to_markdown(both[["model", "split", "auc", "gini", "ks", "brier", "auc_ci_low", "auc_ci_high"]], index=False, float_fmt="{:.4f}"), "",
        f"Test AUC difference (LightGBM minus scorecard): {ad:+.4f}, 95% paired-bootstrap interval {alo:+.4f} to {ahi:+.4f}.", "",
        "## Business impact at the frozen cutoffs (ILLUSTRATIVE assumptions)",
        df_to_markdown(pd.DataFrame([vars(a)]), index=False, float_fmt="{:,.2f}"), "",
        df_to_markdown(biz[["model", "split", "cutoff", "net_benefit", "net_ci_low", "net_ci_high", "treated_share", "precision", "recall"]], index=False, float_fmt="{:,.3f}"), "",
        f"Test net benefit difference (LightGBM minus scorecard), NT$: {nd:,.0f}, 95% interval {nlo:,.0f} to {nhi:,.0f}.", "",
        "Baselines on the test customers:", df_to_markdown(test["baselines"], index=False, float_fmt="{:,.3f}"), "",
        "## Score stability, validation to test",
        f"Score PSI: {score_psi:.4f} ({psi_label(score_psi)}). The split is random, so this is expected to be near zero.", "",
        "## Calibration by risk bucket on TEST",
    ]
    for name, cal in test["calibration"].items():
        lines += [f"### {name}", df_to_markdown(cal, index=False, float_fmt="{:.4f}"), ""]
    lines += ["Limitations that still apply: random split (no out-of-time test), one 2005 Taiwanese cohort, illustrative economics.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehearsal", action="store_true", help="run on VALIDATION to check the code path")
    ap.add_argument("--confirm-final-run", action="store_true", help="run on TEST, once")
    args = ap.parse_args(argv)
    if args.rehearsal == args.confirm_final_run:
        ap.print_usage()
        print("Choose exactly one of --rehearsal or --confirm-final-run.")
        return 2

    if args.confirm_final_run and MARKER.exists():
        print(f"Refusing: {MARKER.name} exists, so the test set has already been used. "
              "Delete the marker only if you accept that the result is no longer a one-time test.")
        return 1

    art = load_artifacts(config.ARTIFACTS)
    valid_raw = load_split("valid")
    valid = evaluate_frozen(valid_raw, art)

    if args.rehearsal:
        print("REHEARSAL on the validation split (test set NOT loaded).")
        print(df_to_markdown(valid["metrics"], index=False, float_fmt="{:.4f}"))
        print(df_to_markdown(valid["business"], index=False, float_fmt="{:,.3f}"))
        # Exercise the report writer too (validation stands in for test), so a formatting bug cannot surface on the real run.
        out = config.DATA_PROCESSED / "rehearsal_report.md"
        out.write_text(render(valid, valid, art))
        print(f"Rehearsal report written to {out} (not a result: it is validation data).")
        return 0

    manifest = json.loads((config.DATA_PROCESSED / "split_manifest.json").read_text())
    test_raw = load_split("test", confirm_test=True)
    if ids_hash(test_raw[config.ID_COL]) != manifest["splits"]["test"]["ids_sha256"]:
        print("Refusing: the test split does not match the locked manifest (IDs differ).")
        return 1
    test = evaluate_frozen(test_raw, art)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "final_test_report.md").write_text(render(test, valid, art))
    MARKER.write_text(json.dumps({
        "used_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_rows": test["rows"],
        "test_ids_sha256": manifest["splits"]["test"]["ids_sha256"],
        "artifact_sha256": {n: sha256_of(config.ARTIFACTS / n) for n in ARTIFACT_FILES},
    }, indent=2))
    print("FINAL TEST RESULTS")
    print(df_to_markdown(test["metrics"], index=False, float_fmt="{:.4f}"))
    print(df_to_markdown(test["business"], index=False, float_fmt="{:,.3f}"))
    ad, lo, hi = test["auc_diff"]
    print(f"AUC diff (LightGBM - scorecard) {ad:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"Wrote reports/final_test_report.md and {MARKER.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
