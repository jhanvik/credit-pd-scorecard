"""Stratified train / validation / test split, saved once and then reused.

Why split first: every choice that "learns" from data (bin edges, feature
selection, hyper-parameters) must see only the training rows. Otherwise the
validation and test scores are optimistic. The test set is locked: it can only
be loaded with an explicit `confirm_test=True`.
"""
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_pd_scorecard import config
from credit_pd_scorecard.load_data import load_raw
from credit_pd_scorecard.report_utils import df_to_markdown


def make_splits(df: pd.DataFrame, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Split into train/valid/test with the default rate preserved in each part."""
    fractions = config.SPLIT_FRACTIONS
    holdout = fractions["valid"] + fractions["test"]
    train, rest = train_test_split(
        df, test_size=holdout, stratify=df[config.TARGET], random_state=seed
    )
    valid, test = train_test_split(
        rest,
        test_size=fractions["test"] / holdout,
        stratify=rest[config.TARGET],
        random_state=seed,
    )
    splits = {"train": train, "valid": valid, "test": test}
    result = {k: v.sort_values(config.ID_COL).reset_index(drop=True) for k, v in splits.items()}
    _check_splits(df, result)
    return result


def _check_splits(df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if the split lost rows, duplicated rows, or overlapped."""
    all_ids = pd.concat([s[config.ID_COL] for s in splits.values()])
    if len(all_ids) != len(df) or all_ids.duplicated().any():
        raise ValueError("Split lost rows or an ID appears in more than one split")
    if set(all_ids) != set(df[config.ID_COL]):
        raise ValueError("Split does not cover exactly the original IDs")


def ids_hash(ids: pd.Series) -> str:
    """Fingerprint of a set of IDs, so we can prove the test set never changed."""
    joined = ",".join(str(i) for i in sorted(ids.tolist()))
    return hashlib.sha256(joined.encode()).hexdigest()


def count_feature_overlap(train: pd.DataFrame, other: pd.DataFrame) -> int:
    """Rows in `other` whose feature values exactly match some training row.

    Identical customers on both sides of a split make scores a little optimistic.
    """
    cols = [c for c in train.columns if c not in (config.ID_COL, config.TARGET)]
    keys = train[cols].drop_duplicates()
    merged = other[cols].merge(keys, on=cols, how="left", indicator=True)
    return int((merged["_merge"] == "both").sum())


def build_manifest(splits: dict[str, pd.DataFrame], seed: int) -> dict:
    manifest = {"seed": seed, "fractions": config.SPLIT_FRACTIONS, "splits": {}}
    for name, part in splits.items():
        manifest["splits"][name] = {
            "rows": len(part),
            "defaults": int(part[config.TARGET].sum()),
            "default_rate": float(part[config.TARGET].mean()),
            "ids_sha256": ids_hash(part[config.ID_COL]),
        }
    manifest["rows_matching_train_features"] = {
        name: count_feature_overlap(splits["train"], splits[name]) for name in ("valid", "test")
    }
    return manifest


def render_markdown(manifest: dict) -> str:
    rows = [
        {
            "split": name,
            "rows": info["rows"],
            "defaults": info["defaults"],
            "default_rate": info["default_rate"],
            "ids_sha256 (first 12)": info["ids_sha256"][:12],
        }
        for name, info in manifest["splits"].items()
    ]
    overlap = manifest["rows_matching_train_features"]
    return "\n".join(
        [
            "# Train / validation / test split",
            "",
            f"- Method: stratified on the target, seed {manifest['seed']}, "
            f"fractions {manifest['fractions']}",
            "- The test set is locked: `load_split('test')` refuses unless `confirm_test=True`.",
            "- The fingerprint (SHA-256 of the sorted IDs) lets anyone check the test set never changed.",
            "",
            df_to_markdown(pd.DataFrame(rows), index=False, float_fmt="{:.4f}"),
            "",
            "Rows whose feature values exactly match a training row "
            f"(small optimism risk): validation {overlap['valid']}, test {overlap['test']}.",
            "",
        ]
    )


def write_splits(seed: int = config.SEED) -> dict:
    """Run the split on the raw data and save parquet files plus a manifest."""
    splits = make_splits(load_raw(), seed)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(config.DATA_PROCESSED / f"{name}.parquet", index=False)
    manifest = build_manifest(splits, seed)
    (config.DATA_PROCESSED / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "split_summary.md").write_text(render_markdown(manifest))
    return manifest


def load_split(name: str, confirm_test: bool = False) -> pd.DataFrame:
    """Load a saved split. The test set needs an explicit confirmation."""
    if name not in config.SPLIT_FRACTIONS:
        raise ValueError(f"Unknown split {name!r}; use one of {list(config.SPLIT_FRACTIONS)}")
    if name == "test" and not confirm_test:
        raise PermissionError(
            "The test set is locked. Pass confirm_test=True only for the final evaluation."
        )
    return pd.read_parquet(config.DATA_PROCESSED / f"{name}.parquet")


def main() -> None:
    manifest = write_splits()
    for name, info in manifest["splits"].items():
        print(f"{name}: {info['rows']:,} rows, default rate {info['default_rate']:.4f}")
    print(f"Saved splits to {config.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
