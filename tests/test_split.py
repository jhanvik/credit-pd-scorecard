"""Tests for the stratified split and the locked test set. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from credit_pd_scorecard import config, split
from credit_pd_scorecard.report_utils import df_to_markdown


def make_df(n: int = 10_000, rate: float = 0.2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            config.ID_COL: np.arange(1, n + 1),
            "x": rng.normal(size=n),
            config.TARGET: (rng.uniform(size=n) < rate).astype(int),
        }
    )


def test_sizes_and_stratification():
    df = make_df()
    parts = split.make_splits(df, seed=1)
    assert [len(parts[k]) for k in ("train", "valid", "test")] == [6_000, 2_000, 2_000]
    for part in parts.values():
        assert part[config.TARGET].mean() == pytest.approx(df[config.TARGET].mean(), abs=0.002)


def test_splits_are_disjoint_and_cover_every_row():
    df = make_df()
    parts = split.make_splits(df)
    ids = np.concatenate([p[config.ID_COL].to_numpy() for p in parts.values()])
    assert len(ids) == len(df) and len(set(ids)) == len(df)


def test_same_seed_same_split_and_different_seed_differs():
    df = make_df()
    a = split.make_splits(df, seed=7)["test"][config.ID_COL].tolist()
    b = split.make_splits(df, seed=7)["test"][config.ID_COL].tolist()
    c = split.make_splits(df, seed=8)["test"][config.ID_COL].tolist()
    assert a == b and a != c


def test_ids_hash_ignores_order_and_detects_change():
    ids = pd.Series([3, 1, 2])
    assert split.ids_hash(ids) == split.ids_hash(pd.Series([1, 2, 3]))
    assert split.ids_hash(ids) != split.ids_hash(pd.Series([1, 2, 4]))


def test_feature_overlap_counts_exact_matches_only():
    train = pd.DataFrame({config.ID_COL: [1, 2], "x": [1, 2], "y": [5, 6], config.TARGET: [0, 1]})
    other = pd.DataFrame({config.ID_COL: [3, 4], "x": [1, 9], "y": [5, 9], config.TARGET: [1, 0]})
    assert split.count_feature_overlap(train, other) == 1


def test_test_set_is_locked_unless_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    parts = split.make_splits(make_df())
    for name, part in parts.items():
        part.to_parquet(tmp_path / f"{name}.parquet", index=False)
    assert len(split.load_split("train")) == 6_000
    with pytest.raises(PermissionError):
        split.load_split("test")
    assert len(split.load_split("test", confirm_test=True)) == 2_000
    with pytest.raises(ValueError):
        split.load_split("nonsense")


def test_markdown_table_helper():
    table = df_to_markdown(pd.DataFrame({"a": [1, 2000], "b": [0.5, np.nan]}), index=False)
    assert table.splitlines()[0] == "| a | b |"
    assert "| 2,000 |  |" in table
