"""Load the UCI 'Default of Credit Card Clients' data and save a clean copy.

Only renames columns and checks the shape. No values are changed here:
every cleaning decision is made later, on purpose, and written down.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "credit_raw_renamed.parquet"
EXPECTED_SHAPE = (30_000, 25)


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the Excel file (the real header is on its 2nd row) and rename two columns."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path, header=1, engine="calamine")
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected shape {EXPECTED_SHAPE}, got {df.shape}")
    return df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "DEFAULT"})


def main() -> None:
    df = load_raw()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Default rate: {df['DEFAULT'].mean():.4f}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
