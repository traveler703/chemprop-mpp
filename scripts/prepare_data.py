"""
prepare_data.py — Download and prepare MoleculeNet regression datasets.

Output (saved to ../data/):
  - {dataset}.csv            Full dataset: smiles, target
  - {dataset}_train.csv      80% train split
  - {dataset}_test.csv       20% test split (fixed by random seed)
  - {dataset}_train_20.csv   20% of training data
  - {dataset}_train_50.csv   50% of training data
  - {dataset}_train_80.csv   80% of training data
  - {dataset}_train_100.csv  100% of training data
"""

import argparse
from io import StringIO
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RANDOM_SEED = 42

ESOL_URLS = [
    "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv",
]

LIPOPHILICITY_URLS = [
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
]


DATASETS = {
    "esol": {
        "display_name": "ESOL",
        "prefix": "esol",
        "urls": ESOL_URLS,
        "target_keywords": [
            ("measured", "log"),
            ("log", "solub"),
            ("log", "s"),
        ],
    },
    "lipophilicity": {
        "display_name": "Lipophilicity",
        "prefix": "lipo",
        "urls": LIPOPHILICITY_URLS,
        "target_keywords": [
            ("exp",),
            ("logd",),
            ("target",),
        ],
    },
}


def download_csv(dataset_name: str) -> str:
    """Download raw CSV text from the first available URL for a dataset."""
    config = DATASETS[dataset_name]
    for url in config["urls"]:
        try:
            print(f"Trying: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            print(f"  Success! ({len(raw)} bytes)")
            return raw
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError(f"Could not download {config['display_name']} dataset from any source.")


def find_smiles_column(df: pd.DataFrame) -> str:
    """Detect SMILES column by name."""
    for col in df.columns:
        if "smiles" in col.lower():
            return col
    raise ValueError(f"Cannot find SMILES column in {list(df.columns)}")


def find_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    """Detect dataset target column from known MoleculeNet column names."""
    config = DATASETS[dataset_name]
    for keywords in config["target_keywords"]:
        for col in df.columns:
            col_lower = col.lower()
            if all(keyword in col_lower for keyword in keywords):
                if dataset_name == "esol" and "esol" in col_lower:
                    continue
                return col

    numeric_cols = [
        col for col in df.columns
        if col != find_smiles_column(df) and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Cannot find target column for {config['display_name']} in {list(df.columns)}.\n"
        "First 5 rows:\n" + str(df.head())
    )


def parse_dataset(raw_csv: str, dataset_name: str) -> pd.DataFrame:
    """Parse raw CSV into a clean DataFrame with columns [smiles, target]."""
    df = pd.read_csv(StringIO(raw_csv))

    print(f"Raw columns: {list(df.columns)}")
    print(f"Raw shape: {df.shape}")

    smiles_col = find_smiles_column(df)
    target_col = find_target_column(df, dataset_name)

    print(f"Using SMILES column: '{smiles_col}'")
    print(f"Using target column: '{target_col}'")

    result = pd.DataFrame({
        "smiles": df[smiles_col].astype(str).str.strip(),
        "target": pd.to_numeric(df[target_col], errors="coerce"),
    })

    # Drop rows with missing values
    before = len(result)
    result = result.dropna()
    print(f"Dropped {before - len(result)} rows with missing values")

    return result.reset_index(drop=True)


def create_splits(df: pd.DataFrame, dataset_name: str):
    """Create train/test and data-scale splits."""
    os.makedirs(DATA_DIR, exist_ok=True)
    prefix = DATASETS[dataset_name]["prefix"]

    # Save full dataset
    full_path = os.path.join(DATA_DIR, f"{prefix}.csv")
    df.to_csv(full_path, index=False)
    print(f"Saved full dataset: {full_path} ({len(df)} rows)")

    # Fixed train/test split (80/20)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED
    )

    train_path = os.path.join(DATA_DIR, f"{prefix}_train.csv")
    test_path = os.path.join(DATA_DIR, f"{prefix}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved test:  {test_path} ({len(test_df)} rows)")

    # Data-scale subsets (20%, 50%, 80% of training data)
    # Use fixed random seed for reproducibility
    rng = np.random.RandomState(RANDOM_SEED)
    n_train = len(train_df)

    for frac, label in [(0.2, "20"), (0.5, "50"), (0.8, "80")]:
        n = max(10, int(n_train * frac))  # at least 10 samples
        idx = rng.choice(n_train, size=n, replace=False)
        subset = train_df.iloc[idx].reset_index(drop=True)

        out_path = os.path.join(DATA_DIR, f"{prefix}_train_{label}.csv")
        subset.to_csv(out_path, index=False)
        print(f"Saved train_{label}%: {out_path} ({len(subset)} rows)")

    # Also save full training set as 100% reference
    train_100_path = os.path.join(DATA_DIR, f"{prefix}_train_100.csv")
    train_df.to_csv(train_100_path, index=False)
    print(f"Saved train_100%: {train_100_path} ({len(train_df)} rows)")

    # Print summary stats
    print("\n=== Dataset Summary ===")
    print(f"Full dataset: {len(df)} molecules")
    print(f"Training set: {len(train_df)} molecules")
    print(f"Test set:     {len(test_df)} molecules")
    print(f"Target range: [{df['target'].min():.3f}, {df['target'].max():.3f}]")
    print(f"Target mean:  {df['target'].mean():.3f}")
    print(f"Target std:   {df['target'].std():.3f}")


def prepare_dataset(dataset_name: str):
    """Download, parse, and split one dataset."""
    config = DATASETS[dataset_name]
    print("=" * 60)
    print(f"{config['display_name']} Dataset Preparation for Chemprop")
    print("=" * 60)

    raw = download_csv(dataset_name)
    df = parse_dataset(raw, dataset_name)
    create_splits(df, dataset_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ESOL and Lipophilicity datasets")
    parser.add_argument(
        "--dataset",
        choices=["esol", "lipophilicity", "all"],
        default="all",
        help="Dataset to prepare",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prepare_dataset(dataset)

    print("\nDone! Data is ready in:", DATA_DIR)
