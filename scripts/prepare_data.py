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
import importlib
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


def scaffold_of_smiles(smiles: str) -> str:
    """Return Bemis-Murcko scaffold for a SMILES; empty if not available."""
    try:
        murcko_module = importlib.import_module("rdkit.Chem.Scaffolds")
        MurckoScaffold = getattr(murcko_module, "MurckoScaffold")
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return ""


def scaffold_train_test_split(df: pd.DataFrame, test_size: float, random_seed: int):
    """Scaffold split: keep scaffolds separated between train/test."""
    scaffold_to_indices = {}
    for idx, smi in enumerate(df["smiles"].tolist()):
        scaffold = scaffold_of_smiles(smi)
        if not scaffold:
            scaffold = f"NO_SCAFFOLD_{idx}"
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    scaffold_groups = list(scaffold_to_indices.values())
    # Randomize tie order for reproducibility with a seed.
    rng = np.random.RandomState(random_seed)
    rng.shuffle(scaffold_groups)
    # Place larger scaffolds first.
    scaffold_groups.sort(key=len, reverse=True)

    train_target = int(len(df) * (1.0 - test_size))
    train_indices = []
    test_indices = []

    for group in scaffold_groups:
        if len(train_indices) + len(group) <= train_target:
            train_indices.extend(group)
        else:
            test_indices.extend(group)

    # Edge case: if split collapses, fall back to random split.
    if len(train_indices) == 0 or len(test_indices) == 0:
        return train_test_split(df, test_size=test_size, random_state=random_seed)

    train_df = df.iloc[train_indices].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    test_df = df.iloc[test_indices].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return train_df, test_df


def save_scale_subsets(train_df: pd.DataFrame, prefix: str, split_name: str):
    """Save 20/50/80/100 subsets for one train split."""
    rng = np.random.RandomState(RANDOM_SEED)
    n_train = len(train_df)

    for frac, label in [(0.2, "20"), (0.5, "50"), (0.8, "80")]:
        n = max(10, int(n_train * frac))
        idx = rng.choice(n_train, size=n, replace=False)
        subset = train_df.iloc[idx].reset_index(drop=True)
        split_subset_path = os.path.join(DATA_DIR, f"{prefix}_{split_name}_train_{label}.csv")
        subset.to_csv(split_subset_path, index=False)
        print(f"Saved {split_name} train_{label}%: {split_subset_path} ({len(subset)} rows)")

        # Keep backward-compatible filenames for random split.
        if split_name == "random":
            legacy_subset_path = os.path.join(DATA_DIR, f"{prefix}_train_{label}.csv")
            subset.to_csv(legacy_subset_path, index=False)
            print(f"Saved legacy train_{label}%: {legacy_subset_path} ({len(subset)} rows)")

    split_train_100 = os.path.join(DATA_DIR, f"{prefix}_{split_name}_train_100.csv")
    train_df.to_csv(split_train_100, index=False)
    print(f"Saved {split_name} train_100%: {split_train_100} ({len(train_df)} rows)")

    if split_name == "random":
        legacy_train_100 = os.path.join(DATA_DIR, f"{prefix}_train_100.csv")
        train_df.to_csv(legacy_train_100, index=False)
        print(f"Saved legacy train_100%: {legacy_train_100} ({len(train_df)} rows)")


def create_splits(df: pd.DataFrame, dataset_name: str, split_name: str):
    """Create train/test and data-scale splits for one split strategy."""
    os.makedirs(DATA_DIR, exist_ok=True)
    prefix = DATASETS[dataset_name]["prefix"]

    # Save full dataset
    full_path = os.path.join(DATA_DIR, f"{prefix}.csv")
    df.to_csv(full_path, index=False)
    print(f"Saved full dataset: {full_path} ({len(df)} rows)")

    # 80/20 split
    if split_name == "random":
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    elif split_name == "scaffold":
        train_df, test_df = scaffold_train_test_split(df, test_size=0.2, random_seed=RANDOM_SEED)
    else:
        raise ValueError(f"Unsupported split strategy: {split_name}")

    train_path = os.path.join(DATA_DIR, f"{prefix}_{split_name}_train.csv")
    test_path = os.path.join(DATA_DIR, f"{prefix}_{split_name}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved test:  {test_path} ({len(test_df)} rows)")

    # Keep backward-compatible random split filenames.
    if split_name == "random":
        legacy_train = os.path.join(DATA_DIR, f"{prefix}_train.csv")
        legacy_test = os.path.join(DATA_DIR, f"{prefix}_test.csv")
        train_df.to_csv(legacy_train, index=False)
        test_df.to_csv(legacy_test, index=False)
        print(f"Saved legacy train: {legacy_train} ({len(train_df)} rows)")
        print(f"Saved legacy test:  {legacy_test} ({len(test_df)} rows)")

    save_scale_subsets(train_df, prefix, split_name)

    # Print summary stats
    print("\n=== Dataset Summary ===")
    print(f"Full dataset: {len(df)} molecules")
    print(f"Split mode:   {split_name}")
    print(f"Training set: {len(train_df)} molecules")
    print(f"Test set:     {len(test_df)} molecules")
    print(f"Target range: [{df['target'].min():.3f}, {df['target'].max():.3f}]")
    print(f"Target mean:  {df['target'].mean():.3f}")
    print(f"Target std:   {df['target'].std():.3f}")


def prepare_dataset(dataset_name: str, split_name: str):
    """Download, parse, and split one dataset."""
    config = DATASETS[dataset_name]
    print("=" * 60)
    print(f"{config['display_name']} Dataset Preparation for Chemprop ({split_name})")
    print("=" * 60)

    raw = download_csv(dataset_name)
    df = parse_dataset(raw, dataset_name)
    create_splits(df, dataset_name, split_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ESOL and Lipophilicity datasets")
    parser.add_argument(
        "--dataset",
        choices=["esol", "lipophilicity", "lipo", "all"],
        default="all",
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--split",
        choices=["random", "scaffold", "both"],
        default="both",
        help="How to split train/test sets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_arg = "lipophilicity" if args.dataset == "lipo" else args.dataset
    datasets = DATASETS.keys() if dataset_arg == "all" else [dataset_arg]
    split_modes = ["random", "scaffold"] if args.split == "both" else [args.split]
    for dataset in datasets:
        for split_mode in split_modes:
            prepare_dataset(dataset, split_mode)

    print("\nDone! Data is ready in:", DATA_DIR)
