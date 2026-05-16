"""
baseline_rf.py — RDKit descriptors + Random Forest regression baseline.

Usage examples:
    python scripts/baseline_rf.py --dataset esol --split random
    python scripts/baseline_rf.py --dataset lipo --split scaffold
    python scripts/baseline_rf.py --dataset all --split all
"""

import argparse
import json
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)


DATASETS = {
    "esol": {
        "display_name": "ESOL",
        "prefix": "esol",
    },
    "lipophilicity": {
        "display_name": "Lipophilicity",
        "prefix": "lipo",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run RDKit+RF baseline on molecular property datasets")
    parser.add_argument("--dataset", choices=["esol", "lipophilicity", "lipo", "all"], default="all")
    parser.add_argument("--split", choices=["random", "scaffold", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    return parser.parse_args()


def resolve_target_column(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    if "logS" in df.columns:
        return "logS"
    raise ValueError(f"Cannot find target column in {list(df.columns)}")


def descriptor_names() -> List[str]:
    return [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "FractionCSP3",
    ]


def calc_descriptors(smiles: str) -> List[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_names())
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feats = [calc_descriptors(smi) for smi in df["smiles"].astype(str).tolist()]
    return np.asarray(feats, dtype=float)


def load_split(dataset_key: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prefix = DATASETS[dataset_key]["prefix"]
    train_path = os.path.join(DATA_DIR, f"{prefix}_{split}_train.csv")
    test_path = os.path.join(DATA_DIR, f"{prefix}_{split}_test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Missing split files: {train_path} or {test_path}. "
            "Please run prepare_data.py first."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def run_rf(dataset_key: str, split: str, seed: int, n_estimators: int) -> dict:
    train_df, test_df = load_split(dataset_key, split)
    target_col = resolve_target_column(train_df)
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' missing in test set.")

    x_train = build_feature_matrix(train_df)
    x_test = build_feature_matrix(test_df)
    y_train = train_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    rf = model.named_steps["rf"]
    importances = {
        name: float(val)
        for name, val in sorted(
            zip(descriptor_names(), rf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    results = {
        "dataset": DATASETS[dataset_key]["display_name"],
        "dataset_key": dataset_key,
        "split": split,
        "model": "RandomForestRegressor",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "feature_importance": importances,
        "predictions": [float(v) for v in y_pred],
        "targets": [float(v) for v in y_test],
    }
    return results


def save_result_json(result: dict):
    dataset_key = result["dataset_key"]
    split = result["split"]
    filename = f"rf_{dataset_key}_{split}_results.json"
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    # Backward-compatible outputs expected by report notes (random split).
    if split == "random":
        legacy_path = os.path.join(RESULTS_DIR, f"rf_{dataset_key}_results.json")
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {legacy_path}")


def load_chemprop_metric(dataset_key: str, split: str):
    prefix = DATASETS[dataset_key]["prefix"]
    candidate_paths = [
        os.path.join(MODELS_DIR, f"results_{prefix}_{split}_train.json"),
    ]
    if split == "random":
        candidate_paths.extend([
            os.path.join(MODELS_DIR, f"results_{prefix}_train.json"),
            os.path.join(MODELS_DIR, f"results_{prefix}_train_100.json"),
        ])

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def make_model_comparison_plot(results: List[dict]):
    rows = []
    for rf_res in results:
        dataset_key = rf_res["dataset_key"]
        split = rf_res["split"]
        rows.append({
            "dataset": dataset_key,
            "split": split,
            "model": "Random Forest",
            "rmse": rf_res["rmse"],
        })
        chemprop_res = load_chemprop_metric(dataset_key, split)
        if chemprop_res is not None:
            rows.append({
                "dataset": dataset_key,
                "split": split,
                "model": "Chemprop MPNN",
                "rmse": chemprop_res["rmse"],
            })

    if not rows:
        print("No results available for model comparison plot.")
        return

    plot_df = pd.DataFrame(rows)
    labels = [f"{d.upper()}-{s}" for d, s in plot_df[["dataset", "split"]].drop_duplicates().to_records(index=False)]
    order = plot_df[["dataset", "split"]].drop_duplicates().to_records(index=False).tolist()

    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 2.2), 5))

    rf_vals = []
    cp_vals = []
    for dataset_key, split in order:
        rf_row = plot_df[
            (plot_df["dataset"] == dataset_key)
            & (plot_df["split"] == split)
            & (plot_df["model"] == "Random Forest")
        ]
        cp_row = plot_df[
            (plot_df["dataset"] == dataset_key)
            & (plot_df["split"] == split)
            & (plot_df["model"] == "Chemprop MPNN")
        ]
        rf_vals.append(float(rf_row["rmse"].iloc[0]) if not rf_row.empty else np.nan)
        cp_vals.append(float(cp_row["rmse"].iloc[0]) if not cp_row.empty else np.nan)

    ax.bar(x - width / 2, rf_vals, width, label="Random Forest")
    ax.bar(x + width / 2, cp_vals, width, label="Chemprop MPNN")
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison: Chemprop vs Random Forest")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    dataset_arg = "lipophilicity" if args.dataset == "lipo" else args.dataset
    datasets = list(DATASETS.keys()) if dataset_arg == "all" else [dataset_arg]
    splits = ["random", "scaffold"] if args.split == "all" else [args.split]

    all_results = []
    for dataset_key in datasets:
        for split in splits:
            print(f"\n=== RF baseline: {DATASETS[dataset_key]['display_name']} ({split}) ===")
            try:
                result = run_rf(dataset_key, split, seed=args.seed, n_estimators=args.n_estimators)
                save_result_json(result)
                all_results.append(result)
                print(
                    f"RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, "
                    f"R2={result['r2']:.4f}, n_train={result['n_train']}"
                )
            except Exception as e:
                print(f"FAILED: {e}")

    make_model_comparison_plot(all_results)


if __name__ == "__main__":
    main()
