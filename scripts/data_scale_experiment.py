"""
data_scale_experiment.py — Train Chemprop on data subsets for ESOL and Lipophilicity.

Produces:
  - ../results/data_scale_results.json
  - ../results/data_scale_results_esol.json
  - ../results/data_scale_results_lipo.json
"""

import os
import sys
import json
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCALES = [20, 50, 80, 100]
N_REPEATS = 3
BASE_SEED = 42

DATASETS = {
    "esol": {
        "display_name": "ESOL Solubility",
        "prefix": "esol",
        "test_csv": os.path.join(DATA_DIR, "esol_test.csv"),
    },
    "lipo": {
        "display_name": "Lipophilicity",
        "prefix": "lipo",
        "test_csv": os.path.join(DATA_DIR, "lipo_test.csv"),
    },
}


def run_training(train_csv: str, test_csv: str, seed: int, output_json: str) -> dict:
    """Run train.py for one configuration and return results."""
    train_script = os.path.join(PROJECT_ROOT, "scripts", "train.py")
    cmd = [
        sys.executable, train_script,
        "--train-data", train_csv,
        "--test-data", test_csv,
        "--epochs", "100",
        "--seed", str(seed),
        "--output", output_json,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr}")
        return None
    with open(output_json) as f:
        return json.load(f)


def summarize_results(all_results: dict) -> dict:
    """Print and return aggregate metrics for one dataset."""
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Scale':>8s}  {'RMSE_mean':>10s}  {'RMSE_std':>10s}  {'R²_mean':>8s}")
    print(f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")

    summary = {}
    for scale_label, runs in all_results.items():
        if not runs:
            print(f"{scale_label:>8s}  {'n/a':>10s}  {'n/a':>10s}  {'n/a':>8s}")
            summary[scale_label] = {
                "rmse_mean": None,
                "rmse_std": None,
                "r2_mean": None,
                "runs": runs,
            }
            continue

        rmses = [r["rmse"] for r in runs]
        r2s = [r["r2"] for r in runs]
        rmse_mean = np.mean(rmses)
        rmse_std = np.std(rmses)
        r2_mean = np.mean(r2s)
        print(f"{scale_label:>8s}  {rmse_mean:10.4f}  {rmse_std:10.4f}  {r2_mean:8.4f}")
        summary[scale_label] = {
            "rmse_mean": float(rmse_mean),
            "rmse_std": float(rmse_std),
            "r2_mean": float(r2_mean),
            "runs": runs,
        }
    return summary


def run_dataset(dataset_key: str, config: dict) -> dict:
    print("=" * 60)
    print(f"Data Scale Experiment — {config['display_name']}")
    print("=" * 60)
    print(f"Scales: {SCALES}% of training data")
    print(f"Repeats per scale: {N_REPEATS}")
    print(f"Output dir: {RESULTS_DIR}")
    print()

    all_results = {}

    for scale in SCALES:
        print(f"\n{'─'*40}")
        print(f"Scale: {scale}% training data")
        print(f"{'─'*40}")

        train_csv = os.path.join(DATA_DIR, f"{config['prefix']}_train_{scale}.csv")

        if not os.path.exists(train_csv) or not os.path.exists(config["test_csv"]):
            print("  WARNING: Dataset files not found. Running prepare_data.py first...")
            subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "scripts", "prepare_data.py")],
                           cwd=PROJECT_ROOT)

        scale_runs = []
        for rep in range(N_REPEATS):
            seed = BASE_SEED + rep * 100
            output_json = os.path.join(
                RESULTS_DIR, f"results_{dataset_key}_scale{scale}_rep{rep}.json"
            )
            print(f"\n  Repeat {rep+1}/{N_REPEATS} (seed={seed})")

            res = run_training(train_csv, config["test_csv"], seed, output_json)
            if res:
                scale_runs.append({
                    "rmse": res["rmse"],
                    "mae": res["mae"],
                    "r2": res["r2"],
                    "n_train": res["n_train"],
                    "seed": seed,
                })
                print(f"    RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, R²={res['r2']:.4f}")

        all_results[f"{scale}%"] = scale_runs

    summary = summarize_results(all_results)
    output_path = os.path.join(RESULTS_DIR, f"data_scale_results_{dataset_key}.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_path}")
    return summary


def main():
    combined_summary = {}
    for dataset_key, config in DATASETS.items():
        combined_summary[dataset_key] = run_dataset(dataset_key, config)

    output_path = os.path.join(RESULTS_DIR, "data_scale_results.json")
    with open(output_path, "w") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
