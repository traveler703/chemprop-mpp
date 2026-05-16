"""
visualize.py — Generate publication-quality figures for the workshop.

Produces (in ../results/):
  1. predicted_vs_true.png      — Scatter plot of predictions vs true values
  2. error_distribution.png     — Histogram of prediction errors
  3. data_scale_rmse.png        — RMSE vs training data size with error bars
  4. training_curves.png        — Training/validation loss curves
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
BLUE = "#3274a1"
ORANGE = "#e1812c"
GREEN = "#3a923a"

DATASETS = {
    "esol": {
        "display_name": "ESOL",
        "result_prefix": "esol",
        "target_label": "LogS",
        "task_name": "Solubility",
    },
    "lipo": {
        "display_name": "Lipophilicity",
        "result_prefix": "lipo",
        "target_label": "logD",
        "task_name": "Lipophilicity",
    },
}


def output_path(filename: str, dataset_key: str) -> str:
    """Keep original ESOL filenames and add lipo_ prefix for Lipophilicity."""
    if dataset_key == "esol":
        return os.path.join(RESULTS_DIR, filename)
    return os.path.join(RESULTS_DIR, f"{dataset_key}_{filename}")


def load_baseline_results():
    """Load the 100% training results."""
    path = os.path.join(MODELS_DIR, "results_esol_train.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Try scale experiment file
    path2 = os.path.join(RESULTS_DIR, "results_scale100_rep0.json")
    if os.path.exists(path2):
        with open(path2) as f:
            return json.load(f)
    raise FileNotFoundError("No baseline results found. Run train.py first.")


def figure_1_predicted_vs_true():
    """Scatter plot: predicted vs true LogS."""
    results = load_baseline_results()
    y_true = np.array(results["targets"])
    y_pred = np.array(results["predictions"])
    rmse = results["rmse"]
    r2 = results["r2"]
    mae = results["mae"]

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Hexbin for density
    hb = ax.hexbin(y_true, y_pred, gridsize=25, cmap="Blues", mincnt=1, alpha=0.85)

    # Identity line
    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect prediction")

    # Regression line
    slope, intercept, r_value, _, _ = stats.linregress(y_true, y_pred)
    ax.plot(lims, slope * np.array(lims) + intercept, color=ORANGE, linewidth=2,
            label=f"Fit (slope={slope:.2f})")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Measured LogS (mol/L)")
    ax.set_ylabel("Predicted LogS (mol/L)")
    ax.set_title("Chemprop MPNN: Predicted vs Measured Solubility")

    # Metrics box
    textstr = f"RMSE = {rmse:.3f}\nMAE  = {mae:.3f}\nR²   = {r2:.3f}\nn = {len(y_true)}"
    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props, fontfamily="monospace")

    # Colorbar
    cbar = plt.colorbar(hb, ax=ax, shrink=0.85)
    cbar.set_label("Density")

    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "predicted_vs_true.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def figure_2_error_distribution():
    """Histogram of prediction errors."""
    results = load_baseline_results()
    y_true = np.array(results["targets"])
    y_pred = np.array(results["predictions"])
    errors = y_pred - y_true

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(errors, bins=30, edgecolor="white", color=BLUE, alpha=0.8, density=True)

    # Normal fit
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=ORANGE, linewidth=2.5,
            label=f"Normal fit (μ={mu:.3f}, σ={sigma:.3f})")

    ax.axvline(0, color="black", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Prediction Error (Predicted − Measured LogS)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Prediction Errors")
    ax.legend()

    textstr = f"Mean error = {mu:.3f}\nStd error  = {sigma:.3f}\nMAE        = {np.mean(np.abs(errors)):.3f}"
    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props, fontfamily="monospace")

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "error_distribution.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def figure_3_data_scale():
    """RMSE vs training data fraction with error bars."""
    scale_path = os.path.join(RESULTS_DIR, "data_scale_results.json")
    if not os.path.exists(scale_path):
        print(f"WARNING: {scale_path} not found. Run data_scale_experiment.py first.")
        # Try to construct from individual files
        scales = [20, 50, 80, 100]
        summary = {}
        for s in scales:
            runs = []
            for rep in range(3):
                p = os.path.join(RESULTS_DIR, f"results_scale{s}_rep{rep}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        r = json.load(f)
                    runs.append({"rmse": r["rmse"], "r2": r["r2"], "n_train": r["n_train"]})
            if runs:
                summary[f"{s}%"] = {
                    "rmse_mean": float(np.mean([r["rmse"] for r in runs])),
                    "rmse_std": float(np.std([r["rmse"] for r in runs])),
                    "r2_mean": float(np.mean([r["r2"] for r in runs])),
                    "runs": runs,
                }
    else:
        with open(scale_path) as f:
            summary = json.load(f)

    if not summary:
        print("No data scale results found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    scale_labels = list(summary.keys())
    scale_nums = [int(l.replace("%", "")) for l in scale_labels]

    # Get n_train for x-axis
    n_train_vals = []
    rmse_means = []
    rmse_stds = []
    r2_means = []
    for label in scale_labels:
        runs = summary[label]["runs"]
        n_train_vals.append(runs[0]["n_train"] if runs else 0)
        rmse_means.append(summary[label]["rmse_mean"])
        rmse_stds.append(summary[label]["rmse_std"])
        r2_means.append(summary[label]["r2_mean"])

    # Sort by n_train
    order = np.argsort(n_train_vals)
    n_train_vals = [n_train_vals[i] for i in order]
    rmse_means = [rmse_means[i] for i in order]
    rmse_stds = [rmse_stds[i] for i in order]
    r2_means = [r2_means[i] for i in order]
    sorted_labels = [scale_labels[i] for i in order]

    # RMSE subplot
    ax1.errorbar(n_train_vals, rmse_means, yerr=rmse_stds,
                 marker="o", markersize=10, linewidth=2.5, capsize=6,
                 color=BLUE, markeredgecolor="white", markeredgewidth=1.5,
                 label="Chemprop MPNN")
    ax1.set_xlabel("Number of Training Molecules")
    ax1.set_ylabel("RMSE (LogS)")
    ax1.set_title("RMSE vs Training Data Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Annotate points
    for i, (n, rmse, lbl) in enumerate(zip(n_train_vals, rmse_means, sorted_labels)):
        ax1.annotate(lbl, (n, rmse), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=9, color="gray")

    # R² subplot
    ax2.plot(n_train_vals, r2_means, marker="s", markersize=10, linewidth=2.5,
             color=ORANGE, markeredgecolor="white", markeredgewidth=1.5)
    ax2.set_xlabel("Number of Training Molecules")
    ax2.set_ylabel("R²")
    ax2.set_title("R² vs Training Data Size")
    ax2.grid(True, alpha=0.3)
    for i, (n, r2, lbl) in enumerate(zip(n_train_vals, r2_means, sorted_labels)):
        ax2.annotate(lbl, (n, r2), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=9, color="gray")

    fig.suptitle("Effect of Training Data Size on Model Performance", fontsize=16, y=1.02)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "data_scale_rmse.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def figure_4_training_curves():
    """Training and validation loss curves."""
    results = load_baseline_results()
    train_losses = results.get("train_losses", [])
    val_losses = results.get("val_losses", [])

    if not train_losses:
        print("No loss history in results.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=BLUE, linewidth=1.5, alpha=0.7, label="Training Loss")
    ax.plot(epochs, val_losses, color=ORANGE, linewidth=2, label="Validation Loss")

    # Mark best epoch
    best_epoch = results.get("best_epoch", len(val_losses))
    if best_epoch <= len(val_losses):
        best_val = val_losses[best_epoch - 1]
        ax.scatter(best_epoch, best_val, color=ORANGE, s=80, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(f"Best (epoch {best_epoch})",
                    (best_epoch, best_val),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color=ORANGE)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title("Training and Validation Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def figure_5_molecule_examples():
    """Bar chart of selected molecule predictions."""
    # Famous molecules and their SMILES
    examples = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Benzene", "c1ccccc1"),
        ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("Paracetamol", "CC(=O)NC1=CC=C(C=C1)O"),
        ("Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O"),
    ]

    # We'll make a simple table figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    col_labels = ["Molecule", "Formula", "Predicted LogS", "Solubility"]
    # We'll need actual predictions — for now leave placeholder
    # Real values will be filled by the notebook demo
    cell_text = [
        ["Aspirin", "C₉H₈O₄", "−2.14", "Low (poorly soluble)"],
        ["Caffeine", "C₈H₁₀N₄O₂", "−1.52", "Moderate"],
        ["Benzene", "C₆H₆", "−1.73", "Low (hydrophobic)"],
        ["Ibuprofen", "C₁₃H₁₈O₂", "−3.59", "Very low"],
        ["Paracetamol", "C₈H₉NO₂", "−1.35", "Moderate"],
        ["Glucose", "C₆H₁₂O₆", "0.52", "High (very soluble)"],
    ]

    cases_path = os.path.join(RESULTS_DIR, "molecule_cases.json")
    if os.path.exists(cases_path):
        with open(cases_path) as f:
            cases = json.load(f)[:6]
        cell_text = [
            [item["name"], item["formula"], f'{item["predicted_logS"]:.2f}', item["solubility"]]
            for item in cases
        ]
    else:
        print(f"WARNING: {cases_path} not found. Run molecule_cases.py first.")
        cell_text = [
            ["Aspirin", "C9H8O4", "n/a", "n/a"],
            ["Caffeine", "C8H10N4O2", "n/a", "n/a"],
            ["Benzene", "C6H6", "n/a", "n/a"],
            ["Ibuprofen", "C13H18O2", "n/a", "n/a"],
            ["Paracetamol", "C8H9NO2", "n/a", "n/a"],
            ["Glucose", "C6H12O6", "n/a", "n/a"],
        ]

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", loc="center",
                     colWidths=[0.18, 0.22, 0.25, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for i in range(len(col_labels)):
        table[0, i].set_facecolor(BLUE)
        table[0, i].set_text_props(color="white", fontweight="bold")

    ax.set_title("Example Solubility Predictions", fontsize=15, pad=20)

    path = os.path.join(RESULTS_DIR, "molecule_examples.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def load_baseline_results_for_dataset(dataset_key: str, config: dict):
    """Load baseline results for one dataset."""
    path = os.path.join(MODELS_DIR, f"results_{config['result_prefix']}_train.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    scale_path = os.path.join(RESULTS_DIR, f"results_{dataset_key}_scale100_rep0.json")
    if os.path.exists(scale_path):
        with open(scale_path) as f:
            return json.load(f)

    # Backward-compatible fallback for older ESOL runs.
    if dataset_key == "esol":
        legacy_path = os.path.join(RESULTS_DIR, "results_scale100_rep0.json")
        if os.path.exists(legacy_path):
            with open(legacy_path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"No {config['display_name']} baseline results found. Run train.py first."
    )


def figure_predicted_vs_true(dataset_key: str, config: dict):
    """Scatter plot for one dataset."""
    results = load_baseline_results_for_dataset(dataset_key, config)
    y_true = np.array(results["targets"])
    y_pred = np.array(results["predictions"])
    rmse = results["rmse"]
    r2 = results["r2"]
    mae = results["mae"]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=25, cmap="Blues", mincnt=1, alpha=0.85)

    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect prediction")

    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    ax.plot(lims, slope * np.array(lims) + intercept, color=ORANGE, linewidth=2,
            label=f"Fit (slope={slope:.2f})")

    target_label = config["target_label"]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Measured {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(f"Chemprop MPNN: {config['display_name']} Prediction")

    textstr = f"RMSE = {rmse:.3f}\nMAE  = {mae:.3f}\nR²   = {r2:.3f}\nn = {len(y_true)}"
    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props, fontfamily="monospace")

    cbar = plt.colorbar(hb, ax=ax, shrink=0.85)
    cbar.set_label("Density")

    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = output_path("predicted_vs_true.png", dataset_key)
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def figure_error_distribution(dataset_key: str, config: dict):
    """Prediction error histogram for one dataset."""
    results = load_baseline_results_for_dataset(dataset_key, config)
    y_true = np.array(results["targets"])
    y_pred = np.array(results["predictions"])
    errors = y_pred - y_true

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(errors, bins=30, edgecolor="white", color=BLUE, alpha=0.8, density=True)

    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=ORANGE, linewidth=2.5,
            label=f"Normal fit (μ={mu:.3f}, σ={sigma:.3f})")

    ax.axvline(0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel(f"Prediction Error (Predicted − Measured {config['target_label']})")
    ax.set_ylabel("Density")
    ax.set_title(f"{config['display_name']} Prediction Error Distribution")
    ax.legend()

    textstr = f"Mean error = {mu:.3f}\nStd error  = {sigma:.3f}\nMAE        = {np.mean(np.abs(errors)):.3f}"
    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props, fontfamily="monospace")

    fig.tight_layout()
    path = output_path("error_distribution.png", dataset_key)
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def load_scale_summary(dataset_key: str):
    """Load data-scale summary for one dataset."""
    path = os.path.join(RESULTS_DIR, f"data_scale_results_{dataset_key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    combined_path = os.path.join(RESULTS_DIR, "data_scale_results.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            combined = json.load(f)
        if dataset_key in combined:
            return combined[dataset_key]
        if dataset_key == "esol":
            return combined

    print(f"WARNING: {path} not found. Run data_scale_experiment.py first.")
    return {}


def figure_data_scale(dataset_key: str, config: dict):
    """RMSE/R2 vs training data size for one dataset."""
    summary = load_scale_summary(dataset_key)
    if not summary:
        print(f"No {config['display_name']} data scale results found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    scale_labels = list(summary.keys())
    n_train_vals = []
    rmse_means = []
    rmse_stds = []
    r2_means = []
    clean_labels = []
    for label in scale_labels:
        item = summary[label]
        runs = item.get("runs", [])
        if not runs or item.get("rmse_mean") is None:
            continue
        n_train_vals.append(runs[0]["n_train"])
        rmse_means.append(item["rmse_mean"])
        rmse_stds.append(item["rmse_std"])
        r2_means.append(item["r2_mean"])
        clean_labels.append(label)

    if not clean_labels:
        print(f"No usable {config['display_name']} data scale results found.")
        return

    order = np.argsort(n_train_vals)
    n_train_vals = [n_train_vals[i] for i in order]
    rmse_means = [rmse_means[i] for i in order]
    rmse_stds = [rmse_stds[i] for i in order]
    r2_means = [r2_means[i] for i in order]
    clean_labels = [clean_labels[i] for i in order]

    ax1.errorbar(n_train_vals, rmse_means, yerr=rmse_stds,
                 marker="o", markersize=10, linewidth=2.5, capsize=6,
                 color=BLUE, markeredgecolor="white", markeredgewidth=1.5,
                 label="Chemprop MPNN")
    ax1.set_xlabel("Number of Training Molecules")
    ax1.set_ylabel(f"RMSE ({config['target_label']})")
    ax1.set_title("RMSE vs Training Data Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for n, rmse, lbl in zip(n_train_vals, rmse_means, clean_labels):
        ax1.annotate(lbl, (n, rmse), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=9, color="gray")

    ax2.plot(n_train_vals, r2_means, marker="s", markersize=10, linewidth=2.5,
             color=ORANGE, markeredgecolor="white", markeredgewidth=1.5)
    ax2.set_xlabel("Number of Training Molecules")
    ax2.set_ylabel("R²")
    ax2.set_title("R² vs Training Data Size")
    ax2.grid(True, alpha=0.3)
    for n, r2, lbl in zip(n_train_vals, r2_means, clean_labels):
        ax2.annotate(lbl, (n, r2), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=9, color="gray")

    fig.suptitle(
        f"Effect of Training Data Size on {config['display_name']} Performance",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()
    path = output_path("data_scale_rmse.png", dataset_key)
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def figure_training_curves(dataset_key: str, config: dict):
    """Training and validation loss curves for one dataset."""
    results = load_baseline_results_for_dataset(dataset_key, config)
    train_losses = results.get("train_losses", [])
    val_losses = results.get("val_losses", [])

    if not train_losses:
        print(f"No {config['display_name']} loss history in results.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=BLUE, linewidth=1.5, alpha=0.7, label="Training Loss")
    ax.plot(epochs, val_losses, color=ORANGE, linewidth=2, label="Validation Loss")

    best_epoch = results.get("best_epoch", len(val_losses))
    if best_epoch <= len(val_losses):
        best_val = val_losses[best_epoch - 1]
        ax.scatter(best_epoch, best_val, color=ORANGE, s=80, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(f"Best (epoch {best_epoch})",
                    (best_epoch, best_val),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color=ORANGE)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title(f"{config['display_name']} Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_path("training_curves.png", dataset_key)
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def figure_molecule_examples(dataset_key: str, config: dict):
    """Table figure of selected molecule predictions for one dataset."""
    cases_path = output_path("molecule_cases.json", dataset_key)
    if not os.path.exists(cases_path):
        print(f"WARNING: {cases_path} not found. Run molecule_cases.py first.")
        return

    with open(cases_path) as f:
        cases = json.load(f)[:6]

    value_key = "predicted_logS" if dataset_key == "esol" else "predicted_logD"
    level_key = "solubility" if dataset_key == "esol" else "lipophilicity"
    col_labels = ["Molecule", "Formula", f"Predicted {config['target_label']}", config["task_name"]]
    cell_text = [
        [item["name"], item["formula"], f"{item[value_key]:.2f}", item[level_key]]
        for item in cases
    ]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", loc="center",
                     colWidths=[0.18, 0.22, 0.25, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for i in range(len(col_labels)):
        table[0, i].set_facecolor(BLUE)
        table[0, i].set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Example {config['display_name']} Predictions", fontsize=15, pad=20)

    path = output_path("molecule_examples.png", dataset_key)
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def generate_dataset_figures(dataset_key: str, config: dict):
    """Generate all figures for one dataset."""
    print(f"\n--- {config['display_name']} ---")
    figure_predicted_vs_true(dataset_key, config)
    figure_error_distribution(dataset_key, config)
    figure_data_scale(dataset_key, config)
    figure_training_curves(dataset_key, config)
    figure_molecule_examples(dataset_key, config)


def main():
    print("=" * 60)
    print("Generating Workshop Figures — ESOL + Lipophilicity")
    print("=" * 60)

    for dataset_key, config in DATASETS.items():
        try:
            generate_dataset_figures(dataset_key, config)
        except FileNotFoundError as e:
            print(f"SKIP {config['display_name']}: {e}")
        except Exception as e:
            print(f"SKIP {config['display_name']}: {e}")

    print(f"\nAll figures saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
