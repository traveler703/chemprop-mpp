"""
train.py — Train a Chemprop MPNN model on a molecular property dataset.

Usage:
    python train.py                           # Train ESOL and Lipophilicity baselines
    python train.py --train-data ../data/esol_train_50.csv  # Subset training
    python train.py --train-data ../data/lipo_train.csv --test-data ../data/lipo_test.csv
    python train.py --epochs 100 --batch-size 32
"""

import os
import sys
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_DATASETS = {
    "esol": {
        "name": "ESOL",
        "train": os.path.join(DATA_DIR, "esol_train.csv"),
        "test": os.path.join(DATA_DIR, "esol_test.csv"),
    },
    "lipophilicity": {
        "name": "Lipophilicity",
        "train": os.path.join(DATA_DIR, "lipo_train.csv"),
        "test": os.path.join(DATA_DIR, "lipo_test.csv"),
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Train Chemprop on a molecular property dataset")
    p.add_argument("--dataset", choices=["esol", "lipophilicity", "all"], default="all",
                   help="Dataset to train when train/test CSVs are not specified")
    p.add_argument("--train-data", default=os.path.join(DATA_DIR, "esol_train.csv"),
                   help="Path to training CSV")
    p.add_argument("--test-data", default=os.path.join(DATA_DIR, "esol_test.csv"),
                   help="Path to test CSV")
    p.add_argument("--target-column", default="target",
                   help="Name of the regression target column")
    p.add_argument("--epochs", type=int, default=100, help="Max epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience")
    p.add_argument("--hidden-size", type=int, default=300,
                   help="Hidden dimension of MPNN")
    p.add_argument("--depth", type=int, default=5, help="Number of message-passing layers")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None,
                   help="Output path for results JSON")
    p.add_argument("--model-save", default=None,
                   help="Path to save model checkpoint")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_target_column(df: pd.DataFrame, requested_column: str) -> str:
    """Return the target column, with a fallback for older ESOL CSV files."""
    if requested_column in df.columns:
        return requested_column
    if requested_column == "target" and "logS" in df.columns:
        print("WARNING: 'target' column not found; using legacy 'logS' column.")
        return "logS"
    raise ValueError(
        f"Target column '{requested_column}' not found. Available columns: {list(df.columns)}"
    )


def load_data(train_path: str, test_path: str, target_column: str):
    """Load CSV data and return DataFrames."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    resolved_target = resolve_target_column(train_df, target_column)
    if resolved_target not in test_df.columns:
        raise ValueError(
            f"Target column '{resolved_target}' not found in test data. "
            f"Available columns: {list(test_df.columns)}"
        )
    if "smiles" not in train_df.columns or "smiles" not in test_df.columns:
        raise ValueError("Both train and test CSV files must contain a 'smiles' column.")

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(
        f"Train target range: "
        f"[{train_df[resolved_target].min():.3f}, {train_df[resolved_target].max():.3f}]"
    )
    return train_df, test_df, resolved_target


def train_chemprop_v2(train_df, test_df, args, target_column: str):
    """
    Train using Chemprop v2 Python API.

    Chemprop v2 uses:
      - chemprop.data.MoleculeDatapoint for each molecule
      - chemprop.data.MoleculeDataset to hold them
      - chemprop.models.MPNN for the model
      - Standard PyTorch training loop
    """
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datapoints
    train_dps = [
        MoleculeDatapoint.from_smi(row["smiles"], y=np.array([row[target_column]], dtype=float))
        for _, row in train_df.iterrows()
    ]
    test_dps = [
        MoleculeDatapoint.from_smi(row["smiles"], y=np.array([row[target_column]], dtype=float))
        for _, row in test_df.iterrows()
    ]

    # Create datasets
    train_dataset = MoleculeDataset(train_dps)
    test_dataset = MoleculeDataset(test_dps)

    # Normalize targets
    scaler = train_dataset.normalize_targets()
    test_dataset.normalize_targets(scaler)

    # Build dataloaders
    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    mp_block = BondMessagePassing(d_h=args.hidden_size, depth=args.depth, dropout=args.dropout)
    agg = MeanAggregation()
    predictor = RegressionFFN(input_dim=args.hidden_size, hidden_dim=args.hidden_size, dropout=args.dropout)
    model = MPNN(
        message_passing=mp_block,
        agg=agg,
        predictor=predictor,
        batch_norm=True,
    )
    model.to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"\n{'='*50}")
    print(f"Training MPNN (hidden={args.hidden_size}, depth={args.depth}, epochs={args.epochs})")
    print(f"{'='*50}")

    start_time = time.time()

    for epoch in range(args.epochs):
        # ── Training ──
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            bmg = batch.bmg
            bmg.to(device)
            V_d = batch.V_d.to(device) if batch.V_d is not None else None
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            targets = batch.Y.to(device)

            optimizer.zero_grad()
            preds = model(bmg, V_d, X_d)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(targets)

        avg_train_loss = total_train_loss / len(train_dataset)

        # ── Validation ──
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                bmg = batch.bmg
                bmg.to(device)
                V_d = batch.V_d.to(device) if batch.V_d is not None else None
                X_d = batch.X_d.to(device) if batch.X_d is not None else None
                targets = batch.Y.to(device)

                preds = model(bmg, V_d, X_d)
                loss = loss_fn(preds, targets)
                total_val_loss += loss.item() * len(targets)

        avg_val_loss = total_val_loss / len(test_dataset)

        # ── Un-normalized metrics ──
        # Evaluate on test set in original scale
        y_true, y_pred = evaluate_model(
            model, test_loader, scaler, device
        )
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"TrainLoss: {avg_train_loss:.4f} | "
                  f"ValLoss: {avg_val_loss:.4f} | "
                  f"RMSE: {rmse:.3f} | R2: {r2:.3f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch})")
                break

    train_time = time.time() - start_time

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation
    y_true, y_pred = evaluate_model(model, test_loader, scaler, device)

    results = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "target_column": target_column,
        "epochs_run": epoch + 1,
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "predictions": [float(x) for x in y_pred],
        "targets": [float(x) for x in y_true],
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    print(f"\n=== Final Results ===")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE:  {results['mae']:.4f}")
    print(f"R2:   {results['r2']:.4f}")
    print(f"Time: {train_time:.1f}s")

    return model, results


def evaluate_model(model, loader, scaler, device):
    """Get predictions in original scale."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            V_d = batch.V_d.to(device) if batch.V_d is not None else None
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            targets = batch.Y.to(device)
            preds = model(bmg, V_d, X_d)

            # Un-normalize
            if scaler is not None:
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Reshape for scaler (n_samples, n_targets=1)
                preds_un = scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
                targets_un = scaler.inverse_transform(targets_np.reshape(-1, 1)).flatten()
            else:
                preds_un = preds.cpu().numpy().flatten()
                targets_un = targets.cpu().numpy().flatten()

            all_preds.extend(preds_un.tolist())
            all_targets.extend(targets_un.tolist())

    return np.array(all_targets), np.array(all_preds)


def has_explicit_data_args() -> bool:
    """Check whether the user passed custom train/test CSV paths."""
    return any(
        arg == "--train-data" or arg.startswith("--train-data=")
        or arg == "--test-data" or arg.startswith("--test-data=")
        for arg in sys.argv[1:]
    )


def make_dataset_args(args, dataset_key: str, reset_outputs: bool = False):
    """Create an argparse namespace for one default dataset run."""
    dataset = DEFAULT_DATASETS[dataset_key]
    run_args = argparse.Namespace(**vars(args))
    run_args.train_data = dataset["train"]
    run_args.test_data = dataset["test"]
    if reset_outputs:
        run_args.output = None
        run_args.model_save = None
    return run_args


def run_training(args):
    """Train one model and save its results/checkpoint."""
    set_seed(args.seed)

    print("=" * 60)
    print("Chemprop MPNN Training — Molecular Property Prediction")
    print("=" * 60)
    print(f"Train data: {args.train_data}")
    print(f"Test data:  {args.test_data}")
    print(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Epochs:     {args.epochs}")
    print()

    train_df, test_df, target_column = load_data(args.train_data, args.test_data, args.target_column)
    model, results = train_chemprop_v2(train_df, test_df, args, target_column)

    # Save results
    if args.output is None:
        train_name = os.path.splitext(os.path.basename(args.train_data))[0]
        args.output = os.path.join(MODELS_DIR, f"results_{train_name}.json")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Save model
    if args.model_save is None:
        train_name = os.path.splitext(os.path.basename(args.train_data))[0]
        args.model_save = os.path.join(MODELS_DIR, f"model_{train_name}.pt")

    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "results": {k: v for k, v in results.items() if k not in ("predictions", "targets", "train_losses", "val_losses")},
    }, args.model_save)
    print(f"Model saved to: {args.model_save}")
    return results


def main():
    args = parse_args()

    if not has_explicit_data_args() and args.dataset == "all":
        for dataset_key in ("esol", "lipophilicity"):
            print(f"\n\n### Default baseline: {DEFAULT_DATASETS[dataset_key]['name']} ###")
            run_training(make_dataset_args(args, dataset_key, reset_outputs=True))
        return

    if not has_explicit_data_args() and args.dataset in DEFAULT_DATASETS:
        args = make_dataset_args(args, args.dataset)

    run_training(args)


if __name__ == "__main__":
    main()
