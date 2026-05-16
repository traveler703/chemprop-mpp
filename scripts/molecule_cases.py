"""
molecule_cases.py — Predict solubility for famous molecules and save results.

Used to populate the molecule examples table in demo/PPT.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Famous molecules
MOLECULES = [
    ("Aspirin",      "CC(=O)Oc1ccccc1C(=O)O",      "C₉H₈O₄",     "Pain reliever / anti-inflammatory"),
    ("Caffeine",     "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C₈H₁₀N₄O₂", "Stimulant"),
    ("Benzene",      "c1ccccc1",                     "C₆H₆",       "Simple aromatic — hydrophobic"),
    ("Ibuprofen",    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "C₁₃H₁₈O₂", "NSAID pain reliever"),
    ("Paracetamol",  "CC(=O)NC1=CC=C(C=C1)O",        "C₈H₉NO₂",   "Analgesic / antipyretic"),
    ("Glucose",      "C(C1C(C(C(C(O1)O)O)O)O)O",     "C₆H₁₂O₆",   "Sugar — very hydrophilic"),
    ("Testosterone", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "C₁₉H₂₈O₂", "Steroid hormone"),
    ("Vitamin C",    "C(C(C1C(=C(C(=O)O1)O)O)O)O",   "C₆H₈O₆",    "Ascorbic acid — antioxidant"),
    ("Tamoxifen",    "CC/C(=C(/c1ccccc1)\\c2ccc(cc2)OCCN(C)C)/c3ccccc3", "C₂₆H₂₉NO", "Breast cancer drug"),
    ("Ampicillin",   "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C", "C₁₆H₁₉N₃O₄S", "Antibiotic"),
]

DATASETS = {
    "esol": {
        "display_name": "ESOL Solubility",
        "model_candidates": ["model_esol_train.pt", "model_esol_train_100.pt"],
        "fallback_train_csv": os.path.join(DATA_DIR, "esol_train.csv"),
        "output_prefix": "",
        "target_label": "LogS",
        "value_key": "predicted_logS",
        "level_key": "solubility",
    },
    "lipo": {
        "display_name": "Lipophilicity",
        "model_candidates": ["model_lipo_train.pt", "model_lipo_train_100.pt"],
        "fallback_train_csv": os.path.join(DATA_DIR, "lipo_train.csv"),
        "output_prefix": "lipo_",
        "target_label": "logD",
        "value_key": "predicted_logD",
        "level_key": "lipophilicity",
    },
}


def output_path(config: dict, filename: str) -> str:
    """Keep original ESOL filenames and add lipo_ prefix for Lipophilicity."""
    return os.path.join(RESULTS_DIR, f"{config['output_prefix']}{filename}")


def build_target_scaler(train_csv):
    """Rebuild the target scaler used during training."""
    train_df = pd.read_csv(train_csv)
    target_col = "target" if "target" in train_df.columns else "logS"
    dps = [
        MoleculeDatapoint.from_smi(row["smiles"], y=np.array([row[target_col]], dtype=float))
        for _, row in train_df.iterrows()
    ]
    dataset = MoleculeDataset(dps)
    return dataset.normalize_targets()


def load_model(dataset_key: str, config: dict):
    """Find and load the trained model for one dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_paths = [
        os.path.join(MODELS_DIR, filename)
        for filename in config["model_candidates"]
    ]

    model_path = None
    for mp in model_paths:
        if os.path.exists(mp):
            model_path = mp
            break

    if model_path is None:
        print(f"WARNING: No {config['display_name']} model found. Run train.py first.")
        print(f"Searched: {model_paths}")
        return None, None, None

    print(f"Loading model: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    hidden_size = int(ckpt_args.get("hidden_size", 300))
    depth = int(ckpt_args.get("depth", 5))
    dropout = float(ckpt_args.get("dropout", 0.1))
    train_csv = ckpt_args.get("train_data", config["fallback_train_csv"])

    mp_block = BondMessagePassing(d_h=hidden_size, depth=depth, dropout=dropout)
    agg = MeanAggregation()
    predictor = RegressionFFN(input_dim=hidden_size, hidden_dim=hidden_size, dropout=dropout)
    model = MPNN(
        message_passing=mp_block,
        agg=agg,
        predictor=predictor,
        batch_norm=True,
    )

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    scaler = build_target_scaler(train_csv)
    return model, device, scaler


def predict(model, device, scaler, smiles):
    """Predict a single molecular property value."""
    datapoint = MoleculeDatapoint.from_smi(smiles)
    dataset = MoleculeDataset([datapoint])
    loader = build_dataloader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            V_d = batch.V_d.to(device) if batch.V_d is not None else None
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            pred = model(bmg, V_d, X_d).cpu().numpy().reshape(-1, 1)
            return float(scaler.inverse_transform(pred).flatten()[0])


def solubility_label(logS):
    """Return human-readable solubility level."""
    if logS > 0:
        return "High (very soluble)"
    elif logS > -2:
        return "Moderate"
    elif logS > -4:
        return "Low (poorly soluble)"
    else:
        return "Very low (insoluble)"


def lipophilicity_label(logD):
    """Return human-readable lipophilicity level."""
    if logD < 1:
        return "Low lipophilicity"
    elif logD < 3:
        return "Moderate lipophilicity"
    elif logD < 4:
        return "High lipophilicity"
    else:
        return "Very high lipophilicity"


def property_label(dataset_key: str, value: float) -> str:
    """Map a predicted value to a human-readable level."""
    if dataset_key == "esol":
        return solubility_label(value)
    return lipophilicity_label(value)


def run_cases(dataset_key: str, config: dict):
    """Generate molecule case predictions for one dataset."""
    print("=" * 60)
    print(f"Molecule Case Studies — {config['display_name']} Prediction")
    print("=" * 60)

    model, device, scaler = load_model(dataset_key, config)
    if model is None:
        return
    print()

    results = []
    for name, smi, formula, desc in MOLECULES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  SKIP {name}: invalid SMILES")
            continue

        value = predict(model, device, scaler, smi)
        level = property_label(dataset_key, value)
        mw = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)

        item = {
            "name": name,
            "smiles": smi,
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "description": desc,
            "molecular_weight": round(mw, 1),
            "logP": round(logP, 2),
        }
        item[config["value_key"]] = round(value, 3)
        item[config["level_key"]] = level
        results.append(item)

        print(
            f"  {name:15s}  {config['target_label']}={value:+.3f}  "
            f"{level}  (MW={mw:.0f}, RDKit LogP={logP:.2f})"
        )

    # Save
    cases_path = output_path(config, "molecule_cases.json")
    with open(cases_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {cases_path}")

    # Generate molecule images
    print("\nGenerating molecule structure images...")
    mols = [Chem.MolFromSmiles(r["smiles"]) for r in results]
    names = [r["name"] for r in results]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 180),
                               legends=names)
    img_path = output_path(config, "molecule_structures.png")
    img.save(img_path)
    print(f"Saved: {img_path}")

    # Print summary table
    df = pd.DataFrame(results)
    df = df[["name", "formula", config["value_key"], config["level_key"], "molecular_weight", "logP"]]
    df.columns = ["Molecule", "Formula", f"Predicted {config['target_label']}", config["display_name"], "MW", "RDKit LogP"]
    print(f"\n{'─'*70}")
    print(df.to_string(index=False))
    print(f"{'─'*70}")


def main():
    for dataset_key, config in DATASETS.items():
        run_cases(dataset_key, config)


if __name__ == "__main__":
    main()
