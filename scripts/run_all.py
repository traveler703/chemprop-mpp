"""
run_all.py — Run the complete workshop pipeline.

Steps:
  1. Prepare ESOL dataset
  2. Train baseline Chemprop model (100% data)
  3. Run data scale experiment (20/50/80/100%)
  4. Generate visualizations
  5. Create PPT presentation

Usage:
    python run_all.py                  # Run everything
    python run_all.py --skip-scales    # Skip data scale experiment (faster)
    python run_all.py --only-data      # Only prepare data
"""

import os
import sys
import subprocess
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def run_step(name: str, script: str, args=None):
    """Run a Python script and report success/failure."""
    print(f"\n{'#'*60}")
    print(f"# STEP: {name}")
    print(f"{'#'*60}")

    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script)]
    if args:
        cmd.extend(args)

    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ {name} — completed in {elapsed:.0f}s")
        return True
    else:
        print(f"\n✗ {name} — FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Chemprop Workshop Pipeline")
    parser.add_argument("--skip-scales", action="store_true",
                        help="Skip data scale experiment (saves time)")
    parser.add_argument("--skip-ppt", action="store_true",
                        help="Skip PPT generation")
    parser.add_argument("--only-data", action="store_true",
                        help="Only prepare the dataset")
    parser.add_argument("--only-train", action="store_true",
                        help="Only train baseline model")
    args = parser.parse_args()

    print("=" * 60)
    print("Chemprop Workshop — Full Pipeline")
    print("=" * 60)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python:  {sys.executable}")

    total_start = time.time()
    steps_ok = 0
    steps_fail = 0

    # ── Step 1: Prepare Data ──
    if run_step("Prepare ESOL Dataset", "prepare_data.py"):
        steps_ok += 1
    else:
        steps_fail += 1
        print("CRITICAL: Data preparation failed. Cannot continue.")
        return

    if args.only_data:
        print("\nDone (data only).")
        return

    # ── Step 2: Train Baseline ──
    if run_step("Train Baseline Models", "train.py", ["--epochs", "100"]):
        steps_ok += 1
    else:
        steps_fail += 1

    if args.only_train:
        print("\nDone (baseline training only).")
        return

    # ── Step 3: Data Scale Experiment ──
    if not args.skip_scales:
        if run_step("Data Scale Experiment", "data_scale_experiment.py"):
            steps_ok += 1
        else:
            steps_fail += 1
    else:
        print("\n⏭ Skipping data scale experiment.")

    # ── Step 4: Molecule Case Studies ──
    if run_step("Molecule Case Studies", "molecule_cases.py"):
        steps_ok += 1
    else:
        steps_fail += 1

    # ── Step 5: Visualizations ──
    if run_step("Generate Visualizations", "visualize.py"):
        steps_ok += 1
    else:
        steps_fail += 1

    # ── Step 6: PPT ──
    if not args.skip_ppt:
        if run_step("Create PPT Presentation", "create_ppt.py"):
            steps_ok += 1
        else:
            steps_fail += 1
    else:
        print("\n⏭ Skipping PPT generation.")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"  Steps passed: {steps_ok}")
    print(f"  Steps failed: {steps_fail}")
    print(f"  Total time:   {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    print(f"\nDeliverables:")
    print(f"  Data:     {os.path.join(PROJECT_ROOT, 'data')}")
    print(f"  Models:   {os.path.join(PROJECT_ROOT, 'models')}")
    print(f"  Figures:  {os.path.join(PROJECT_ROOT, 'results')}")
    print(f"  PPT:      {os.path.join(PROJECT_ROOT, 'ppt')}")
    print(f"  Notebook: {os.path.join(PROJECT_ROOT, 'notebooks', 'demo.ipynb')}")


if __name__ == "__main__":
    main()
