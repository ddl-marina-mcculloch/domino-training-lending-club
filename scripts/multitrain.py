"""
multitrain.py
-------------
Orchestrates the full model training pipeline for the LendingClub workshop.

Steps:
    1. Run preprocessing (optional — skip if lending_clean.csv already exists)
    2. Train sklearn Random Forest
    3. Train XGBoost classifier
    4. Train H2O AutoML
    5. Print MLflow experiment summary comparing all three runs

Usage:
    python scripts/multitrain.py
    python scripts/multitrain.py --skip-preprocess
"""

import os
import sys
import argparse
import logging
import subprocess

import pandas as pd
import mlflow

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_NAME  = os.environ.get("DOMINO_PROJECT_NAME", "LendingClubProject")
RAW_DATA      = f"/domino/datasets/local/{PROJECT_NAME}/lending_raw.csv"
CLEAN_DATA    = f"/domino/datasets/local/{PROJECT_NAME}/lending_clean.csv"
SCRIPTS_DIR   = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT    = "LendingClub-CreditRisk"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run full LendingClub model training pipeline")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing if lending_clean.csv already exists")
    p.add_argument("--skip-sklearn",    action="store_true")
    p.add_argument("--skip-xgboost",    action="store_true")
    p.add_argument("--skip-h2o",        action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run a script as a subprocess
# ---------------------------------------------------------------------------
def run_script(script_name: str, extra_args: list = None):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [sys.executable, script_path] + (extra_args or [])
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode


# ---------------------------------------------------------------------------
# Fetch MLflow experiment summary
# ---------------------------------------------------------------------------
def print_experiment_summary():
    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT SUMMARY")
    log.info("=" * 60)

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT)
        if experiment is None:
            log.warning(f"Experiment '{EXPERIMENT}' not found in MLflow.")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.auc DESC"]
        )

        if runs.empty:
            log.warning("No runs found.")
            return

        cols = ["tags.mlflow.runName", "tags.framework",
                "metrics.auc", "metrics.f1",
                "metrics.precision", "metrics.recall"]
        cols = [c for c in cols if c in runs.columns]

        summary = runs[cols].rename(columns={
            "tags.mlflow.runName": "run_name",
            "tags.framework":      "framework",
            "metrics.auc":        "AUC",
            "metrics.f1":         "F1",
            "metrics.precision":  "Precision",
            "metrics.recall":     "Recall",
        })

        log.info(f"\n{summary.to_string(index=False)}")

        best_run = summary.iloc[0]
        log.info(f"\n🏆 Best model : {best_run.get('run_name', 'N/A')} "
                 f"(AUC: {best_run.get('AUC', 'N/A'):.4f})")

    except Exception as e:
        log.warning(f"Could not fetch MLflow summary: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Step 1: Preprocess
    if not args.skip_preprocess:
        if os.path.exists(CLEAN_DATA):
            log.info(f"Cleaned data already exists at {CLEAN_DATA} — skipping preprocessing.")
            log.info("Pass --skip-preprocess to suppress this message, or delete the file to re-run.")
        else:
            log.info("Step 1: Running preprocessing pipeline...")
            run_script("preprocess.py", ["--input", RAW_DATA, "--output", CLEAN_DATA])
    else:
        log.info("Step 1: Preprocessing skipped (--skip-preprocess)")

    # Validate cleaned data exists before training
    if not os.path.exists(CLEAN_DATA):
        log.error(f"Cleaned data not found at {CLEAN_DATA}. Cannot proceed with training.")
        sys.exit(1)

    # Step 2: sklearn
    if not args.skip_sklearn:
        log.info("Step 2: Training sklearn Random Forest...")
        run_script("train_sklearn.py")
    else:
        log.info("Step 2: sklearn training skipped")

    # Step 3: XGBoost
    if not args.skip_xgboost:
        log.info("Step 3: Training XGBoost classifier...")
        run_script("train_xgboost.py")
    else:
        log.info("Step 3: XGBoost training skipped")

    # Step 4: H2O AutoML
    if not args.skip_h2o:
        log.info("Step 4: Training H2O AutoML...")
        run_script("train_h2o.py")
    else:
        log.info("Step 4: H2O AutoML skipped")

    # Step 5: Summary
    print_experiment_summary()

    log.info("\n✅ multitrain.py complete. Navigate to Experiments in your Domino project to compare runs.")


if __name__ == "__main__":
    main()
