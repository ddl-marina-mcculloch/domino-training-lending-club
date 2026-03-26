"""
evaluate.py
-----------
Step 6 of the Domino Flows retraining pipeline.
Queries MLflow for the latest runs from all three training frameworks,
selects the best model by AUC, and writes evaluation_result.json.

The Flows conditional branching in retraining_flow.yaml reads
outputs.evaluate.auc to decide whether to promote or hold.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --auc-threshold 0.80
"""

import os
import json
import argparse
import logging

import mlflow

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH     = os.path.join(RESULTS_DIR, "evaluation_result.json")
EXPERIMENT_NAME = "LendingClub-CreditRisk"
FRAMEWORKS      = ["sklearn", "xgboost", "h2o"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and select best model from MLflow runs")
    p.add_argument("--auc-threshold", type=float, default=0.80,
                   help="Minimum AUC required to promote model to endpoint")
    p.add_argument("--experiment",    default=EXPERIMENT_NAME)
    p.add_argument("--output",        default=OUTPUT_PATH)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fetch latest runs from MLflow
# ---------------------------------------------------------------------------
def get_latest_runs(experiment_name: str) -> list[dict]:
    """
    Returns the most recent run for each framework, sorted by AUC descending.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' not found. "
            "Ensure training scripts have been run before evaluate.py."
        )

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        filter_string="attributes.status = 'FINISHED'",
    )

    if runs_df.empty:
        raise ValueError("No finished MLflow runs found. Cannot evaluate.")

    results = []
    for framework in FRAMEWORKS:
        fw_runs = runs_df[runs_df.get("tags.framework", runs_df.index) == framework] \
            if "tags.framework" in runs_df.columns else runs_df

        # Filter by framework tag
        if "tags.framework" in runs_df.columns:
            fw_runs = runs_df[runs_df["tags.framework"] == framework]
        else:
            fw_runs = runs_df

        if fw_runs.empty:
            log.warning(f"No runs found for framework: {framework}")
            continue

        # Take the most recent run for this framework
        latest = fw_runs.iloc[0]

        auc  = latest.get("metrics.auc",       None)
        f1   = latest.get("metrics.f1",        None)
        prec = latest.get("metrics.precision", None)
        rec  = latest.get("metrics.recall",    None)

        if auc is None:
            log.warning(f"No AUC metric found for {framework} run — skipping")
            continue

        results.append({
            "framework":  framework,
            "run_id":     latest["run_id"],
            "run_name":   latest.get("tags.mlflow.runName", framework),
            "auc":        round(float(auc),  4),
            "f1":         round(float(f1),   4) if f1   is not None else None,
            "precision":  round(float(prec), 4) if prec is not None else None,
            "recall":     round(float(rec),  4) if rec  is not None else None,
        })

    if not results:
        raise ValueError("Could not retrieve metrics for any framework.")

    # Sort by AUC descending
    results.sort(key=lambda r: r["auc"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Model path resolver
# ---------------------------------------------------------------------------
def get_model_path(framework: str) -> str:
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    paths = {
        "sklearn":  os.path.join(model_dir, "sklearn_rf_model.pkl"),
        "xgboost":  os.path.join(model_dir, "xgboost_model.pkl"),
        "h2o":      model_dir,   # H2O saves to directory
    }
    return paths.get(framework, model_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info(f"Fetching MLflow runs from experiment: {args.experiment}")
    runs = get_latest_runs(args.experiment)

    log.info("\n=== Model Comparison ===")
    for r in runs:
        log.info(
            f"  {r['framework']:<10} AUC: {r['auc']:.4f}  "
            f"F1: {r['f1']:.4f}  "
            f"Precision: {r['precision']:.4f}  "
            f"Recall: {r['recall']:.4f}"
        )

    best = runs[0]
    gate_passed = best["auc"] >= args.auc_threshold

    log.info(f"\n🏆 Best model  : {best['framework']}  (AUC: {best['auc']:.4f})")
    log.info(f"AUC threshold  : {args.auc_threshold}")
    log.info(f"Gate passed    : {'✅ YES — promote' if gate_passed else '🚨 NO — hold'}")

    # Write evaluation result
    result = {
        "best_model": {
            "framework":  best["framework"],
            "run_id":     best["run_id"],
            "model_path": get_model_path(best["framework"]),
            "auc":        best["auc"],
            "f1":         best["f1"],
            "precision":  best["precision"],
            "recall":     best["recall"],
        },
        "auc":            best["auc"],        # Top-level for Flows condition check
        "gate_passed":    gate_passed,
        "auc_threshold":  args.auc_threshold,
        "all_runs":       runs,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"\nEvaluation result written to: {args.output}")
    return result


if __name__ == "__main__":
    main()
