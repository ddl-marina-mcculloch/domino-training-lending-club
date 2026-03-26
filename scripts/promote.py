"""
promote.py
----------
Step 7a of the Domino Flows retraining pipeline.
Reads evaluation_result.json and promotes the best model to the
live Domino scoring endpoint via the Domino API.

Only runs when the Flows condition `outputs.evaluate.auc > 0.80` is met.

Reference:
    https://docs.dominodatalab.com/en/latest/api_guide/model_apis/

Usage:
    python scripts/promote.py
    python scripts/promote.py --endpoint-name wine-model-yourname
"""

import os
import json
import argparse
import logging
import shutil

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOMINO_API_HOST  = os.environ.get("DOMINO_API_HOST", "")
DOMINO_API_TOKEN = os.environ.get("DOMINO_USER_API_KEY", "")
PROJECT_NAME     = os.environ.get("DOMINO_PROJECT_NAME", "LendingClubProject")
PROJECT_OWNER    = os.environ.get("DOMINO_PROJECT_OWNER", "")

RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
EVAL_RESULT_PATH = os.path.join(RESULTS_DIR, "evaluation_result.json")
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "..", "models")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Promote best model to Domino endpoint")
    p.add_argument("--eval-result",    default=EVAL_RESULT_PATH)
    p.add_argument("--endpoint-name",  default="lending-credit-risk",
                   help="Name of the Domino endpoint to update")
    p.add_argument("--predict-file",   default="scripts/predict.py",
                   help="Scoring script path")
    p.add_argument("--predict-func",   default="predict",
                   help="Scoring function name")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print promotion details without calling the API")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load evaluation result
# ---------------------------------------------------------------------------
def load_eval_result(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation result not found at {path}. "
            "Run evaluate.py before promote.py."
        )
    with open(path) as f:
        result = json.load(f)
    log.info(f"Loaded evaluation result: {path}")
    return result


# ---------------------------------------------------------------------------
# Copy winning model to a stable path for the endpoint
# ---------------------------------------------------------------------------
def stage_model(eval_result: dict) -> str:
    """
    Copies the best model artifact to a stable 'active_model' path
    so the endpoint always points to a known location.
    """
    framework   = eval_result["best_model"]["framework"]
    source_path = eval_result["best_model"]["model_path"]
    staged_path = os.path.join(MODEL_DIR, "active_model.pkl")

    if framework == "h2o":
        # H2O model is a directory — copy the whole thing
        staged_dir = os.path.join(MODEL_DIR, "active_h2o_model")
        if os.path.exists(staged_dir):
            shutil.rmtree(staged_dir)
        shutil.copytree(source_path, staged_dir)
        log.info(f"H2O model staged to: {staged_dir}")
        return staged_dir
    else:
        shutil.copy2(source_path, staged_path)
        log.info(f"Model staged to: {staged_path}  (from: {source_path})")
        return staged_path


# ---------------------------------------------------------------------------
# Domino API — trigger model version update
# ---------------------------------------------------------------------------
def promote_via_api(endpoint_name: str, predict_file: str,
                    predict_func: str, eval_result: dict) -> dict:
    """
    Calls the Domino API to publish a new version of the model endpoint.
    This creates a new model version and starts the build process.
    """
    if not DOMINO_API_HOST or not DOMINO_API_TOKEN:
        raise EnvironmentError(
            "DOMINO_API_HOST and DOMINO_USER_API_KEY must be set to call the Domino API."
        )

    headers = {
        "Content-Type":  "application/json",
        "X-Domino-Api-Key": DOMINO_API_TOKEN,
    }

    best = eval_result["best_model"]

    # Get model ID by name
    models_url = (
        f"{DOMINO_API_HOST}/v4/models"
        f"?projectOwnerName={PROJECT_OWNER}&projectName={PROJECT_NAME}"
    )
    resp = requests.get(models_url, headers=headers, timeout=15)
    resp.raise_for_status()

    models = resp.json().get("data", [])
    model  = next((m for m in models if m.get("name") == endpoint_name), None)

    if model is None:
        raise ValueError(
            f"Endpoint '{endpoint_name}' not found in project '{PROJECT_NAME}'. "
            f"Available endpoints: {[m['name'] for m in models]}"
        )

    model_id = model["id"]
    log.info(f"Found endpoint '{endpoint_name}' with ID: {model_id}")

    # Publish new model version
    publish_url = f"{DOMINO_API_HOST}/v4/models/{model_id}/versions"
    payload = {
        "file":             predict_file,
        "function":         predict_func,
        "description":      (
            f"Auto-promoted by retraining pipeline. "
            f"Framework: {best['framework']}  AUC: {best['auc']:.4f}  "
            f"F1: {best['f1']:.4f}  MLflow run: {best['run_id']}"
        ),
    }

    log.info(f"Publishing new version to endpoint: {endpoint_name}")
    pub_resp = requests.post(publish_url, headers=headers,
                             data=json.dumps(payload), timeout=30)
    pub_resp.raise_for_status()
    version_data = pub_resp.json()

    log.info(f"New version created: {version_data.get('number', 'unknown')}")
    return version_data


# ---------------------------------------------------------------------------
# Write promotion log
# ---------------------------------------------------------------------------
def write_promotion_log(eval_result: dict, version_data: dict, dry_run: bool):
    log_path = os.path.join(RESULTS_DIR, "promotion_log.json")
    log_entry = {
        "status":          "dry_run" if dry_run else "promoted",
        "best_model":      eval_result["best_model"],
        "endpoint_version": version_data.get("number"),
        "auc":             eval_result["auc"],
        "gate_passed":     eval_result["gate_passed"],
    }
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)
    log.info(f"Promotion log written to: {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args      = parse_args()
    eval_result = load_eval_result(args.eval_result)
    best      = eval_result["best_model"]

    log.info("=== Model Promotion ===")
    log.info(f"Best model  : {best['framework']}")
    log.info(f"AUC         : {best['auc']:.4f}")
    log.info(f"F1          : {best['f1']:.4f}")
    log.info(f"Gate passed : {eval_result['gate_passed']}")
    log.info(f"Endpoint    : {args.endpoint_name}")

    if not eval_result["gate_passed"]:
        log.error("Gate not passed — promote.py should not have been called. "
                  "Check Flows conditional logic.")
        raise SystemExit(1)

    # Stage model artifact
    staged_path = stage_model(eval_result)
    log.info(f"Model staged at: {staged_path}")

    if args.dry_run:
        log.info("DRY RUN — skipping Domino API call")
        write_promotion_log(eval_result, {}, dry_run=True)
        log.info("Dry run complete — model would have been promoted")
        return

    # Call Domino API to publish new endpoint version
    version_data = promote_via_api(
        args.endpoint_name,
        args.predict_file,
        args.predict_func,
        eval_result,
    )

    write_promotion_log(eval_result, version_data, dry_run=False)
    log.info(f"\n✅ Model promoted successfully — new endpoint version: "
             f"{version_data.get('number', 'unknown')}")


if __name__ == "__main__":
    main()
