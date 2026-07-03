"""
alert.py
--------
Step 7b of the Domino Flows retraining pipeline.
Fires when the AUC gate is NOT met (AUC <= 0.80).
Logs a detailed alert, writes an alert report, and exits
with a non-zero code so the Flow surfaces a visible failure.

Usage:
    python scripts/alert.py
    python scripts/alert.py --reason "AUC below 0.80 threshold"
"""

import os
import json
import argparse
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
EVAL_RESULT_PATH = os.path.join(RESULTS_DIR, "evaluation_result.json")
ALERT_REPORT_PATH = os.path.join(RESULTS_DIR, "alert_report.json")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Alert and hold — model did not meet AUC threshold")
    p.add_argument("--reason",       default="AUC below promotion threshold",
                   help="Human-readable reason for the alert")
    p.add_argument("--eval-result",  default=EVAL_RESULT_PATH)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load evaluation result
# ---------------------------------------------------------------------------
def load_eval_result(path: str) -> dict:
    if not os.path.exists(path):
        log.warning(f"Evaluation result not found at {path} — using empty result")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Write alert report
# ---------------------------------------------------------------------------
def write_alert_report(reason: str, eval_result: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best  = eval_result.get("best_model", {})
    runs  = eval_result.get("all_runs",   [])

    report = {
        "alert_time":     datetime.now(timezone.utc).isoformat(),
        "status":         "HELD — model not promoted",
        "reason":         reason,
        "auc_achieved":   eval_result.get("auc"),
        "auc_threshold":  eval_result.get("auc_threshold", 0.80),
        "best_model": {
            "framework":  best.get("framework"),
            "auc":        best.get("auc"),
            "f1":         best.get("f1"),
            "run_id":     best.get("run_id"),
        },
        "all_runs": [
            {
                "framework": r.get("framework"),
                "auc":       r.get("auc"),
                "f1":        r.get("f1"),
            }
            for r in runs
        ],
        "recommended_actions": [
            "Review MLflow experiment runs for signs of data or training issues",
            "Check preprocessing pipeline for unexpected data quality changes",
            "Inspect monitoring PSI report for significant feature drift",
            "Consider adjusting model hyperparameters or rerunning with more data",
            "Escalate to senior data scientist if issue persists",
        ],
    }

    with open(ALERT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args        = parse_args()
    eval_result = load_eval_result(args.eval_result)

    auc           = eval_result.get("auc", "unknown")
    threshold     = eval_result.get("auc_threshold", 0.80)
    best_fw       = eval_result.get("best_model", {}).get("framework", "unknown")

    log.error("=" * 55)
    log.error("🚨 MODEL PROMOTION HELD — AUC GATE NOT MET")
    log.error("=" * 55)
    log.error(f"Reason        : {args.reason}")
    log.error(f"Best framework: {best_fw}")
    log.error(f"AUC achieved  : {auc}")
    log.error(f"AUC required  : {threshold}")
    log.error("")
    log.error("The current production model has NOT been replaced.")
    log.error("Review the evaluation_result.json and MLflow experiment")
    log.error("before rerunning the retraining pipeline.")
    log.error("=" * 55)

    report = write_alert_report(args.reason, eval_result)
    log.info(f"Alert report written to: {ALERT_REPORT_PATH}")
    log.info("Recommended actions:")
    for action in report["recommended_actions"]:
        log.info(f"  → {action}")

    # Exit with non-zero code so Domino Flows marks this step as failed
    # and surfaces it clearly in the Flow run UI
    raise SystemExit(1)


if __name__ == "__main__":
    main()
