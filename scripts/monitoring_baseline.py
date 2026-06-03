"""
monitoring_baseline.py
----------------------
Generates baseline statistics and simulates prediction drift
for the LendingClub credit risk model in Domino Model Monitor.

Two modes:
    1. baseline   — computes and saves baseline statistics from the training set
    2. simulate   — injects a shifted dataset and registers it as prediction data
                    in Domino Model Monitor via the REST API to trigger drift alerts

Usage:
    python scripts/monitoring_baseline.py --mode baseline
    python scripts/monitoring_baseline.py --mode simulate --shift-severity medium
    python scripts/monitoring_baseline.py --mode both --shift-severity high

Data source:
    Reads training data from the 'lending-club-data' Domino Data Source (S3),
    falling back to object_store if injected; raises RuntimeError otherwise.

Model Monitor integration:
    Uses the Domino Model Monitor REST API to register the simulated drift CSV
    as a prediction dataset. Requires DOMINO_USER_API_KEY and DOMINO_API_HOST
    environment variables (both are injected automatically in Domino workspaces).
"""

import os
import argparse
import logging
import json
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
PROJECT_NAME   = os.environ.get("DOMINO_PROJECT_NAME", "domino-training-lending-club")
MONITORING_DIR = os.path.join(os.path.dirname(__file__), "..", "monitoring")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "results")
BASELINE_PATH  = os.path.join(MONITORING_DIR, "baseline_stats.json")
DRIFT_DATA_PATH = os.path.join(MONITORING_DIR, "drift_simulation.csv")

# Domino Model Monitor settings
DOMINO_API_HOST   = os.environ.get("DOMINO_API_HOST", "https://cloud-dogfood.domino.tech")
DOMINO_USER_API_KEY = os.environ.get("DOMINO_USER_API_KEY", "")

# Data source name (as configured in Model Monitor > Data Sources)
DATA_SOURCE_NAME = "lending-club-data"
TRAINING_FILENAME = "lending_clean.csv"

# Local data source fallback path
LOCAL_DATA_PATH = f"/domino/datasources/{DATA_SOURCE_NAME}/{TRAINING_FILENAME}"

# ---------------------------------------------------------------------------
# Features to monitor — most impactful per SHAP analysis
# ---------------------------------------------------------------------------
MONITOR_FEATURES = [
    "dti",
    "int_rate",
    "loan_amnt",
    "annual_inc",
    "revol_util",
    "loan_to_income",
    "credit_utilization",
    "payment_to_income",
]

# ---------------------------------------------------------------------------
# Shift configurations for drift simulation
# ---------------------------------------------------------------------------
SHIFT_CONFIGS = {
    "low": {
        "dti":              {"mean_shift": 2.0,  "std_scale": 1.1},
        "int_rate":         {"mean_shift": 1.0,  "std_scale": 1.05},
        "loan_amnt":        {"mean_shift": 500,  "std_scale": 1.0},
        "revol_util":       {"mean_shift": 3.0,  "std_scale": 1.05},
    },
    "medium": {
        "dti":              {"mean_shift": 5.0,  "std_scale": 1.3},
        "int_rate":         {"mean_shift": 2.5,  "std_scale": 1.2},
        "loan_amnt":        {"mean_shift": 2000, "std_scale": 1.1},
        "revol_util":       {"mean_shift": 8.0,  "std_scale": 1.2},
        "annual_inc":       {"mean_shift": -5000,"std_scale": 1.1},
    },
    "high": {
        "dti":              {"mean_shift": 10.0, "std_scale": 1.6},
        "int_rate":         {"mean_shift": 4.0,  "std_scale": 1.4},
        "loan_amnt":        {"mean_shift": 5000, "std_scale": 1.3},
        "revol_util":       {"mean_shift": 15.0, "std_scale": 1.5},
        "annual_inc":       {"mean_shift": -12000,"std_scale": 1.3},
        "credit_utilization": {"mean_shift": 0.1, "std_scale": 1.3},
    },
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Monitoring baseline and drift simulation")
    p.add_argument("--mode",           choices=["baseline", "simulate", "both"],
                   default="both")
    p.add_argument("--source",         default=DATA_SOURCE_NAME,
                   help="Domino Data Source name for training data")
    p.add_argument("--filename",       default=TRAINING_FILENAME,
                   help="Filename within the data source")
    p.add_argument("--sample-size",    type=int,   default=5000,
                   help="Number of rows to sample for baseline/simulation")
    p.add_argument("--shift-severity", choices=["low", "medium", "high"],
                   default="medium", help="Magnitude of simulated distribution shift")
    p.add_argument("--random-state",   type=int, default=42)
    p.add_argument("--model-id",       default=None,
                   help="Model Monitor model ID (overrides auto-detection)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load data from Domino Data Source (S3) or local fallback
# ---------------------------------------------------------------------------
def load_sample(source_name, filename, n, random_state):
    """
    Load training data from the named Domino Data Source.
    Falls back to object_store if injected; raises RuntimeError otherwise.
    Mirrors the pattern established in ingest.py.
    """
    try:
        import domino

        log.info(f"Connecting to Domino Data Source: {source_name}")
        client = domino.Domino(project=PROJECT_NAME)
        datasources = client.data_sources()

        ds = next((d for d in datasources if d["name"] == source_name), None)
        if ds is None:
            raise ValueError(
                f"Data source '{source_name}' not found. "
                f"Available: {[d['name'] for d in datasources]}"
            )

        log.info(f"Fetching '{filename}' from data source...")
        raw = ds["client"].get(filename)
        df  = pd.read_csv(StringIO(str(raw, "utf-8")), low_memory=False)
        log.info(f"Loaded {len(df):,} rows from data source '{source_name}'")

    except ImportError:
        # Domino SDK not available — use object_store if injected, else raise error
        # Mirrors the pattern established in ingest.py
        if "object_store" in dir():
            log.info(f"Using injected object_store to fetch '{filename}'...")
            raw = object_store.get(filename)
            df  = pd.read_csv(StringIO(str(raw, "utf-8")), low_memory=False)
        else:
            log.warning(
                "Domino SDK not available — add the Data Source connection snippet. "
                "Paste your object_store snippet below to connect to the data source."
            )
            # ── Paste Data Source snippet here ──────────────────────────────────
            # object_store = ...
            # raw = object_store.get(filename)
            # ────────────────────────────────────────────────────────────────────
            raise RuntimeError(
                "Could not connect to data source. "
                "Add the Data Source connection snippet to monitoring_baseline.py."
            )

    df = df.sample(min(n, len(df)), random_state=random_state).reset_index(drop=True)
    log.info(f"Sampled {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Mode 1: Compute baseline statistics
# ---------------------------------------------------------------------------
def compute_baseline(df: pd.DataFrame) -> dict:
    log.info("Computing baseline statistics...")
    os.makedirs(MONITORING_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = {}
    present = [f for f in MONITOR_FEATURES if f in df.columns]

    for feat in present:
        col = df[feat].dropna()
        stats[feat] = {
            "mean":   round(float(col.mean()), 4),
            "std":    round(float(col.std()),  4),
            "min":    round(float(col.min()),  4),
            "max":    round(float(col.max()),  4),
            "p25":    round(float(col.quantile(0.25)), 4),
            "p50":    round(float(col.quantile(0.50)), 4),
            "p75":    round(float(col.quantile(0.75)), 4),
            "p95":    round(float(col.quantile(0.95)), 4),
            "null_rate": round(float(df[feat].isnull().mean()), 4),
        }

    # Target distribution
    stats["__target__"] = {
        "default_rate": round(float(df["is_default"].mean()), 4),
        "n_samples": int(len(df)),
    }

    with open(BASELINE_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Baseline statistics saved to: {BASELINE_PATH}")

    _plot_baseline_distributions(df, present)
    return stats


def _plot_baseline_distributions(df, features):
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()
    sns.set_theme(style="darkgrid")

    for i, feat in enumerate(features):
        col = df[feat].dropna()
        axes[i].hist(col, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        axes[i].axvline(col.mean(), color="red", linestyle="--", linewidth=1.5,
                        label=f"mean={col.mean():.2f}")
        axes[i].set_title(feat, fontsize=10)
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Baseline Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "monitoring_baseline_distributions.png")
    plt.savefig(path, dpi=150)
    plt.show()
    log.info(f"Baseline distribution plot saved to: {path}")


# ---------------------------------------------------------------------------
# Mode 2: Simulate drift
# ---------------------------------------------------------------------------
def simulate_drift(df: pd.DataFrame, severity: str, random_state: int) -> pd.DataFrame:
    log.info(f"Simulating '{severity}' distribution shift...")
    rng = np.random.default_rng(random_state)

    df_shifted = df.copy()
    shift_cfg  = SHIFT_CONFIGS[severity]

    for feat, cfg in shift_cfg.items():
        if feat not in df_shifted.columns:
            log.warning(f"Feature '{feat}' not found — skipping shift")
            continue

        col = df_shifted[feat].copy()
        mean_shift  = cfg.get("mean_shift", 0)
        std_scale   = cfg.get("std_scale",  1.0)

        noise = rng.normal(loc=mean_shift, scale=col.std() * (std_scale - 1), size=len(col))
        df_shifted[feat] = col + noise

        if feat in ["dti", "revol_util", "credit_utilization"]:
            df_shifted[feat] = df_shifted[feat].clip(lower=0)
        if feat == "annual_inc":
            df_shifted[feat] = df_shifted[feat].clip(lower=1000)

        orig_mean = col.mean()
        new_mean  = df_shifted[feat].mean()
        log.info(f"  {feat}: mean {orig_mean:.2f} -> {new_mean:.2f}  "
                 f"(shift: {new_mean - orig_mean:+.2f})")

    os.makedirs(MONITORING_DIR, exist_ok=True)
    df_shifted.to_csv(DRIFT_DATA_PATH, index=False)
    log.info(f"Shifted dataset saved to: {DRIFT_DATA_PATH}")

    return df_shifted


# ---------------------------------------------------------------------------
# PSI (Population Stability Index) — drift detection metric
# ---------------------------------------------------------------------------
def compute_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    PSI < 0.1  : No significant change
    PSI 0.1-0.2: Moderate change — monitor
    PSI > 0.2  : Significant shift — investigate / retrain
    """
    def _bucket(s, edges):
        counts, _ = np.histogram(s, bins=edges)
        pct = counts / len(s)
        return np.where(pct == 0, 1e-4, pct)

    edges    = np.percentile(baseline.dropna(), np.linspace(0, 100, bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    base_pct = _bucket(baseline.dropna(), edges)
    curr_pct = _bucket(current.dropna(), edges)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return round(float(psi), 4)


def compute_all_psi(df_base: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    log.info("\nComputing PSI scores (drift detection)...")
    present = [f for f in MONITOR_FEATURES if f in df_base.columns and f in df_curr.columns]

    rows = []
    for feat in present:
        psi = compute_psi(df_base[feat], df_curr[feat])
        status = ("Stable" if psi < 0.1
                  else "Monitor" if psi < 0.2
                  else "Retrain")
        rows.append({"feature": feat, "psi": psi, "status": status})

    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    log.info("\n" + psi_df.to_string(index=False))
    return psi_df


def plot_psi_summary(psi_df: pd.DataFrame):
    colors = psi_df["psi"].apply(
        lambda v: "#e74c3c" if v >= 0.2 else "#f39c12" if v >= 0.1 else "#2ecc71"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(psi_df["feature"][::-1], psi_df["psi"][::-1], color=colors[::-1])
    ax.axvline(0.1, color="orange", linestyle="--", linewidth=1.5, label="Monitor threshold (0.1)")
    ax.axvline(0.2, color="red",    linestyle="--", linewidth=1.5, label="Retrain threshold (0.2)")
    ax.set_title("PSI Drift Report - Baseline vs Simulated Data", fontsize=12, fontweight="bold")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, psi_df["psi"][::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "monitoring_psi_report.png")
    plt.savefig(path, dpi=150)
    plt.show()
    log.info(f"PSI report saved to: {path}")


# ---------------------------------------------------------------------------
# Model Monitor API helpers
# ---------------------------------------------------------------------------
def _mm_headers():
    """Return auth headers for the Domino Model Monitor REST API."""
    api_key = DOMINO_USER_API_KEY
    if not api_key:
        raise EnvironmentError(
            "DOMINO_USER_API_KEY environment variable is not set. "
            "This is injected automatically in Domino workspaces and jobs."
        )
    return {
        "X-Domino-Api-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def get_model_id(model_name: str = "lending-club-sklearn-rf") -> str:
    """
    Look up the Model Monitor model ID by name via the REST API.
    Returns the model ID string.
    """
    url = f"{DOMINO_API_HOST}/model-monitor/v2/api/models?pageNumber=0&pageSize=50&numberOfLastChecksToFetch=1"
    resp = requests.get(url, headers=_mm_headers(), timeout=30)
    resp.raise_for_status()

    models = resp.json().get("modelDashboardItems", [])
    match = next((m for m in models if m["name"] == model_name), None)
    if match is None:
        names = [m["name"] for m in models]
        raise ValueError(
            f"Model '{model_name}' not found in Model Monitor. "
            f"Registered models: {names}"
        )
    log.info(f"Found Model Monitor model '{model_name}' with ID: {match['id']}")
    return match["id"]


def register_prediction_data(model_id: str, drift_csv_path: str, datasource_name: str):
    """
    Register the simulated drift CSV as prediction data in Domino Model Monitor.

    This follows the prediction config JSON format described in the Domino docs:
    https://docs.dominodatalab.com/en/latest/user_guide/86bc1f/drift-detection-for-monitored-models/

    The CSV is read from the same data source registered with the model, so
    Model Monitor can pull it directly without needing a file upload.
    """
    import csv

    # Read the drift CSV to get column names for the prediction config
    df_preview = pd.read_csv(drift_csv_path, nrows=1)
    feature_cols = [c for c in df_preview.columns if c in MONITOR_FEATURES]

    # Build the prediction config JSON — mirrors monitoring_config.json structure
    drift_filename = os.path.basename(drift_csv_path)

    prediction_config = {
        "variables": [
            {
                "name": feat,
                "valueType": "numerical",
                "variableType": "feature",
            }
            for feat in feature_cols
        ] + [
            {
                "name": "is_default",
                "valueType": "categorical",
                "variableType": "prediction",
            },
            {
                "name": "default_probability",
                "valueType": "numerical",
                "variableType": "prediction_probability",
            },
        ],
        "datasetDetails": {
            "name": drift_filename,
            "datasetType": "file",
            "datasetConfig": {
                "path": drift_filename,
                "fileFormat": "csv",
            },
            "datasourceName": datasource_name,
            "datasourceType": "s3",
        },
    }

    url = f"{DOMINO_API_HOST}/model-monitor/v2/api/model/{model_id}/prediction-data"
    log.info(f"Registering prediction data with Model Monitor...")
    log.info(f"  Model ID : {model_id}")
    log.info(f"  File     : {drift_filename}")
    log.info(f"  Source   : {datasource_name}")

    resp = requests.post(url, headers=_mm_headers(), json=prediction_config, timeout=60)

    if resp.status_code in (200, 201, 202):
        log.info(f"Prediction data registered successfully (HTTP {resp.status_code})")
        log.info(f"Response: {resp.json()}")
    else:
        log.error(f"Failed to register prediction data (HTTP {resp.status_code}): {resp.text}")
        resp.raise_for_status()

    return resp.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_sample(args.source, args.filename, args.sample_size, args.random_state)

    if args.mode in ("baseline", "both"):
        baseline_stats = compute_baseline(df)
        log.info(f"\nBaseline default rate : {baseline_stats['__target__']['default_rate']:.2%}")

    if args.mode in ("simulate", "both"):
        df_shifted = simulate_drift(df, args.shift_severity, args.random_state)
        psi_df     = compute_all_psi(df, df_shifted)
        plot_psi_summary(psi_df)

        n_retrain = (psi_df["psi"] >= 0.2).sum()
        n_monitor = ((psi_df["psi"] >= 0.1) & (psi_df["psi"] < 0.2)).sum()
        log.info(f"\nDrift summary ({args.shift_severity} shift):")
        log.info(f"  {n_retrain} feature(s) exceed retrain threshold (PSI >= 0.2)")
        log.info(f"  {n_monitor} feature(s) in monitor zone (0.1 <= PSI < 0.2)")

        # Register the shifted dataset as prediction data in Model Monitor
        if DOMINO_USER_API_KEY:
            try:
                model_id = args.model_id or get_model_id("lending-club-sklearn-rf")
                register_prediction_data(model_id, DRIFT_DATA_PATH, args.source)
            except Exception as e:
                log.warning(f"Could not register prediction data with Model Monitor: {e}")
                log.warning("The drift simulation CSV has been saved locally. "
                            "You can register it manually via the Model Monitor UI.")
        else:
            log.warning("DOMINO_USER_API_KEY not set — skipping Model Monitor API registration.")
            log.warning(f"Drift CSV saved to: {DRIFT_DATA_PATH}")
            log.warning("Register it manually: Model Monitor > Data Drift > Register Prediction.")


if __name__ == "__main__":
    main()
