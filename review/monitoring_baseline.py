"""
monitoring_baseline.py
----------------------
Generates baseline statistics and simulates prediction drift
for the LendingClub credit risk model in Domino Model Monitor.

Two modes:
    1. baseline   — computes and saves baseline statistics from the training set
    2. simulate   — generates a shifted dataset and saves it as a CSV for
                    manual registration in Domino Model Monitor as prediction data

Usage:
    python scripts/monitoring_baseline.py --mode baseline
    python scripts/monitoring_baseline.py --mode simulate --shift-severity medium
    python scripts/monitoring_baseline.py --mode both --shift-severity high

Data source:
    Reads training data from S3 using boto3. boto3 picks up AWS credentials
    automatically from environment variables (AWS_ACCESS_KEY_ID /
    AWS_SECRET_ACCESS_KEY) which Domino injects when the Data Source is
    connected, or from the AWS credential file (AWS_SHARED_CREDENTIALS_FILE).

    Set S3_BUCKET and S3_REGION below (or via environment variables) to match
    the bucket configured in your Domino Data Source.

Model Monitor integration:
    After running --mode simulate, upload the generated drift CSV to Model Monitor
    manually via: Model Monitor > Data Drift > Register Prediction > Upload Prediction Config
    The CSV is saved to: monitoring/drift_simulation.csv
"""

import os
import io
import argparse
import logging
import json

import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
MONITORING_DIR  = os.path.join(os.path.dirname(__file__), "..", "monitoring")
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "..", "results")
BASELINE_PATH   = os.path.join(MONITORING_DIR, "baseline_stats.json")
DRIFT_DATA_PATH = os.path.join(MONITORING_DIR, "drift_simulation.csv")

# ---------------------------------------------------------------------------
# S3 configuration — update to match your Domino Data Source bucket
# ---------------------------------------------------------------------------
S3_BUCKET         = os.environ.get("S3_BUCKET", "lending-club-mm")
S3_REGION         = os.environ.get("S3_REGION", "us-west-2")
TRAINING_FILENAME = "lending_clean.csv"

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
        "loan_to_income":   {"mean_shift": 0.05, "std_scale": 1.2},
    },
    "high": {
        "dti":              {"mean_shift": 10.0, "std_scale": 1.6},
        "int_rate":         {"mean_shift": 4.0,  "std_scale": 1.4},
        "loan_amnt":        {"mean_shift": 5000, "std_scale": 1.3},
        "revol_util":       {"mean_shift": 15.0, "std_scale": 1.5},
        "annual_inc":       {"mean_shift": -10000,"std_scale": 1.3},
        "loan_to_income":   {"mean_shift": 0.10, "std_scale": 1.4},
        "credit_utilization": {"mean_shift": 10.0, "std_scale": 1.4},
        "payment_to_income":  {"mean_shift": 0.05, "std_scale": 1.3},
    },
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Monitoring baseline and drift simulation")
    p.add_argument("--mode",           choices=["baseline", "simulate", "both"],
                   default="both")
    p.add_argument("--bucket",         default=S3_BUCKET,
                   help="S3 bucket name (must match your Domino Data Source)")
    p.add_argument("--region",         default=S3_REGION,
                   help="AWS region of the S3 bucket")
    p.add_argument("--filename",       default=TRAINING_FILENAME,
                   help="Object key / filename within the S3 bucket")
    p.add_argument("--sample-size",    type=int, default=5000,
                   help="Number of rows to sample for baseline/simulation")
    p.add_argument("--shift-severity", choices=["low", "medium", "high"],
                   default="medium", help="Magnitude of simulated distribution shift")
    p.add_argument("--random-state",   type=int, default=42)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Load data from S3 using boto3
# ---------------------------------------------------------------------------
def load_data(bucket, region, filename, n, random_state):
    """
    Download training data from S3 using boto3.

    boto3 authenticates using standard AWS credential sources in order:
      1. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
         (Domino injects these when the Data Source is connected to the project)
      2. AWS credential file at AWS_SHARED_CREDENTIALS_FILE
         (populated automatically by Domino's credential propagation feature)

    No explicit credentials are needed in the code.
    """
    log.info(f"Connecting to S3 bucket '{bucket}' in region '{region}'...")
    # Create a new S3 client. boto3 picks up AWS credentials automatically
    # from the environment variables / credential file that Domino injects
    # when the S3 Data Source is connected (see Domino "Connect to Amazon S3"
    # docs -> "Python and boto3"). No explicit credentials are passed here.
    client = boto3.client("s3", region_name=region)

    # Download the object from the bucket to a local file, following the
    # boto3 download_file pattern recommended in the Domino documentation.
    local_path = os.path.join(MONITORING_DIR, os.path.basename(filename))
    os.makedirs(MONITORING_DIR, exist_ok=True)
    log.info(f"Downloading '{filename}' -> '{local_path}'...")
    client.download_file(bucket, filename, local_path)

    df  = pd.read_csv(local_path, low_memory=False)
    log.info(f"Loaded {len(df):,} rows from s3://{bucket}/{filename}")

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
            "mean":      round(float(col.mean()), 4),
            "std":       round(float(col.std()),  4),
            "min":       round(float(col.min()),  4),
            "max":       round(float(col.max()),  4),
            "p25":       round(float(col.quantile(0.25)), 4),
            "p50":       round(float(col.quantile(0.50)), 4),
            "p75":       round(float(col.quantile(0.75)), 4),
            "p95":       round(float(col.quantile(0.95)), 4),
            "null_rate": round(float(df[feat].isnull().mean()), 4),
        }

    # Target distribution
    stats["__target__"] = {
        "default_rate": round(float(df["is_default"].mean()), 4),
        "n_samples":    int(len(df)),
    }

    with open(BASELINE_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Baseline statistics saved to: {BASELINE_PATH}")

    _plot_baseline_distributions(df, present)
    return stats


def _plot_baseline_distributions(df, features):
    n    = len(features)
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

        col        = df_shifted[feat].copy()
        mean_shift = cfg.get("mean_shift", 0)
        std_scale  = cfg.get("std_scale",  1.0)

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

    edges     = np.percentile(baseline.dropna(), np.linspace(0, 100, bins + 1))
    edges[0]  -= 1e-6
    edges[-1] += 1e-6

    base_pct = _bucket(baseline.dropna(), edges)
    curr_pct = _bucket(current.dropna(), edges)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return round(float(psi), 4)


def compute_all_psi(df_base: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    log.info("\nComputing PSI scores (drift detection)...")
    present = [f for f in MONITOR_FEATURES if f in df_base.columns and f in df_curr.columns]
    records = []
    for feat in present:
        psi_val = compute_psi(df_base[feat], df_curr[feat])
        status  = "retrain" if psi_val >= 0.2 else ("monitor" if psi_val >= 0.1 else "ok")
        records.append({"feature": feat, "psi": psi_val, "status": status})
        log.info(f"  {feat:25s}: PSI={psi_val:.4f}  [{status}]")
    return pd.DataFrame(records).sort_values("psi", ascending=False)


def plot_psi_summary(psi_df: pd.DataFrame):
    colours = psi_df["status"].map({"ok": "green", "monitor": "orange", "retrain": "red"})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(psi_df["feature"], psi_df["psi"], color=colours)
    ax.axhline(0.1, color="orange", linestyle="--", linewidth=1.2, label="Monitor threshold (0.1)")
    ax.axhline(0.2, color="red",    linestyle="--", linewidth=1.2, label="Retrain threshold (0.2)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("PSI")
    ax.set_title("Feature Drift — Population Stability Index")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "psi_summary.png")
    plt.savefig(path, dpi=150)
    plt.show()
    log.info(f"PSI summary chart saved to: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(args.bucket, args.region, args.filename, args.sample_size, args.random_state)

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
        log.info(f"\nDrift simulation CSV saved to: {DRIFT_DATA_PATH}")
        log.info("Next step: register this file in Model Monitor.")
        log.info("  Model Monitor > Data Drift > Register Prediction > Upload Prediction Config")


if __name__ == "__main__":
    main()
