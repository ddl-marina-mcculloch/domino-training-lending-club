"""
validate_s3_data.py
-------------------
Simple validation script for S3-ingested data.

Reads the CSV from the previous ingest step and performs basic
data quality checks, then outputs a validation report.

Usage:
    python scripts/validate_s3_data.py
    python scripts/validate_s3_data.py --input results/s3_ingest_data.csv
"""

import os
import argparse
import logging
import json

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEFAULT_INPUT = os.path.join(RESULTS_DIR, "s3_ingest_data.csv")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "s3_validation_report.json")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Validate S3-ingested data")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Path to input CSV from ingest step")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Path to write validation report JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    log.info("=== S3 Data Validation ===")
    log.info(f"Input: {args.input}")

    # Read data
    df = pd.read_csv(args.input, low_memory=False)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Data quality checks
    checks = {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "missing_values": {},
        "data_types": {},
        "numeric_summary": {},
        "validation_passed": True,
        "issues": []
    }

    # Check for missing values
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        checks["missing_values"][col] = {
            "count": int(missing_count),
            "percentage": round(missing_pct, 2)
        }
        checks["data_types"][col] = str(df[col].dtype)

    # Check required columns
    required_cols = ["loan_status", "loan_amnt", "int_rate", "annual_inc", "dti"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        checks["validation_passed"] = False
        checks["issues"].append(f"Missing required columns: {missing_required}")

    # Numeric summaries for key columns
    numeric_cols = ["loan_amnt", "int_rate", "annual_inc", "dti"]
    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            checks["numeric_summary"][col] = {
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "std": round(float(df[col].std()), 2),
            }

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        checks["issues"].append(f"Found {duplicates} duplicate rows")
        log.warning(f"⚠️  {duplicates} duplicate rows detected")

    # Log results
    log.info("\n=== Validation Summary ===")
    log.info(f"Total rows:    {checks['total_rows']:,}")
    log.info(f"Total columns: {checks['total_columns']}")
    log.info(f"Validation:    {'✅ PASSED' if checks['validation_passed'] else '❌ FAILED'}")

    if checks["issues"]:
        log.warning(f"Issues found:  {len(checks['issues'])}")
        for issue in checks["issues"]:
            log.warning(f"  - {issue}")
    else:
        log.info("No issues found ✅")

    # Write report
    with open(args.output, "w") as f:
        json.dump(checks, f, indent=2)

    log.info(f"\nValidation report written to: {args.output}")
    log.info("=== Validation Complete ===")

    return checks


if __name__ == "__main__":
    main()
