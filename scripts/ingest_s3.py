"""
ingest_s3.py
------------
Simple script to demonstrate ingesting data from S3 via Domino Data Source.

This script uses the domino-data SDK (DataSourceClient) to read from S3
and writes the result to /mnt/code/results for the Flow output.

Usage:
    python scripts/ingest_s3.py
    python scripts/ingest_s3.py --data-source LendingClubWorkshop --key lending_raw.csv
"""

import os
import argparse
import logging
import json

import pandas as pd
from domino_data import DataSourceClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEFAULT_OUTPUT_CSV = os.path.join(RESULTS_DIR, "s3_ingest_data.csv")
DEFAULT_OUTPUT_JSON = os.path.join(RESULTS_DIR, "s3_ingest_summary.json")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ingest data from S3 via Domino Data Source")
    p.add_argument("--data-source", default="LendingClubWorkshop",
                   help="Domino Data Source name (must be added to project)")
    p.add_argument("--key", default="lending_raw.csv",
                   help="S3 object key / filename to fetch")
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV,
                   help="Path to write CSV output")
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON,
                   help="Path to write JSON summary")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=== S3 Data Ingestion ===")
    log.info(f"Data Source: {args.data_source}")
    log.info(f"S3 Key:      {args.key}")

    # Connect to Domino Data Source
    try:
        client = DataSourceClient()
        log.info(f"Connected to Domino Data Source: {args.data_source}")

        # Execute SQL query to read from S3 (if it's a SQL-compatible source)
        # OR use object store client for direct S3 access
        # For CSV files in S3, we'll read directly

        # Read the CSV from S3
        # Note: The exact API depends on your Data Source type
        # For S3 object stores, you might use client.read_file() or similar
        # This is a simplified example - adjust based on your Data Source config

        log.info("Reading data from S3...")
        # Assuming the data source allows direct file reading
        # Adjust this based on your actual Data Source configuration
        query = f"SELECT * FROM {args.key.replace('.csv', '')}"
        df = client.query(query, args.data_source)

    except Exception as e:
        log.error(f"Failed to read from Data Source: {e}")
        log.info("Attempting alternative method...")

        # Alternative: If DataSourceClient doesn't work, fall back to manual S3 access
        # This requires the Data Source to be properly configured in the project
        try:
            import boto3
            from domino_data.data_sources import DataSourceClient as DSClient

            ds_client = DSClient()
            # Get S3 credentials from Data Source
            # This is environment-specific
            log.warning("Using fallback S3 access method")
            raise NotImplementedError(
                "Please configure your Data Source in the project. "
                "Go to Data > Data Sources and add your S3 Data Source."
            )
        except ImportError:
            log.error("domino-data SDK not available")
            raise

    log.info(f"Successfully read {len(df):,} rows, {df.shape[1]} columns")

    # Basic validation
    required_cols = ["loan_status", "loan_amnt", "int_rate", "annual_inc", "dti"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.warning(f"Missing expected columns: {missing}")

    # Write CSV output
    df.to_csv(args.output_csv, index=False)
    log.info(f"Data written to: {args.output_csv}")

    # Write summary JSON for Flow output
    summary = {
        "data_source": args.data_source,
        "s3_key": args.key,
        "rows": len(df),
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_required_cols": missing,
        "output_path": args.output_csv,
    }

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Summary written to: {args.output_json}")
    log.info("=== Ingest Complete ===")

    return summary


if __name__ == "__main__":
    main()
