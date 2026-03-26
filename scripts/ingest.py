"""
ingest.py
---------
Step 1 of the Domino Flows retraining pipeline.
Pulls the latest LendingClub loan data from the S3 data source
and writes it to the Domino Dataset as lending_raw.csv.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --source LendingClubWorkshop --filename lending_raw.csv
"""

import os
import argparse
import logging
from io import StringIO

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME", "LendingClubProject")
DEFAULT_OUTPUT = f"/domino/datasets/local/{PROJECT_NAME}/lending_raw.csv"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ingest raw LendingClub data from S3 data source")
    p.add_argument("--source",   default="LendingClubWorkshop",
                   help="Domino Data Source name")
    p.add_argument("--filename", default="lending_raw.csv",
                   help="Filename within the data source")
    p.add_argument("--output",   default=DEFAULT_OUTPUT,
                   help="Output path in Domino Dataset")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    log.info(f"Connecting to data source: {args.source}")
    log.info(f"Fetching file: {args.filename}")

    # Connect to Domino Data Source
    # The object_store variable is injected by Domino when a Data Source
    # is added to the project — this mirrors the pattern used in the EDA notebook
    try:
        from domino_data.vectordb import domino_is_connected
        import domino

        client = domino.Domino(project=PROJECT_NAME)
        datasources = client.data_sources()

        # Get the named data source
        ds = next((d for d in datasources if d["name"] == args.source), None)
        if ds is None:
            raise ValueError(
                f"Data source '{args.source}' not found. "
                f"Available: {[d['name'] for d in datasources]}"
            )

        # Read raw CSV from S3
        log.info("Reading CSV from data source...")
        raw = ds["client"].get(args.filename)
        s   = str(raw, "utf-8")
        df  = pd.read_csv(StringIO(s), low_memory=False)

    except ImportError:
        # Fallback for environments where domino SDK is not available
        # In this case the data source connection snippet is used directly
        # as shown in Lab 2.1 — paste your object_store snippet here
        log.warning(
            "Domino SDK not available — using object_store directly. "
            "Paste your Data Source connection snippet below."
        )
        # ── Paste Data Source snippet here ──────────────────────────────────
        # object_store = ...
        # raw = object_store.get(args.filename)
        # ────────────────────────────────────────────────────────────────────
        raise RuntimeError(
            "Could not connect to data source. "
            "Add the Data Source connection snippet to ingest.py line 83."
        )

    log.info(f"Fetched {len(df):,} rows, {df.shape[1]} columns")

    # Basic validation
    required_cols = ["loan_status", "loan_amnt", "int_rate", "annual_inc", "dti"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in raw data: {missing}")

    # Write to Domino Dataset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info(f"Raw data written to: {args.output}")

    # Summary
    log.info("=== Ingest Summary ===")
    log.info(f"Rows      : {len(df):,}")
    log.info(f"Columns   : {df.shape[1]}")
    log.info(f"Output    : {args.output}")


if __name__ == "__main__":
    main()
