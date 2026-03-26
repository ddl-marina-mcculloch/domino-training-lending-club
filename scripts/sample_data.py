"""
create_sample_dataset.py
------------------------
One-off script to generate a 50k row sample from the full LendingClub
dataset for use in the Domino training workshop.

Run this locally before uploading to S3 / your Domino Data Source.

Download the full dataset from:
    https://www.kaggle.com/datasets/wordsforthewise/lending-club

The full dataset is typically split across two CSVs:
    - accepted_2007_to_2018Q4.csv   (~2.2M rows)
    - rejected_2007_to_2018Q4.csv   (rejected applications — not used)

Usage:
    python scripts/create_sample_dataset.py --input accepted_2007_to_2018Q4.csv
    python scripts/create_sample_dataset.py --input accepted_2007_to_2018Q4.csv --n 100000
"""

import argparse
import logging
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns to keep
# Trimmed to the features used in training + key metadata
# Keeps the file size manageable for a workshop setting
# ---------------------------------------------------------------------------
KEEP_COLS = [
    # Target
    "loan_status",

    # Loan details
    "loan_amnt",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "term",
    "purpose",
    "title",

    # Applicant
    "annual_inc",
    "verification_status",
    "home_ownership",
    "emp_length",
    "addr_state",
    "zip_code",

    # Credit profile
    "dti",
    "revol_bal",
    "revol_util",
    "open_acc",
    "total_acc",
    "pub_rec",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mort_acc",
    "open_rv_12m",
    "bc_util",
    "num_bc_tl",

    # Metadata
    "id",
    "member_id",
    "issue_d",
    "url",
    "desc",
    "policy_code",
]

# Resolved loan statuses to filter to
RESOLVED_STATUSES = ["Fully Paid", "Charged Off", "Default"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate workshop sample dataset from LendingClub CSV")
    p.add_argument("--input",          required=True,
                   help="Path to the full LendingClub accepted loans CSV")
    p.add_argument("--output",         default="data/lending_sample.csv",
                   help="Output path for the sample CSV")
    p.add_argument("--n",              type=int, default=50000,
                   help="Number of rows to sample (default: 50,000)")
    p.add_argument("--random-state",   type=int, default=42)
    p.add_argument("--resolved-only",  action="store_true", default=True,
                   help="Filter to resolved loans only before sampling")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    log.info(f"Loading full dataset from: {args.input}")
    log.info("This may take a moment for the full 2.2M row file...")

    # Read in chunks to handle large file
    chunks = []
    chunk_size = 100_000
    total_rows = 0

    for chunk in pd.read_csv(args.input, low_memory=False, chunksize=chunk_size,
                              skiprows=1,  # LendingClub CSVs have a note row at top
                              header=0):
        total_rows += len(chunk)
        if args.resolved_only:
            chunk = chunk[chunk["loan_status"].isin(RESOLVED_STATUSES)]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Total rows loaded   : {total_rows:,}")
    log.info(f"After status filter : {len(df):,}")

    # Keep only relevant columns (ignore missing ones gracefully)
    cols = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        log.warning(f"Columns not found in source file (will be skipped): {missing}")
    df = df[cols]

    # Stratified sample — preserve default rate
    n = min(args.n, len(df))
    df_sample = df.groupby("loan_status", group_keys=False).apply(
        lambda x: x.sample(frac=n / len(df), random_state=args.random_state)
    ).reset_index(drop=True)

    # Trim to exactly n rows in case of rounding
    df_sample = df_sample.sample(n=min(n, len(df_sample)),
                                 random_state=args.random_state).reset_index(drop=True)

    # Summary
    default_rate = df_sample["loan_status"].isin(["Charged Off", "Default"]).mean()
    log.info(f"Sample size         : {len(df_sample):,}")
    log.info(f"Default rate        : {default_rate:.2%}")
    log.info(f"Columns             : {df_sample.shape[1]}")

    # Save
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_sample.to_csv(args.output, index=False)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    log.info(f"Sample saved to     : {args.output}  ({size_mb:.1f} MB)")

    log.info("\n=== Next steps ===")
    log.info(f"1. Upload {args.output} to your S3 bucket as 'lending_raw.csv'")
    log.info("2. Ensure the S3 bucket is configured as the 'LendingClubWorkshop' Data Source in Domino")
    log.info("3. Run scripts/preprocess.py to generate lending_clean.csv")


if __name__ == "__main__":
    main()
