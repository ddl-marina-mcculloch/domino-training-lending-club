# Lab: S3 Data Ingestion with Domino Flows

## Overview
This short lab demonstrates how to use Domino Flows to orchestrate data ingestion from S3 using Domino Data Sources.

**Duration:** 10-15 minutes

**Prerequisites:**
- S3 Data Source configured in your Domino project
- `lending_raw.csv` uploaded to your S3 bucket
- Data Source name: `LendingClubWorkshop` (or update the flow)

---

## Architecture

This Flow has 2 simple steps:

```
[Ingest from S3] → [Validate Data]
```

1. **Ingest from S3**: Uses `scripts/ingest_s3.py` to read data from S3 via Domino Data Source
2. **Validate Data**: Uses `scripts/validate_s3_data.py` to perform data quality checks

---

## Lab Steps

### Step 1: Verify S3 Data Source Setup

1. Navigate to **Data > Data Sources** in your Domino project
2. Confirm you see a Data Source named `LendingClubWorkshop` (or note the actual name)
3. Verify it's connected to your S3 bucket with `lending_raw.csv`

**If you need to create the Data Source:**
- Click **Add Data Source**
- Select **Amazon S3**
- Enter your S3 bucket details and credentials
- Name it `LendingClubWorkshop`
- Test the connection

---

### Step 2: Review the Flow Code

Open [flows/s3_data_flow.py](s3_data_flow.py) and review:

- **Workflow function**: `s3_data_flow`
- **Parameters**:
  - `data_source_name` (default: `"LendingClubWorkshop"`)
  - `s3_key` (default: `"lending_raw.csv"`)
- **Two tasks**: Ingest → Validate
- **Output**: Validation report JSON

---

### Step 3: Run the S3 Flow

From your Domino Workspace terminal:

```bash
pyflyte run --remote flows/s3_data_flow.py s3_data_flow
```

**Optional**: Customize the Data Source name or S3 key:

```bash
pyflyte run --remote flows/s3_data_flow.py s3_data_flow \
  --data_source_name "YourDataSourceName" \
  --s3_key "your_file.csv"
```

---

### Step 4: Monitor the Flow

1. The command will output a URL like:
   ```
   https://your-domino-instance/flows/redirect/[execution-name]
   ```

2. Click the link or navigate to **Flows** in the left sidebar

3. You'll see the execution with two steps:
   - **Ingest Data from S3** (running first)
   - **Validate Data Quality** (waits for ingest to complete)

4. Click into each step to view:
   - Real-time logs
   - Job status
   - Output artifacts

---

### Step 5: Review the Results

Once the Flow completes:

1. Navigate to **Artifacts > results/** in your project

2. You'll find two output files:
   - `s3_ingest_summary.json` - Ingestion metadata (rows, columns, etc.)
   - `s3_validation_report.json` - Data quality report

3. Open `s3_validation_report.json` to see:
   - Total rows and columns
   - Missing value analysis
   - Data type summary
   - Numeric statistics (mean, median, min, max)
   - Validation status (PASSED/FAILED)
   - Any data quality issues detected

---

## Expected Output

### Ingest Summary (`s3_ingest_summary.json`):
```json
{
  "data_source": "LendingClubWorkshop",
  "s3_key": "lending_raw.csv",
  "rows": 50000,
  "columns": 75,
  "column_names": ["loan_status", "loan_amnt", "int_rate", ...],
  "missing_required_cols": [],
  "output_path": "results/s3_ingest_data.csv"
}
```

### Validation Report (`s3_validation_report.json`):
```json
{
  "total_rows": 50000,
  "total_columns": 75,
  "validation_passed": true,
  "issues": [],
  "missing_values": { ... },
  "numeric_summary": {
    "loan_amnt": { "mean": 14755.2, "median": 13000.0, ... },
    "int_rate": { "mean": 13.67, "median": 13.11, ... }
  }
}
```

---

## Key Takeaways

✅ **Domino Flows** orchestrate multi-step data pipelines
✅ **Data Sources** provide secure, managed access to S3
✅ **Task dependencies** ensure proper execution order
✅ **JSON outputs** enable downstream tasks to consume results
✅ **Real-time monitoring** shows pipeline progress and logs

---

## Troubleshooting

### Issue: "Data Source not found"
- **Solution**: Update `data_source_name` parameter to match your actual Data Source name
- Check available Data Sources in **Data > Data Sources**

### Issue: "ImportError: domino_data"
- **Solution**: Verify your environment has `dominodatalab-data` package installed
- Check **Environments** and ensure the package is in your compute environment

### Issue: "File not found in S3"
- **Solution**: Verify the `s3_key` parameter matches the actual file name in your S3 bucket
- Check S3 bucket contents via AWS Console or Data Source preview

---

## Next Steps

- Extend the Flow to add a **preprocessing** step after validation
- Add a **training** step that uses the validated S3 data
- Schedule the Flow to run on a recurring basis (daily/weekly)
- Integrate with the existing `model_evaluation_flow` for end-to-end retraining

---

## Related Files

- Flow definition: [flows/s3_data_flow.py](s3_data_flow.py)
- Ingest script: [scripts/ingest_s3.py](../scripts/ingest_s3.py)
- Validation script: [scripts/validate_s3_data.py](../scripts/validate_s3_data.py)
