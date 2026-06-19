"""
s3_data_flow.py
===============
Simple Domino Flow demonstrating S3 data ingestion via Domino Data Source.

This workflow shows how to:
1. Ingest data from S3 using Domino Data Source
2. Validate the ingested data quality

Prerequisites:
- S3 Data Source configured in project (Data > Data Sources)
- Data Source name: "LendingClubWorkshop" (or update below)
- S3 file: lending_raw.csv

DAG:
    [Ingest from S3] -> [Validate Data]

Register and run from the Workspace terminal:
    pyflyte run --remote flows/s3_data_flow.py s3_data_flow

Then watch progress under Flows > s3_data_flow > <run name>.
"""

from typing import TypeVar

from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekitplugins.domino.helpers import Input, Output, run_domino_job_task

# Domino compute environment
ENVIRONMENT_NAME = "LendingClubProject-TrainingEnvironment"

JSON = TypeVar("json")


@workflow
def s3_data_flow(
    data_source_name: str = "LendingClubWorkshop",
    s3_key: str = "lending_raw.csv",
) -> FlyteFile[JSON]:
    """
    Simple 2-step S3 data workflow.

    Step 1: Ingest data from S3 via Domino Data Source
    Step 2: Validate data quality and output report
    """

    # ----------------------------------------------------------------------
    # Step 1 - Ingest from S3
    # ----------------------------------------------------------------------
    ingest = run_domino_job_task(
        flyte_task_name="Ingest Data from S3",
        command="python scripts/ingest_s3.py --data-source LendingClubWorkshop --key lending_raw.csv",
        inputs=[],
        output_specs=[Output(name="ingest_summary", type=FlyteFile[JSON])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    # ----------------------------------------------------------------------
    # Step 2 - Validate ingested data
    # ----------------------------------------------------------------------
    validate = run_domino_job_task(
        flyte_task_name="Validate Data Quality",
        command="python scripts/validate_s3_data.py",
        inputs=[
            Input(name="ingest_summary", type=FlyteFile[JSON], value=ingest["ingest_summary"]),
        ],
        output_specs=[Output(name="validation_report", type=FlyteFile[JSON])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    return validate["validation_report"]
