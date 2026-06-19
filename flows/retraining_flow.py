"""
retraining_flow.py
==================
Simple Domino Flow demonstrating model evaluation from MLflow.

This workflow assumes training has already been completed in previous labs.
It queries MLflow for trained models and selects the best one.

DAG:
    [evaluate]

Register and run from the Workspace terminal:
    pyflyte run --remote flows/retraining_flow.py model_evaluation_flow

Then watch progress under Flows > model_evaluation_flow > <run name>.
"""

from typing import TypeVar

from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekitplugins.domino.helpers import Input, Output, run_domino_job_task

# Domino compute environment used for every step
ENVIRONMENT_NAME = "LendingClubProject-TrainingEnvironment"

JSON = TypeVar("json")


@workflow
def model_evaluation_flow(
    auc_threshold: float = 0.80,
):
    """
    Simple 1-step workflow: evaluate models from MLflow and select the best.

    This assumes sklearn, xgboost, and h2o models have already been trained
    in previous labs and logged to MLflow.

    The evaluation result is written to results/evaluation_result.json
    in the job execution and can be viewed in the job artifacts.
    """

    # ----------------------------------------------------------------------
    # Evaluate models from MLflow and select the best
    # ----------------------------------------------------------------------
    run_domino_job_task(
        flyte_task_name="Evaluate & Select Best Model",
        command="python scripts/evaluate.py --auc-threshold 0.80",
        inputs=[],
        output_specs=[],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )
