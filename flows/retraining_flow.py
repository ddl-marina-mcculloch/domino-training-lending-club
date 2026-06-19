"""
retraining_flow.py
==================
LendingClub credit-risk model retraining pipeline, defined as a Domino Flow.

This is the Python (flytekit) definition of a Flows. Domino Flows are authored in Python using
flytekit and the Domino flytekit plugin, not YAML -- see:
https://docs.dominodatalab.com/en/latest/user_guide/5b5259/get-started-with-flows/

DAG:

    [ingest] -> [preprocess] -> +-> [train_sklearn] -+
                                +-> [train_xgboost] -+-> [evaluate] -> [decision]
                                +-> [train_h2o]    --+

The three training tasks run in parallel after preprocessing (Flows derives
this automatically from the data dependencies). `evaluate` waits for all three.
The final `decision` task promotes the best model if AUC > 0.80, otherwise it
raises an alert and holds -- the AUC threshold logic lives in the step scripts
(evaluate.py / promote.py / alert.py), keeping the workflow graph static as
flytekit requires.

Register and run from the Workspace terminal:

    pyflyte run --remote flows/retraining_flow.py retraining_flow

Then watch progress under Flows > retraining_flow > <run name>.
"""

from typing import TypeVar

from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekitplugins.domino.helpers import Input, Output, run_domino_job_task

# Domino compute environment used for every step (override per task if needed).
ENVIRONMENT_NAME = "LendingClubProject-TrainingEnvironment"

CSV = TypeVar("csv")


@workflow
def retraining_flow(
    s3_bucket: str = "lending-club",
    s3_region: str = "us-west-2",
) -> FlyteFile[CSV]:
    """End-to-end retraining pipeline for the LendingClub credit-risk model."""

    # ----------------------------------------------------------------------
    # Step 1 - Ingest raw loan data from S3
    # ----------------------------------------------------------------------
    ingest = run_domino_job_task(
        flyte_task_name="Ingest Raw Data from S3",
        command="python scripts/ingest.py",
        inputs=[
            Input(name="bucket", type=str, value=s3_bucket),
            Input(name="region", type=str, value=s3_region),
        ],
        output_specs=[Output(name="lending_raw", type=FlyteFile[CSV])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    # ----------------------------------------------------------------------
    # Step 2 - Preprocess / feature engineering (creates the is_default target)
    # ----------------------------------------------------------------------
    preprocess = run_domino_job_task(
        flyte_task_name="Feature Engineering & Preprocessing",
        command="python scripts/preprocess.py",
        inputs=[Input(name="lending_raw", type=FlyteFile[CSV], value=ingest["lending_raw"])],
        output_specs=[Output(name="lending_clean", type=FlyteFile[CSV])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    clean = preprocess["lending_clean"]

    # ----------------------------------------------------------------------
    # Steps 3-5 - Train three frameworks in parallel (sklearn, XGBoost, H2O)
    # ----------------------------------------------------------------------
    train_sklearn = run_domino_job_task(
        flyte_task_name="Train sklearn",
        command="python scripts/train_sklearn.py",
        inputs=[Input(name="lending_clean", type=FlyteFile[CSV], value=clean)],
        output_specs=[Output(name="sklearn_model", type=FlyteFile[TypeVar("pkl")])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    train_xgboost = run_domino_job_task(
        flyte_task_name="Train XGBoost",
        command="python scripts/train_xgboost.py",
        inputs=[Input(name="lending_clean", type=FlyteFile[CSV], value=clean)],
        output_specs=[Output(name="xgboost_model", type=FlyteFile[TypeVar("pkl")])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    train_h2o = run_domino_job_task(
        flyte_task_name="Train H2O AutoML",
        command="python scripts/train_h2o.py",
        inputs=[Input(name="lending_clean", type=FlyteFile[CSV], value=clean)],
        output_specs=[Output(name="h2o_model", type=FlyteFile[TypeVar("zip")])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Large",  # H2O benefits from more memory
        use_project_defaults_for_omitted=True,
    )

    # ----------------------------------------------------------------------
    # Step 6 - Evaluate and select the best model (waits for all 3 trainers)
    # ----------------------------------------------------------------------
    evaluate = run_domino_job_task(
        flyte_task_name="Evaluate & Select Best Model",
        command="python scripts/evaluate.py",
        inputs=[
            Input(name="sklearn_model", type=FlyteFile[TypeVar("pkl")], value=train_sklearn["sklearn_model"]),
            Input(name="xgboost_model", type=FlyteFile[TypeVar("pkl")], value=train_xgboost["xgboost_model"]),
            Input(name="h2o_model", type=FlyteFile[TypeVar("zip")], value=train_h2o["h2o_model"]),
        ],
        output_specs=[Output(name="evaluation_result", type=FlyteFile[TypeVar("json")])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    # ----------------------------------------------------------------------
    # Step 7 - Conditional promote / alert.
    # The script inspects evaluation_result: AUC > 0.80 -> promote endpoint,
    # AUC <= 0.80 -> alert & hold. Threshold logic lives in the script so the
    # flytekit workflow graph stays static.
    # ----------------------------------------------------------------------
    decision = run_domino_job_task(
        flyte_task_name="Promote or Alert",
        command="python scripts/promote.py",
        inputs=[Input(name="evaluation_result", type=FlyteFile[TypeVar("json")], value=evaluate["evaluation_result"])],
        output_specs=[Output(name="decision_result", type=FlyteFile[TypeVar("json")])],
        environment_name=ENVIRONMENT_NAME,
        hardware_tier_name="Small",
        use_project_defaults_for_omitted=True,
    )

    return decision["decision_result"]
