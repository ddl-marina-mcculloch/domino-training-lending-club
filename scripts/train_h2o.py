"""ls -la

train_h2o.py
------------
Trains an H2O AutoML model on the cleaned LendingClub dataset.
Logs parameters, metrics, and model artifact to MLflow.

H2O AutoML searches over GBM, Random Forest, Deep Learning, GLM
and stacked ensembles, selecting the best model by AUC.

Usage:
    python scripts/train_h2o.py
    python scripts/train_h2o.py --max-models 20 --max-runtime-secs 300
"""

import os
import argparse
import logging
import pickle
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME", "LendingClubProject")
DATA_PATH    = f"/mnt/data/{PROJECT_NAME}/lending_clean.csv"
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "h2o_best_model.pkl")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train H2O AutoML model for loan default prediction")
    p.add_argument("--data",             default=DATA_PATH)
    p.add_argument("--max-models",       type=int,   default=15,
                   help="Maximum number of models for AutoML to train")
    p.add_argument("--max-runtime-secs", type=int,   default=120,
                   help="Wall-clock time budget in seconds for AutoML")
    p.add_argument("--test-size",        type=float, default=0.2)
    p.add_argument("--random-state",     type=int,   default=42)
    p.add_argument("--model",            default=MODEL_PATH,
                   help="Path to save the best model as a pickle")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Load & split
# ---------------------------------------------------------------------------
def load_and_split(path, test_size, random_state):
    log.info(f"Loading cleaned data from: {path}")
    df = pd.read_csv(path)
    log.info(f"Dataset shape: {df.shape}")

    X = df.drop(columns=["is_default"])
    y = df["is_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
    log.info(f"Train default rate: {y_train.mean():.2%}  Test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------
# H2O AutoML wrapper — wraps the best H2O model in a sklearn-compatible class
# so predict_proba works the same way as the other two scripts.
# ---------------------------------------------------------------------------
class H2OModelWrapper:
    """Thin sklearn-compatible wrapper around the H2O leader model."""

    def __init__(self, h2o_model, feature_names):
        self.h2o_model    = h2o_model
        self.feature_names = feature_names

    def predict_proba(self, X):
        import h2o
        df_h2o = h2o.H2OFrame(pd.DataFrame(X, columns=self.feature_names))
        preds  = self.h2o_model.predict(df_h2o).as_data_frame()
        # H2O returns columns: predict, p0, p1
        proba_neg = preds["p0"].values
        proba_pos = preds["p1"].values
        return np.column_stack([proba_neg, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test, run_dir):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    log.info(f"AUC       : {auc:.4f}")
    log.info(f"F1        : {f1:.4f}")
    log.info(f"Precision : {prec:.4f}")
    log.info(f"Recall    : {rec:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Fully Paid','Default'])}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Fully Paid", "Default"],
        cmap="Blues", ax=ax
    )
    ax.set_title("H2O AutoML — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(run_dir, "h2o_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()

    # Feature importance (variable importance from H2O leader)
    try:
        import h2o
        varimp = model.h2o_model.varimp(use_pandas=True)
        if varimp is not None and not varimp.empty:
            top = varimp.head(15)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(top["variable"][::-1], top["scaled_importance"][::-1], color="steelblue")
            ax2.set_xlabel("Scaled Importance")
            ax2.set_title("H2O AutoML — Variable Importance (top 15)")
            plt.tight_layout()
            fi_path = os.path.join(run_dir, "h2o_feature_importance.png")
            plt.savefig(fi_path, dpi=150)
            plt.show()
        else:
            fi_path = cm_path  # stacked ensembles may not expose varimp
    except Exception as e:
        log.warning(f"Could not generate feature importance plot: {e}")
        fi_path = cm_path

    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec}, cm_path, fi_path

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved to: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args    = parse_args()
    run_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(run_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split(
        args.data, args.test_size, args.random_state
    )

    # Initialise H2O
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError:
        log.error(
            "h2o package not found. Install it with: pip install h2o\n"
            "Or run the conda environment that includes h2o."
        )
        raise

    log.info("Initialising H2O cluster...")
    h2o.init(nthreads=-1, max_mem_size="4G")
    h2o.no_progress()  # suppress verbose progress bars in Domino logs

    # Convert to H2OFrames
    feature_names = list(X_train.columns)
    target        = "is_default"

    train_df       = X_train.copy()
    train_df[target] = y_train.values
    test_df        = X_test.copy()
    test_df[target]  = y_test.values

    h2o_train = h2o.H2OFrame(train_df)
    h2o_test  = h2o.H2OFrame(test_df)

    # H2O needs the target as a factor for classification
    h2o_train[target] = h2o_train[target].asfactor()
    h2o_test[target]  = h2o_test[target].asfactor()

    mlflow.set_experiment("LendingClub-CreditRisk")

    with mlflow.start_run(run_name="h2o-automl"):
        mlflow.set_tag("framework",  "h2o")
        mlflow.set_tag("model_type", "H2OAutoML")

        params = {
            "max_models":       args.max_models,
            "max_runtime_secs": args.max_runtime_secs,
            "test_size":        args.test_size,
            "random_state":     args.random_state,
        }
        mlflow.log_params(params)

        # Run AutoML
        log.info(
            f"Running H2O AutoML (max_models={args.max_models}, "
            f"max_runtime_secs={args.max_runtime_secs})..."
        )
        aml = H2OAutoML(
            max_models=args.max_models,
            max_runtime_secs=args.max_runtime_secs,
            seed=args.random_state,
            sort_metric="AUC",
            balance_classes=True,      # handles class imbalance
            stopping_metric="AUC",
            stopping_rounds=5,
        )
        aml.train(
            x=feature_names,
            y=target,
            training_frame=h2o_train,
            leaderboard_frame=h2o_test,
        )

        leader = aml.leader
        log.info(f"AutoML leader: {leader.model_id}")
        log.info(f"\n{aml.leaderboard.head(10)}")

        # Wrap in sklearn-compatible class
        model = H2OModelWrapper(leader, feature_names)

        # Evaluate
        metrics, cm_path, fi_path = evaluate(model, X_test, y_test, run_dir)
        mlflow.log_metrics(metrics)

        # Log artifacts
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        # Log model via sklearn flavour (the wrapper is pickleable)
        mlflow.sklearn.log_model(model, artifact_path="h2o_automl_model")

        # Save locally for endpoint (predict.py loads xgboost_model.pkl;
        # h2o model is saved separately for reference)
        save_model(model, args.model)

        log.info(f"MLflow run complete - AUC: {metrics['auc']:.4f}  F1: {metrics['f1']:.4f}")

    # Shut down H2O cluster
    h2o.cluster().shutdown()


if __name__ == "__main__":
    main()
