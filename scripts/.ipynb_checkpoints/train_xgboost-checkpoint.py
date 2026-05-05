"""
train_xgboost.py
----------------
Trains an XGBoost classifier on the cleaned LendingClub dataset.
Logs parameters, metrics, and model artifact to MLflow.

Usage:
    python scripts/train_xgboost.py
    python scripts/train_xgboost.py --max-depth 6 --learning-rate 0.05
"""

import os
import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, ConfusionMatrixDisplay
)

import mlflow
import mlflow.xgboost

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME", "LendingClubProject")
DATA_PATH    = f"/domino/datasets/local/{PROJECT_NAME}/lending_clean.csv"
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "xgboost_model.pkl")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost classifier for loan default prediction")
    p.add_argument("--data",           default=DATA_PATH)
    p.add_argument("--n-estimators",   type=int,   default=200)
    p.add_argument("--max-depth",      type=int,   default=6)
    p.add_argument("--learning-rate",  type=float, default=0.1)
    p.add_argument("--subsample",      type=float, default=0.8)
    p.add_argument("--colsample",      type=float, default=0.8)
    p.add_argument("--test-size",      type=float, default=0.2)
    p.add_argument("--random-state",   type=int,   default=42)
    p.add_argument("--model",          default=MODEL_PATH)
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

    # XGBoost handles class imbalance via scale_pos_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
    log.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
    return X_train, X_test, y_train, y_test, scale_pos_weight


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(n_estimators, max_depth, lr, subsample, colsample,
                scale_pos_weight, random_state):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=colsample,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )


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
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Fully Paid", "Default"],
        cmap="Oranges", ax=ax
    )
    ax.set_title("XGBoost — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(run_dir, "xgboost_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()

    # Feature importance
    feat_imp = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("XGBoost — Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(run_dir, "xgboost_feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.show()

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
    args = parse_args()
    run_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(run_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, scale_pos_weight = load_and_split(
        args.data, args.test_size, args.random_state
    )

    mlflow.set_experiment("LendingClub-CreditRisk")

    with mlflow.start_run(run_name="xgboost-classifier"):
        mlflow.set_tag("framework", "xgboost")
        mlflow.set_tag("model_type", "XGBClassifier")

        params = {
            "n_estimators":      args.n_estimators,
            "max_depth":         args.max_depth,
            "learning_rate":     args.learning_rate,
            "subsample":         args.subsample,
            "colsample_bytree":  args.colsample,
            "scale_pos_weight":  round(scale_pos_weight, 2),
            "test_size":         args.test_size,
            "random_state":      args.random_state,
        }
        mlflow.log_params(params)

        # Train
        log.info("Training XGBoost classifier...")
        model = build_model(
            args.n_estimators, args.max_depth, args.learning_rate,
            args.subsample, args.colsample, scale_pos_weight, args.random_state
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Evaluate
        metrics, cm_path, fi_path = evaluate(model, X_test, y_test, run_dir)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        # Log model
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        # Save locally for endpoint
        save_model(model, args.model)

        log.info(f"MLflow run complete — AUC: {metrics['auc']:.4f}  F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
