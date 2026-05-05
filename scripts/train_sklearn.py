"""
train_sklearn.py
----------------
Trains a Random Forest classifier on the cleaned LendingClub dataset.
Logs parameters, metrics, and model artifact to MLflow.

Usage:
    python scripts/train_sklearn.py
    python scripts/train_sklearn.py --n-estimators 200 --max-depth 10
"""

import os
import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
DATA_PATH    = f"/domino/datasets/local/{PROJECT_NAME}/lending_clean.csv"
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "sklearn_rf_model.pkl")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train sklearn Random Forest for loan default prediction")
    p.add_argument("--data",          default=DATA_PATH)
    p.add_argument("--n-estimators",  type=int,   default=150)
    p.add_argument("--max-depth",     type=int,   default=12)
    p.add_argument("--min-samples",   type=int,   default=20)
    p.add_argument("--test-size",     type=float, default=0.2)
    p.add_argument("--random-state",  type=int,   default=42)
    p.add_argument("--model",         default=MODEL_PATH)
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
# Build model
# ---------------------------------------------------------------------------
def build_model(n_estimators, max_depth, min_samples, random_state):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples,
        class_weight="balanced",   # handles class imbalance
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
        cmap="Blues", ax=ax
    )
    ax.set_title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(run_dir, "sklearn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()

    # Feature importance
    feat_imp = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest — Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(run_dir, "sklearn_feature_importance.png")
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

    X_train, X_test, y_train, y_test = load_and_split(
        args.data, args.test_size, args.random_state
    )

    mlflow.set_experiment("LendingClub-CreditRisk")

    with mlflow.start_run(run_name="sklearn-random-forest"):
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Log params
        params = {
            "n_estimators": args.n_estimators,
            "max_depth":     args.max_depth,
            "min_samples_leaf": args.min_samples,
            "test_size":     args.test_size,
            "random_state":  args.random_state,
            "class_weight":  "balanced",
        }
        mlflow.log_params(params)

        # Train
        log.info("Training Random Forest...")
        model = build_model(
            args.n_estimators, args.max_depth,
            args.min_samples, args.random_state
        )
        model.fit(X_train, y_train)

        # Evaluate
        metrics, cm_path, fi_path = evaluate(model, X_test, y_test, run_dir)
        mlflow.log_metrics(metrics)

        # Log artifacts
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="sklearn_rf_model")

        # Save locally for endpoint
        save_model(model, args.model)

        log.info(f"MLflow run complete — AUC: {metrics['auc']:.4f}  F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
