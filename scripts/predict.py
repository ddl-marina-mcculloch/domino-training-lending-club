"""
predict.py
----------
Scoring function for the LendingClub loan default prediction endpoint.
Loaded by the Domino Model API — exposes a single `predict()` function.

Returns:
    - default_probability : float  (0–1)
    - risk_tier           : str    (Low / Medium / High)
    - risk_score          : int    (0–100, inverted probability for readability)
    - recommendation      : str    (Approve / Review / Decline)
    - shap_values         : dict   (top 5 feature contributions for explainability)

Sample request:
    {
      "data": {
        "loan_amnt": 15000,
        "int_rate": 13.5,
        "grade": "C",
        "annual_inc": 65000,
        "dti": 18.5,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "term": "36"
      }
    }
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd

# SHAP — optional, gracefully degrades if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME", "LendingClubProject")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl")

# ---------------------------------------------------------------------------
# Risk tier thresholds
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "low":    0.15,   # default_probability < 0.15  → Low
    "medium": 0.35,   # 0.15 <= probability < 0.35  → Medium
                      # probability >= 0.35          → High
}

RECOMMENDATIONS = {
    "Low":    "Approve",
    "Medium": "Review",
    "High":   "Decline",
}

# ---------------------------------------------------------------------------
# Feature schema
# Defines expected input features and their defaults.
# Matches the one-hot encoding applied in preprocess.py.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "installment",
    "revol_bal",
    "revol_util",
    "open_acc",
    "total_acc",
    "pub_rec",
    "delinq_2yrs",
]

FEATURE_DEFAULTS = {
    "loan_amnt":    10000,
    "int_rate":     12.0,
    "annual_inc":   60000,
    "dti":          15.0,
    "installment":  300.0,
    "revol_bal":    10000,
    "revol_util":   40.0,
    "open_acc":     8,
    "total_acc":    20,
    "pub_rec":      0,
    "delinq_2yrs":  0,
}

# One-hot encoded columns produced by preprocess.py
# These must exactly match the training feature set
OHE_COLUMNS = [
    "grade_B", "grade_C", "grade_D", "grade_E", "grade_F", "grade_G",
    "home_ownership_OTHER", "home_ownership_OWN", "home_ownership_RENT",
    "purpose_credit_card", "purpose_debt_consolidation", "purpose_educational",
    "purpose_home_improvement", "purpose_house", "purpose_major_purchase",
    "purpose_medical", "purpose_moving", "purpose_other",
    "purpose_renewable_energy", "purpose_small_business",
    "purpose_vacation", "purpose_wedding",
    "term_60",
    "verification_status_Source Verified", "verification_status_Verified",
]

# Engineered features (computed inside predict())
ENGINEERED_FEATURES = [
    "loan_to_income",
    "credit_utilization",
    "payment_to_income",
    "has_derog",
    "credit_breadth",
]

ALL_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES + OHE_COLUMNS


# ---------------------------------------------------------------------------
# Model loader — cached after first load
# ---------------------------------------------------------------------------
_model = None
_explainer = None

def _load_model():
    global _model, _explainer
    if _model is None:
        log.info(f"Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        log.info("Model loaded successfully")

        if SHAP_AVAILABLE:
            try:
                _explainer = shap.TreeExplainer(_model)
                log.info("SHAP TreeExplainer initialised")
            except Exception as e:
                log.warning(f"SHAP explainer could not be initialised: {e}")

    return _model, _explainer


# ---------------------------------------------------------------------------
# Feature engineering (mirrors preprocess.py)
# ---------------------------------------------------------------------------
def _engineer_features(row: dict) -> dict:
    annual_inc  = float(row.get("annual_inc", FEATURE_DEFAULTS["annual_inc"]))
    loan_amnt   = float(row.get("loan_amnt",  FEATURE_DEFAULTS["loan_amnt"]))
    installment = float(row.get("installment", FEATURE_DEFAULTS["installment"]))
    revol_util  = float(row.get("revol_util",  FEATURE_DEFAULTS["revol_util"]))
    pub_rec     = float(row.get("pub_rec",     FEATURE_DEFAULTS["pub_rec"]))
    delinq_2yrs = float(row.get("delinq_2yrs", FEATURE_DEFAULTS["delinq_2yrs"]))
    open_acc    = float(row.get("open_acc",    FEATURE_DEFAULTS["open_acc"]))
    total_acc   = float(row.get("total_acc",   FEATURE_DEFAULTS["total_acc"]))

    return {
        "loan_to_income":     loan_amnt / annual_inc if annual_inc > 0 else 0,
        "credit_utilization": min(revol_util, 100) / 100,
        "payment_to_income":  (installment * 12) / annual_inc if annual_inc > 0 else 0,
        "has_derog":          int(pub_rec > 0 or delinq_2yrs > 0),
        "credit_breadth":     open_acc / total_acc if total_acc > 0 else 0,
    }


# ---------------------------------------------------------------------------
# One-hot encoding (mirrors preprocess.py get_dummies output)
# ---------------------------------------------------------------------------
def _encode_categoricals(row: dict) -> dict:
    ohe = {col: 0 for col in OHE_COLUMNS}

    grade = str(row.get("grade", "A")).strip().upper()
    if f"grade_{grade}" in ohe:
        ohe[f"grade_{grade}"] = 1

    ownership = str(row.get("home_ownership", "RENT")).strip().upper()
    if f"home_ownership_{ownership}" in ohe:
        ohe[f"home_ownership_{ownership}"] = 1

    purpose = str(row.get("purpose", "other")).strip().lower().replace(" ", "_")
    if f"purpose_{purpose}" in ohe:
        ohe[f"purpose_{purpose}"] = 1

    term = str(row.get("term", "36")).strip().replace(" months", "")
    if term == "60":
        ohe["term_60"] = 1

    verification = str(row.get("verification_status", "Not Verified")).strip()
    if f"verification_status_{verification}" in ohe:
        ohe[f"verification_status_{verification}"] = 1

    return ohe


# ---------------------------------------------------------------------------
# Build feature vector
# ---------------------------------------------------------------------------
def _build_feature_vector(data: dict) -> pd.DataFrame:
    row = {}

    # Numeric features with defaults
    for feat in NUMERIC_FEATURES:
        row[feat] = float(data.get(feat, FEATURE_DEFAULTS.get(feat, 0)))

    # Engineered features
    row.update(_engineer_features(data))

    # One-hot encoded features
    row.update(_encode_categoricals(data))

    df = pd.DataFrame([row])[ALL_FEATURES]
    return df


# ---------------------------------------------------------------------------
# Risk tier & recommendation
# ---------------------------------------------------------------------------
def _get_risk_tier(prob: float) -> str:
    if prob < THRESHOLDS["low"]:
        return "Low"
    elif prob < THRESHOLDS["medium"]:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------
def _get_shap_explanation(explainer, feature_vector: pd.DataFrame, top_n: int = 5) -> dict:
    if explainer is None or not SHAP_AVAILABLE:
        return {}

    try:
        sv = explainer.shap_values(feature_vector)
        # For binary classifiers shap_values returns list [class0, class1]
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]  # single row

        shap_df = pd.DataFrame({
            "feature":    feature_vector.columns,
            "shap_value": sv,
            "abs_shap":   np.abs(sv),
        }).sort_values("abs_shap", ascending=False).head(top_n)

        return {
            row["feature"]: round(float(row["shap_value"]), 4)
            for _, row in shap_df.iterrows()
        }
    except Exception as e:
        log.warning(f"SHAP calculation failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main predict function — called by Domino Model API
# ---------------------------------------------------------------------------
def predict(data: dict) -> dict:
    """
    Score a single loan application for default risk.

    Args:
        data: dict of loan features (see sample request in module docstring)

    Returns:
        dict with default_probability, risk_score, risk_tier,
        recommendation, and shap_values
    """
    model, explainer = _load_model()

    # Build feature vector
    feature_vector = _build_feature_vector(data)

    # Score
    prob = float(model.predict_proba(feature_vector)[0][1])
    risk_tier      = _get_risk_tier(prob)
    risk_score     = int(round((1 - prob) * 100))   # 100 = safest, 0 = riskiest
    recommendation = RECOMMENDATIONS[risk_tier]

    # SHAP explanation
    shap_values = _get_shap_explanation(explainer, feature_vector)

    result = {
        "default_probability": round(prob, 4),
        "risk_score":          risk_score,
        "risk_tier":           risk_tier,
        "recommendation":      recommendation,
        "shap_values":         shap_values,
    }

    log.info(f"Scored loan — prob: {prob:.4f}, tier: {risk_tier}, rec: {recommendation}")
    return result


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "loan_amnt":          15000,
        "int_rate":           13.5,
        "grade":              "C",
        "annual_inc":         65000,
        "dti":                18.5,
        "home_ownership":     "RENT",
        "purpose":            "debt_consolidation",
        "term":               "36",
        "installment":        350.0,
        "revol_bal":          12000,
        "revol_util":         55.0,
        "open_acc":           7,
        "total_acc":          18,
        "pub_rec":            0,
        "delinq_2yrs":        0,
        "verification_status": "Verified",
    }

    result = predict(sample)
    print("\n=== Prediction Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
