"""
app.py
------
Loan Officer Dashboard — Domino App (Streamlit)

Provides a form-based UI for loan officers to:
  - Enter loan application details
  - Submit to the deployed Domino Model API endpoint
  - View default probability, risk tier, recommendation
  - See the top factors driving the decision (feature contributions / SHAP)

Environment variables required:
    DOMINO_MODEL_API_URL : Full URL of the deployed Domino endpoint
    DOMINO_MODEL_PAT     : API key for the endpoint

Usage:
    streamlit run scripts/app.py --server.port 8888 --server.address 0.0.0.0
"""

import os
import json

import requests
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Domino endpoint config
# ---------------------------------------------------------------------------
MODEL_API_URL = os.environ.get("DOMINO_MODEL_API_URL", "")
MODEL_API_KEY = os.environ.get("DOMINO_MODEL_PAT", "")

COLORS = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
    "text":   "#2c3e50",
    "muted":  "#7f8c8d",
}

st.set_page_config(page_title="Loan Officer Dashboard", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def build_gauge(prob: float) -> go.Figure:
    color = (COLORS["Low"] if prob < 0.15
             else COLORS["Medium"] if prob < 0.35
             else COLORS["High"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  15],  "color": "#d5f5e3"},
                {"range": [15, 35],  "color": "#fdebd0"},
                {"range": [35, 100], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def build_drivers_chart(contribs: dict) -> go.Figure:
    features = list(contribs.keys())
    values   = list(contribs.values())
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=[COLORS["High"] if v > 0 else COLORS["Low"] for v in values],
        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        height=280,
        title={"text": "Red = increases default risk  |  Green = decreases",
               "font": {"size": 11, "color": COLORS["muted"]}},
        xaxis_title="Contribution",
        margin=dict(t=40, b=20, l=10, r=60),
        yaxis={"autorange": "reversed"},
    )
    return fig


# ---------------------------------------------------------------------------
# Endpoint call
# ---------------------------------------------------------------------------
def score(payload: dict) -> dict:
    response = requests.post(
        MODEL_API_URL,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {MODEL_API_KEY}"},
        data=json.dumps({"data": payload}),
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    return body.get("result", body)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("🏦 Loan Officer Dashboard")
st.caption("LendingClub Credit Risk Scoring — Powered by Domino")

if not MODEL_API_URL:
    st.warning("DOMINO_MODEL_API_URL is not set. Add it in the App environment variables.")

left, right = st.columns([1, 3], gap="large")

with left:
    with st.form("application"):
        st.subheader("Loan Application")

        st.markdown("**Loan Details**")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0.0, value=15000.0, step=500.0)
        int_rate  = st.number_input("Interest Rate (%)", min_value=0.0, value=13.5, step=0.1)
        term      = st.selectbox("Term (months)", ["36", "60"])
        purpose   = st.selectbox("Purpose", [
            "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "medical", "moving", "vacation", "wedding", "other"])
        grade     = st.selectbox("Grade", list("ABCDEFG"), index=2)

        st.markdown("**Applicant Details**")
        annual_inc   = st.number_input("Annual Income ($)", min_value=0.0, value=65000.0, step=1000.0)
        dti          = st.number_input("DTI (%)", min_value=0.0, value=18.5, step=0.1)
        home         = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        verification = st.selectbox("Verification", ["Not Verified", "Source Verified", "Verified"], index=2)

        st.markdown("**Credit Profile**")
        installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=350.0)
        revol_bal   = st.number_input("Revolving Balance ($)", min_value=0.0, value=12000.0)
        revol_util  = st.number_input("Revolving Util (%)", min_value=0.0, value=55.0)
        open_acc    = st.number_input("Open Accounts", min_value=0, value=7, step=1)
        total_acc   = st.number_input("Total Accounts", min_value=0, value=18, step=1)
        pub_rec     = st.number_input("Public Records", min_value=0, value=0, step=1)
        delinq      = st.number_input("Delinquencies (2yr)", min_value=0, value=0, step=1)

        submitted = st.form_submit_button("Score Application", type="primary",
                                          width="stretch")

with right:
    if not submitted:
        st.info("Fill in the application and click **Score Application**.")
    else:
        payload = {
            "loan_amnt":           float(loan_amnt),
            "int_rate":            float(int_rate),
            "term":                term,
            "purpose":             purpose,
            "grade":               grade,
            "annual_inc":          float(annual_inc),
            "dti":                 float(dti),
            "home_ownership":      home,
            "verification_status": verification,
            "installment":         float(installment),
            "revol_bal":           float(revol_bal),
            "revol_util":          float(revol_util),
            "open_acc":            int(open_acc),
            "total_acc":           int(total_acc),
            "pub_rec":             int(pub_rec),
            "delinq_2yrs":         int(delinq),
        }
        try:
            with st.spinner("Scoring..."):
                result = score(payload)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the model endpoint. Check DOMINO_MODEL_API_URL.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The endpoint may be starting up — try again.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

        prob     = float(result.get("default_probability", 0))
        tier     = result.get("risk_tier", "—")
        contribs = result.get("contributions") or result.get("shap_values") or {}

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Default Probability", f"{prob:.1%}")
        m2.metric("Risk Score", str(result.get("risk_score", "—")), help="100 = safest, 0 = riskiest")
        m3.metric("Risk Tier", tier)
        m4.metric("Recommendation", result.get("recommendation", "—"))

        g, d = st.columns([2, 3])
        with g:
            st.markdown("**Default Probability**")
            st.plotly_chart(build_gauge(prob), width="stretch")
        with d:
            st.markdown("**Decision Drivers**")
            if contribs:
                st.plotly_chart(build_drivers_chart(contribs), width="stretch")
            else:
                st.caption("No feature contributions returned by the endpoint.")

        with st.expander("Raw API Response"):
            st.code(json.dumps(result, indent=2), language="json")
