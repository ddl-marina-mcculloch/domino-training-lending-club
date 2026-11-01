#!/usr/bin/env bash
# ==============================================================================
# app.sh — Domino App launcher for the Loan Officer Dashboard (Streamlit)
#
# Required environment variables (set in Domino App settings):
#   DOMINO_MODEL_API_URL  : Full URL of the deployed Domino scoring endpoint
#   DOMINO_MODEL_PAT      : API key for the endpoint (Find in your Account settings)
# ==============================================================================

set -e

# Run from this script's folder so it works from /mnt/code or scripts/
cd "$(dirname "$0")"

echo "Starting Loan Officer Dashboard..."
echo "Model API URL: ${DOMINO_MODEL_API_URL:-'NOT SET — update in App environment variables'}"
pip install streamlit plotly requests --quiet

# Deployed Domino Apps must listen on 0.0.0.0:8888. 8888 is already taken inside a
# workspace, so 8889 is for testing there. Set --server.port back to 8888 before
# publishing the App.
streamlit run app.py \
  --server.port 8889 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
