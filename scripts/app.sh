#!/usr/bin/env bash
# ==============================================================================
# app.sh — Domino App launcher for the Loan Officer Dashboard (Flask)
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

# Install dependencies
pip install flask gunicorn requests flask-cors --quiet

# Run Flask app with Gunicorn
# Port 8888 for Domino App deployment
gunicorn app:app \
  --bind 0.0.0.0:8888 \
  --workers 2 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
