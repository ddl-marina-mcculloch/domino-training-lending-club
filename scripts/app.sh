pkill -f 'python.*app.py' 2>/dev/null; sleep 1; echo DOMINO_MODEL_API_URL=$DOMINO_MODEL_API_URL
pkill -f 'python.*app' 2>/dev/null; sleep 1; cd /mnt/code/scripts && python3 app.py 2>&1 | head -40 &
#!/usr/bin/env bash
# =============================================================================
# app.sh — Domino App launcher for the Loan Officer Dashboard (Dash)
#
# Required environment variables (set in Domino App settings):
#   DOMINO_MODEL_API_URL  : Full URL of the deployed Domino scoring endpoint
#   DOMINO_MODEL_API_KEY  : API key for the endpoint
# =============================================================================

set -e

echo "Starting Loan Officer Dashboard..."
echo "Model API URL: ${DOMINO_MODEL_API_URL:-'NOT SET — update in App environment variables'}"

# Install any missing dependencies at startup
pip install dash dash-bootstrap-components plotly requests --quiet

# Launch Dash app
python app.py
