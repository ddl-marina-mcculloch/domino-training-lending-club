"""
app.py
------
Loan Officer Dashboard — Domino App (Flask)

Provides a form-based UI for loan officers to:
  - Enter loan application details
  - Submit to the deployed Domino Model API endpoint
  - View default probability, risk tier, recommendation
  - See the top factors driving the decision (feature contributions / SHAP)

Environment variables required:
    DOMINO_MODEL_API_URL : Full URL of the deployed Domino endpoint
    DOMINO_MODEL_PAT     : API key for the endpoint

Usage:
    gunicorn app:app --bind 0.0.0.0:8890
"""

import os
import json
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Domino endpoint config
MODEL_API_URL = os.environ.get("DOMINO_MODEL_API_URL", "")

# Get access token from Domino's token service (for calling model APIs from apps)
def get_domino_token():
    try:
        return requests.get("http://localhost:8899/access-token").text
    except:
        # Fallback to env var if token service unavailable (e.g., local testing)
        return os.environ.get("DOMINO_MODEL_PAT", "")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Officer Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }
        h1 { color: #2c3e50; margin-bottom: 10px; }
        .subtitle { color: #7f8c8d; margin-bottom: 30px; }
        .layout { display: flex; gap: 30px; }
        .form-section { flex: 1; min-width: 300px; }
        .results-section { flex: 3; }
        .form-group { margin-bottom: 15px; }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 5px;
            color: #2c3e50;
        }
        input, select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .section-header {
            font-weight: 700;
            margin: 20px 0 10px 0;
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
        }
        button:hover { background: #2980b9; }
        button:disabled { background: #95a5a6; cursor: not-allowed; }
        .warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 4px;
            color: #856404;
            margin-bottom: 20px;
        }
        .info {
            background: #d1ecf1;
            border: 1px solid #17a2b8;
            padding: 20px;
            border-radius: 4px;
            color: #0c5460;
            text-align: center;
        }
        .results {
            display: none;
        }
        .results.show {
            display: block;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-label {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            margin-top: 8px;
            color: #2c3e50;
        }
        .metric.low .metric-value { color: #2ecc71; }
        .metric.medium .metric-value { color: #f39c12; }
        .metric.high .metric-value { color: #e74c3c; }
        .gauge {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .gauge-value {
            font-size: 48px;
            font-weight: 700;
            margin: 20px 0;
        }
        .drivers {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        .driver-item {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding: 8px;
            background: white;
            border-radius: 4px;
        }
        .driver-label {
            flex: 1;
            font-weight: 500;
        }
        .driver-bar {
            flex: 2;
            height: 24px;
            border-radius: 4px;
            position: relative;
            background: #ecf0f1;
        }
        .driver-bar-fill {
            height: 100%;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            font-size: 12px;
            font-weight: 600;
            color: white;
        }
        .driver-bar-fill.positive { background: #e74c3c; }
        .driver-bar-fill.negative { background: #2ecc71; }
        .raw-response {
            margin-top: 20px;
            padding: 15px;
            background: #2c3e50;
            color: #ecf0f1;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading { display: none; text-align: center; }
        .loading.show { display: block; }
        .error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 15px;
            border-radius: 4px;
            color: #721c24;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Loan Officer Dashboard</h1>
        <p class="subtitle">LendingClub Credit Risk Scoring — Powered by Domino</p>

        {% if not model_url %}
        <div class="warning">
            <strong>Warning:</strong> DOMINO_MODEL_API_URL is not set. Add it in the App environment variables.
        </div>
        {% endif %}


        <div class="layout">
            <div class="form-section">
                <div id="loanForm">
                    <div class="section-header">Loan Details</div>
                    <div class="form-group">
                        <label>Loan Amount ($)</label>
                        <input type="number" name="loan_amnt" value="15000" step="500" required>
                    </div>
                    <div class="form-group">
                        <label>Interest Rate (%)</label>
                        <input type="number" name="int_rate" value="13.5" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>Term (months)</label>
                        <select name="term" required>
                            <option value="36">36</option>
                            <option value="60">60</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Purpose</label>
                        <select name="purpose" required>
                            <option value="debt_consolidation">Debt Consolidation</option>
                            <option value="credit_card">Credit Card</option>
                            <option value="home_improvement">Home Improvement</option>
                            <option value="major_purchase">Major Purchase</option>
                            <option value="small_business">Small Business</option>
                            <option value="medical">Medical</option>
                            <option value="moving">Moving</option>
                            <option value="vacation">Vacation</option>
                            <option value="wedding">Wedding</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Grade</label>
                        <select name="grade" required>
                            <option value="A">A</option>
                            <option value="B">B</option>
                            <option value="C" selected>C</option>
                            <option value="D">D</option>
                            <option value="E">E</option>
                            <option value="F">F</option>
                            <option value="G">G</option>
                        </select>
                    </div>

                    <div class="section-header">Applicant Details</div>
                    <div class="form-group">
                        <label>Annual Income ($)</label>
                        <input type="number" name="annual_inc" value="65000" step="1000" required>
                    </div>
                    <div class="form-group">
                        <label>DTI (%)</label>
                        <input type="number" name="dti" value="18.5" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>Home Ownership</label>
                        <select name="home_ownership" required>
                            <option value="RENT">RENT</option>
                            <option value="MORTGAGE">MORTGAGE</option>
                            <option value="OWN">OWN</option>
                            <option value="OTHER">OTHER</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Verification Status</label>
                        <select name="verification_status" required>
                            <option value="Not Verified">Not Verified</option>
                            <option value="Source Verified">Source Verified</option>
                            <option value="Verified" selected>Verified</option>
                        </select>
                    </div>

                    <div class="section-header">Credit Profile</div>
                    <div class="form-group">
                        <label>Monthly Installment ($)</label>
                        <input type="number" name="installment" value="350" step="10" required>
                    </div>
                    <div class="form-group">
                        <label>Revolving Balance ($)</label>
                        <input type="number" name="revol_bal" value="12000" step="100" required>
                    </div>
                    <div class="form-group">
                        <label>Revolving Util (%)</label>
                        <input type="number" name="revol_util" value="55" step="1" required>
                    </div>
                    <div class="form-group">
                        <label>Open Accounts</label>
                        <input type="number" name="open_acc" value="7" step="1" required>
                    </div>
                    <div class="form-group">
                        <label>Total Accounts</label>
                        <input type="number" name="total_acc" value="18" step="1" required>
                    </div>
                    <div class="form-group">
                        <label>Public Records</label>
                        <input type="number" name="pub_rec" value="0" step="1" required>
                    </div>
                    <div class="form-group">
                        <label>Delinquencies (2yr)</label>
                        <input type="number" name="delinq_2yrs" value="0" step="1" required>
                    </div>

                    <button type="button" onclick="submitScore()">Score Application</button>
                </div>
            </div>

            <div class="results-section">
                <div class="info" id="initialInfo">
                    Fill in the application and click <strong>Score Application</strong>.
                </div>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Scoring application...</p>
                </div>

                <div class="results" id="results">
                    <div class="metrics" id="metrics"></div>
                    <div class="gauge" id="gauge"></div>
                    <div class="drivers" id="drivers"></div>
                    <details style="margin-top: 20px;">
                        <summary style="cursor: pointer; font-weight: 600; padding: 10px; background: #ecf0f1; border-radius: 4px;">
                            Raw API Response
                        </summary>
                        <div class="raw-response" id="rawResponse"></div>
                    </details>
                </div>

                <div class="error" id="error" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        async function submitScore() {
            document.getElementById('loading').classList.add('show');
            document.getElementById('initialInfo').style.display = 'none';
            document.getElementById('error').style.display = 'none';

            const inputs = document.querySelectorAll('#loanForm input, #loanForm select');
            const payload = {};

            inputs.forEach(input => {
                const val = input.value;
                if (['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'installment', 'revol_bal', 'revol_util'].includes(input.name)) {
                    payload[input.name] = parseFloat(val);
                } else if (['open_acc', 'total_acc', 'pub_rec', 'delinq_2yrs'].includes(input.name)) {
                    payload[input.name] = parseInt(val);
                } else {
                    payload[input.name] = val;
                }
            });

            try {
                const res = await fetch('score', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    displayResults(data);
                }
            } catch (e) {
                document.getElementById('error').textContent = 'Error: ' + e.message;
                document.getElementById('error').style.display = 'block';
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }

        function displayResults(result) {
            const prob = result.default_probability || 0;
            const tier = result.risk_tier || '—';
            const riskScore = result.risk_score || '—';
            const recommendation = result.recommendation || '—';
            const contribs = result.contributions || result.shap_values || {};

            // Determine risk class
            let riskClass = 'low';
            if (prob >= 0.35) riskClass = 'high';
            else if (prob >= 0.15) riskClass = 'medium';

            // Metrics
            const metricsHTML = `
                <div class="metric ${riskClass}">
                    <div class="metric-label">Default Probability</div>
                    <div class="metric-value">${(prob * 100).toFixed(1)}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Risk Score</div>
                    <div class="metric-value">${riskScore}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Risk Tier</div>
                    <div class="metric-value">${tier}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Recommendation</div>
                    <div class="metric-value" style="font-size: 20px;">${recommendation}</div>
                </div>
            `;
            document.getElementById('metrics').innerHTML = metricsHTML;

            // Gauge
            const gaugeColor = riskClass === 'low' ? '#2ecc71' : riskClass === 'medium' ? '#f39c12' : '#e74c3c';
            const gaugeHTML = `
                <h3 style="margin-bottom: 10px;">Default Probability</h3>
                <div class="gauge-value" style="color: ${gaugeColor};">${(prob * 100).toFixed(1)}%</div>
                <div style="height: 30px; background: linear-gradient(to right, #d5f5e3 0%, #d5f5e3 15%, #fdebd0 15%, #fdebd0 35%, #fadbd8 35%, #fadbd8 100%); border-radius: 15px; position: relative;">
                    <div style="position: absolute; left: ${prob * 100}%; top: 0; bottom: 0; width: 3px; background: black;"></div>
                </div>
            `;
            document.getElementById('gauge').innerHTML = gaugeHTML;

            // Drivers
            if (Object.keys(contribs).length > 0) {
                const maxAbs = Math.max(...Object.values(contribs).map(Math.abs));
                const driversHTML = `
                    <h3 style="margin-bottom: 15px;">Decision Drivers</h3>
                    <p style="font-size: 12px; color: #7f8c8d; margin-bottom: 15px;">Red = increases default risk | Green = decreases risk</p>
                    ${Object.entries(contribs).map(([feature, value]) => {
                        const isPositive = value > 0;
                        const width = (Math.abs(value) / maxAbs) * 100;
                        return `
                            <div class="driver-item">
                                <div class="driver-label">${feature}</div>
                                <div class="driver-bar">
                                    <div class="driver-bar-fill ${isPositive ? 'positive' : 'negative'}" style="width: ${width}%;">
                                        ${value > 0 ? '+' : ''}${value.toFixed(3)}
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                `;
                document.getElementById('drivers').innerHTML = driversHTML;
            } else {
                document.getElementById('drivers').innerHTML = '<p style="color: #7f8c8d;">No feature contributions returned by the endpoint.</p>';
            }

            // Raw response
            document.getElementById('rawResponse').textContent = JSON.stringify(result, null, 2);

            // Show results
            document.getElementById('results').classList.add('show');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, model_url=MODEL_API_URL)

@app.route('/score', methods=['POST'])
def score():
    return jsonify(score_application())

def score_application():
    if not MODEL_API_URL:
        return {"error": "DOMINO_MODEL_API_URL is not configured"}

    try:
        payload = request.json
        token = get_domino_token()

        response = requests.post(
            MODEL_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            },
            json={"data": payload},
            timeout=30
        )
        response.raise_for_status()
        body = response.json()
        return body.get("result", body)

    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the model endpoint"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890, debug=False)
