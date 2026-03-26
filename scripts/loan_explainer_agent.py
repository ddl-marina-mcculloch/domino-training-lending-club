"""
loan_explainer_agent.py
-----------------------
Agentic LLM system that explains credit decisions to loan officers.

The agent has access to two tools:
    1. score_loan       — calls the Domino scoring endpoint (predict.py)
    2. retrieve_policy  — retrieves relevant lending policy documents (RAG)

Given a loan application, the agent:
    1. Scores the loan to get default probability, risk tier, and SHAP values
    2. Retrieves policy documents relevant to the top risk drivers
    3. Generates a plain-English explanation of the decision with policy citations

Can be run standalone or imported and called from the Dash app (Lab 6.4).

Environment variables:
    DOMINO_MODEL_API_URL  : Deployed scoring endpoint URL
    DOMINO_MODEL_API_KEY  : Endpoint API key
    ANTHROPIC_API_KEY     : API key for the LLM (Claude)

Usage:
    python genai/loan_explainer_agent.py
    python genai/loan_explainer_agent.py --interactive
"""

import os
import json
import logging
import argparse
from typing import Any

import requests
import anthropic
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_API_URL  = os.environ.get("DOMINO_MODEL_API_URL", "")
MODEL_API_KEY  = os.environ.get("DOMINO_MODEL_API_KEY", "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
POLICY_DIR     = Path(__file__).parent / "policy_docs"
LLM_MODEL      = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Policy document store
# Simple file-based RAG — each .txt file in policy_docs/ is one policy section
# ---------------------------------------------------------------------------
POLICY_KEYWORDS = {
    "dti":                  ["debt_to_income_policy.txt"],
    "int_rate":             ["interest_rate_policy.txt"],
    "credit_utilization":   ["credit_utilization_policy.txt"],
    "loan_to_income":       ["loan_to_income_policy.txt"],
    "payment_to_income":    ["debt_to_income_policy.txt"],
    "has_derog":            ["derogatory_marks_policy.txt"],
    "annual_inc":           ["income_verification_policy.txt"],
    "revol_util":           ["credit_utilization_policy.txt"],
    "grade":                ["credit_grade_policy.txt"],
    "home_ownership":       ["collateral_policy.txt"],
}

DEFAULT_POLICY_FILE = "general_lending_policy.txt"


# ---------------------------------------------------------------------------
# Tool 1: Score loan via Domino endpoint
# ---------------------------------------------------------------------------
def score_loan(loan_data: dict) -> dict:
    """
    Calls the Domino scoring endpoint and returns:
        default_probability, risk_score, risk_tier,
        recommendation, shap_values
    """
    log.info("Tool: score_loan called")

    if not MODEL_API_URL:
        # Return mock result if endpoint not configured (for local dev/testing)
        log.warning("DOMINO_MODEL_API_URL not set — returning mock score")
        return {
            "default_probability": 0.31,
            "risk_score":          69,
            "risk_tier":           "Medium",
            "recommendation":      "Review",
            "shap_values": {
                "dti":                  0.142,
                "int_rate":             0.098,
                "credit_utilization":  -0.071,
                "loan_to_income":       0.065,
                "annual_inc":          -0.043,
            },
        }

    try:
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {MODEL_API_KEY}",
        }
        response = requests.post(
            MODEL_API_URL,
            headers=headers,
            data=json.dumps({"data": loan_data}),
            timeout=15,
        )
        response.raise_for_status()
        result = response.json().get("result", response.json())
        log.info(f"Score: prob={result.get('default_probability')}, "
                 f"tier={result.get('risk_tier')}")
        return result

    except Exception as e:
        log.error(f"score_loan failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 2: Retrieve policy documents (RAG)
# ---------------------------------------------------------------------------
def retrieve_policy(risk_factors: list[str]) -> str:
    """
    Given a list of top risk factor feature names, retrieves the most
    relevant policy document sections from the policy_docs/ directory.

    Returns a concatenated string of relevant policy text.
    """
    log.info(f"Tool: retrieve_policy called for factors: {risk_factors}")

    retrieved_files = set()
    for factor in risk_factors:
        # Match factor name to policy files
        for keyword, files in POLICY_KEYWORDS.items():
            if keyword in factor.lower():
                retrieved_files.update(files)

    if not retrieved_files:
        retrieved_files.add(DEFAULT_POLICY_FILE)

    policy_text = []
    for fname in sorted(retrieved_files):
        fpath = POLICY_DIR / fname
        if fpath.exists():
            content = fpath.read_text(encoding="utf-8").strip()
            policy_text.append(f"--- {fname} ---\n{content}")
            log.info(f"Retrieved: {fname}")
        else:
            log.warning(f"Policy file not found: {fpath}")
            policy_text.append(
                f"--- {fname} ---\n"
                f"[Policy document not available. Apply standard lending guidelines.]"
            )

    return "\n\n".join(policy_text) if policy_text else "No relevant policy documents found."


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "score_loan",
        "description": (
            "Score a loan application using the deployed credit risk model. "
            "Returns default_probability (0–1), risk_score (0–100), risk_tier "
            "(Low/Medium/High), recommendation (Approve/Review/Decline), and "
            "shap_values showing the top features driving the prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "loan_data": {
                    "type": "object",
                    "description": "Loan application features as key-value pairs",
                    "properties": {
                        "loan_amnt":           {"type": "number"},
                        "int_rate":            {"type": "number"},
                        "grade":               {"type": "string"},
                        "annual_inc":          {"type": "number"},
                        "dti":                 {"type": "number"},
                        "home_ownership":      {"type": "string"},
                        "purpose":             {"type": "string"},
                        "term":                {"type": "string"},
                        "installment":         {"type": "number"},
                        "revol_bal":           {"type": "number"},
                        "revol_util":          {"type": "number"},
                        "open_acc":            {"type": "integer"},
                        "total_acc":           {"type": "integer"},
                        "pub_rec":             {"type": "integer"},
                        "delinq_2yrs":         {"type": "integer"},
                        "verification_status": {"type": "string"},
                    },
                    "required": ["loan_amnt", "int_rate", "annual_inc", "dti"],
                }
            },
            "required": ["loan_data"],
        },
    },
    {
        "name": "retrieve_policy",
        "description": (
            "Retrieve relevant lending policy documents based on the top risk "
            "factors identified by SHAP values. Returns policy text that should "
            "be cited in the decision explanation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "risk_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of feature names that are the top drivers of the "
                        "credit decision, e.g. ['dti', 'int_rate', 'credit_utilization']"
                    ),
                }
            },
            "required": ["risk_factors"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def dispatch_tool(tool_name: str, tool_input: dict) -> Any:
    if tool_name == "score_loan":
        return score_loan(tool_input["loan_data"])
    elif tool_name == "retrieve_policy":
        return retrieve_policy(tool_input["risk_factors"])
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI credit analyst assistant for a lending company.
Your role is to help loan officers understand credit decisions made by the 
machine learning model.

When given a loan application, you must:
1. Call score_loan to get the model's prediction and SHAP values
2. Identify the top risk factors from the SHAP values (positive SHAP = increases risk)
3. Call retrieve_policy to get relevant policy documents for those risk factors
4. Write a clear, professional explanation of the decision that:
   - States the recommendation (Approve / Review / Decline) upfront
   - Explains the default probability and what it means in plain English
   - Describes the top 3–5 factors driving the decision, citing policy where relevant
   - Suggests any conditions or next steps for the loan officer
   - Uses accessible language — avoid jargon where possible

Format your response as:
### Decision Summary
[One sentence recommendation with probability]

### Key Risk Factors
[Bullet points for each top driver]

### Policy References
[Relevant policy citations]

### Recommended Next Steps
[Actionable guidance for the loan officer]

Be factual, balanced, and never make promises about approval outcomes."""


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------
def explain_loan_decision(loan_application: dict, max_turns: int = 6) -> str:
    """
    Runs the agentic loop:
        1. Send loan application to Claude with tools available
        2. Claude calls score_loan → gets prediction + SHAP
        3. Claude calls retrieve_policy → gets relevant policy docs
        4. Claude generates plain-English explanation
        5. Return final explanation text

    Args:
        loan_application : dict of loan features
        max_turns        : max agentic turns before stopping

    Returns:
        str — plain-English decision explanation
    """
    if not ANTHROPIC_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    messages = [
        {
            "role": "user",
            "content": (
                f"Please explain the credit decision for this loan application:\n\n"
                f"```json\n{json.dumps(loan_application, indent=2)}\n```"
            ),
        }
    ]

    log.info("Starting agentic loop...")
    turns = 0

    while turns < max_turns:
        turns += 1
        log.info(f"Turn {turns}")

        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        log.info(f"Stop reason: {response.stop_reason}")

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        # If Claude is done — return the final text
        if response.stop_reason == "end_turn":
            final_text = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            log.info("Agentic loop complete")
            return final_text.strip()

        # Process tool calls
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                log.info(f"Tool call: {block.name}  input: {block.input}")
                result = dispatch_tool(block.name, block.input)
                log.info(f"Tool result: {str(result)[:200]}")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

    log.warning(f"Reached max_turns ({max_turns}) without end_turn")
    return "Could not generate explanation — maximum turns reached."


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
def interactive_mode():
    print("\n🏦 Loan Decision Explainer — Interactive Mode")
    print("=" * 50)
    print("Enter loan details (press Enter to use defaults)\n")

    def prompt_float(label, default):
        val = input(f"  {label} [{default}]: ").strip()
        return float(val) if val else default

    def prompt_str(label, default):
        val = input(f"  {label} [{default}]: ").strip()
        return val if val else default

    def prompt_int(label, default):
        val = input(f"  {label} [{default}]: ").strip()
        return int(val) if val else default

    loan = {
        "loan_amnt":           prompt_float("Loan Amount ($)",       15000),
        "int_rate":            prompt_float("Interest Rate (%)",     13.5),
        "grade":               prompt_str("Grade (A-G)",             "C"),
        "annual_inc":          prompt_float("Annual Income ($)",     65000),
        "dti":                 prompt_float("DTI (%)",               18.5),
        "home_ownership":      prompt_str("Home Ownership",         "RENT"),
        "purpose":             prompt_str("Purpose",                "debt_consolidation"),
        "term":                prompt_str("Term (36/60)",            "36"),
        "installment":         prompt_float("Monthly Installment ($)", 350),
        "revol_bal":           prompt_float("Revolving Balance ($)", 12000),
        "revol_util":          prompt_float("Revolving Util (%)",    55),
        "open_acc":            prompt_int("Open Accounts",           7),
        "total_acc":           prompt_int("Total Accounts",         18),
        "pub_rec":             prompt_int("Public Records",          0),
        "delinq_2yrs":         prompt_int("Delinquencies (2yr)",     0),
        "verification_status": prompt_str("Verification Status",    "Verified"),
    }

    print("\n⏳ Generating explanation...\n")
    explanation = explain_loan_decision(loan)
    print(explanation)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan decision explainer agent")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive prompt mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        # Default sample application
        sample = {
            "loan_amnt":           15000,
            "int_rate":            13.5,
            "grade":               "C",
            "annual_inc":          65000,
            "dti":                 18.5,
            "home_ownership":      "RENT",
            "purpose":             "debt_consolidation",
            "term":                "36",
            "installment":         350.0,
            "revol_bal":           12000,
            "revol_util":          55.0,
            "open_acc":            7,
            "total_acc":           18,
            "pub_rec":             0,
            "delinq_2yrs":         0,
            "verification_status": "Verified",
        }

        print("\n⏳ Generating loan decision explanation...\n")
        explanation = explain_loan_decision(sample)
        print(explanation)
