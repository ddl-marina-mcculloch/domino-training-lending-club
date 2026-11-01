# Loan Default Prediction: Concepts in Plain English

A companion guide for the LendingClub training. No data science background needed.

---

## 1. The big picture

A bank has to decide: **"If we lend this person money, will they pay it back?"**

We give a computer thousands of past loans where we already know the outcome (paid back, or didn't). It looks for patterns, such as "people with high debt and low credit scores default more often". It then uses those patterns to estimate the risk on a **new** loan. That computer program is called a **model**.

The workflow in this training:

1. **Data**: collect past loans
2. **Preprocess**: clean them up and turn them into numbers the model can read
3. **Train**: let the model learn patterns (we train three kinds and compare)
4. **Evaluate**: check how good each one is
5. **Deploy**: put the best one behind an API so an app can ask it questions

---

## 2. The LendingClub dataset

**LendingClub** was a US peer-to-peer lending company. People borrowed money (for example to pay off credit cards), and investors funded the loans. LendingClub published data on its past loans. Each **row** is one loan and each **column** is one fact about it.

Our file `data/lending_raw.csv` has **50,000 loans** and 36 columns. About **80% were paid back** and **20% were not**.

### The outcome we want to predict

The `loan_status` column says what happened to the loan:

| Status | Meaning | Count |
|---|---|---|
| Fully Paid | Borrower repaid everything | ~40,000 |
| Charged Off | Lender gave up collecting and wrote it off as a loss | ~10,000 |
| Default | Borrower stopped paying | 1 |

During preprocessing we turn this into a simple yes/no column called **`is_default`**: 1 means the loan went bad, 0 means it was repaid. This is what the model learns to predict.

### The facts we use to predict it (the "features")

| Group | Columns | In plain words |
|---|---|---|
| **The loan** | `loan_amnt`, `int_rate`, `installment`, `term`, `purpose` | How much, what interest rate, monthly payment, 36 or 60 months, and why they're borrowing |
| **LendingClub's rating** | `grade` (A to G) | LendingClub's own risk letter grade. A is safest and G is riskiest |
| **The borrower's finances** | `annual_inc`, `dti`, `home_ownership`, `verification_status` | Yearly income, **debt-to-income** (monthly debt payments as a % of monthly income), rent/own/mortgage, and whether LendingClub verified their income |
| **Credit history** | `fico_range_low/high`, `earliest_cr_line`, `open_acc`, `total_acc`, `inq_last_6mths`, `mort_acc` | Credit score (FICO), when they opened their first credit account, number of accounts, recent credit applications, number of mortgages |
| **Credit card usage** | `revol_bal`, `revol_util`, `bc_util`, `num_bc_tl` | Card balances, how much of their card limit they use, and number of bank cards |
| **Red flags** | `pub_rec`, `delinq_2yrs` | Public records (such as bankruptcy) and late payments in the last 2 years |

### Cleaning and new columns (`preprocess.py`)

- **Drop** columns that are mostly empty, free-text, or would leak the answer.
- **Keep only finished loans**: ones that were clearly paid or clearly lost.
- **Create helpful ratios**:
  - `loan_to_income`: loan size relative to yearly income
  - `payment_to_income`: how much of their income the repayments eat up
  - `credit_utilization`: share of available credit being used
  - `has_derog`: 1 if they have any bankruptcy or late-payment history
  - `credit_breadth`: share of their accounts that are open
  - `credit_age_years`: how long they've had credit
- **One-hot encoding**: models only understand numbers, so a text column like `grade` = "C" becomes yes/no columns (`grade_B`, `grade_C`, and so on). The first value in each group (such as grade A) is left out because "none of the others" already means A.

> **Good to know:** the data has some messy extremes (for example one `dti` of 999 and an income of $6.5M). Real data is always like this, which is why we explore it before training.

---

## 3. How we judge a model, without the jargon

Because only ~20% of loans default, a lazy model that says **"nobody will ever default"** would be right 80% of the time and still be useless. So we use better measures than "percent correct".

### Confusion matrix: the scorecard

Every prediction lands in one of four boxes:

|  | Model says "Will default" | Model says "Will repay" |
|---|---|---|
| **Actually defaulted** | Caught it (true positive) | Missed it (false negative) |
| **Actually repaid** | False alarm (false positive) | Correct (true negative) |

- A **missed** default means the bank loses money.
- A **false alarm** means the bank turns away a good customer and loses the profit it would have made.

### Precision: "When the model raises the alarm, how often is it right?"

Of all the loans the model flagged as risky, what share really defaulted? High precision means few false alarms.

### Recall: "Of all the real defaults, how many did the model catch?"

Of all the loans that truly defaulted, what share did the model spot? High recall means few missed defaults.

Precision and recall pull against each other. If the model flags everyone, it catches every default (great recall) but wrongly flags tons of good borrowers (terrible precision).

### F1: one number that balances the two

F1 blends precision and recall into a single score from 0 to 1. A high F1 means the model is doing well on **both**.

### AUC: "How well can the model rank risky vs. safe?"

AUC is our main score. Imagine picking **one borrower who defaulted** and **one who repaid**, at random. AUC is the chance that the model gave the defaulter the **higher risk score**.

| AUC | What it means |
|---|---|
| **0.5** | Coin flip. The model has learned nothing |
| **0.6 to 0.7** | Weak but better than guessing |
| **0.7 to 0.8** | Decent. This is typical for credit risk |
| **0.8 to 0.9** | Strong |
| **1.0** | Perfect (be suspicious, it usually means a bug or leaked answers) |

Why we like AUC: it doesn't depend on picking a cutoff, and it isn't fooled by the 80/20 imbalance.

### Train/test split: the "exam" idea

We hide 20% of the loans from the model while it studies. After training, we test it on those unseen loans. This is like grading a student on questions they've never seen, so we know whether they **understood** or just **memorized**.

---

## 4. What `train_h2o.py` does, in plain words

**H2O AutoML** is like hiring a team of many model-builders at once. It tries lots of different kinds of models (GBM, random forest, deep learning, GLM, and combinations of them), scores each one by **AUC**, and ranks them on a **leaderboard**. The top one is the "leader" and gets saved to `models/h2o_best_model.pkl`.

Settings worth explaining:

- `max_models` / `max_runtime_secs`: caps on how many models to try and for how long.
- `balance_classes`: makes up for the 80/20 imbalance so the model doesn't ignore defaulters.
- `stopping_rounds`: stop a model early once it has stopped improving.
- **MLflow** records every run (settings, scores, charts) so you can compare the H2O, scikit-learn and XGBoost models side by side.

The **feature importance chart** in `results/` shows which facts mattered most to the model.

---

## 5. How the endpoint turns a model into an answer (`predict.py`)

An **endpoint** is a web address: send it a loan application and it sends back a risk assessment. `predict.py` is the code behind it.

### The sample request, line by line

```json
{
  "data": {
    "loan_amnt": 15000,
    "int_rate": 13.5,
    "grade": "C",
    "annual_inc": 65000,
    "dti": 18.5,
    "home_ownership": "RENT",
    "purpose": "debt_consolidation",
    "term": "36",
    "fico_range_low": 680,
    "fico_range_high": 684,
    "inq_last_6mths": 1,
    "mort_acc": 0,
    "bc_util": 45.0,
    "num_bc_tl": 3,
    "earliest_cr_line": "2010-01-01"
  }
}
```

This describes **one imaginary applicant**:

| Field | Value | Plain English |
|---|---|---|
| `loan_amnt` | 15000 | Wants to borrow **$15,000** |
| `int_rate` | 13.5 | Interest rate of **13.5%** a year |
| `grade` | "C" | LendingClub rated them **mid-risk** (a middle grade on the A to G scale) |
| `annual_inc` | 65000 | Earns **$65,000** a year |
| `dti` | 18.5 | **18.5%** of their monthly income already goes to debt payments |
| `home_ownership` | "RENT" | They **rent** their home |
| `purpose` | "debt_consolidation" | The loan is to **pay off other debts** (the most common reason) |
| `term` | "36" | Repay over **36 months** (3 years) |
| `fico_range_low` / `high` | 680 / 684 | Credit score is about **682**, which is "fair to good" |
| `inq_last_6mths` | 1 | Applied for credit **once** in the last 6 months |
| `mort_acc` | 0 | Has **no mortgage accounts** |
| `bc_util` | 45.0 | Using **45%** of their bank-card limits |
| `num_bc_tl` | 3 | Has **3** bank cards |
| `earliest_cr_line` | "2010-01-01" | First credit account opened **January 2010**, so about 16 years of credit history |

**What if a field is missing?** `predict.py` fills in a typical default (for example $60,000 income), so the call won't crash. Fewer details mean a less reliable answer, though.

**What does the code do with this?** It calculates the helper ratios (loan-to-income and so on), converts the text fields into yes/no columns the same way `preprocess.py` did, and hands all the numbers to the model. The model returns a probability, and the code adds a label and a recommendation.

### The response

Running the sample request through the current H2O model gives:

```json
{
  "default_probability": 0.213,
  "risk_score": 79,
  "risk_tier": "Medium",
  "recommendation": "Review",
  "contributions": {
    "fico_range_high": -1.6497,
    "fico_range_low": -1.623,
    "int_rate": 0.6067,
    "grade_C": 0.5194,
    "dti": 0.2357
  }
}
```

| Output | Plain English |
|---|---|
| `default_probability` | **21.3% chance** this loan defaults |
| `risk_score` | A friendlier number: 100 minus the % risk, so **higher is safer** (79 out of 100) |
| `risk_tier` | **Low** (under 15%), **Medium** (15 to 35%) or **High** (35% and over) |
| `recommendation` | Low gives **Approve**, Medium gives **Review** (a human takes a look), High gives **Decline** |
| `contributions` | **Why** the model said what it said: the 5 facts that pushed the score the most |

### Reading `contributions`

- A **negative** number pushes the risk **down** (good for the applicant). Here the FICO credit score helps a lot.
- A **positive** number pushes the risk **up**. Here the interest rate, the grade C and the debt-to-income ratio all raise it.
- Bigger numbers (positive or negative) mean a bigger influence.
- The units are "log-odds", which you can ignore. Just compare sizes and signs.

Lenders often need to explain *why* a loan was approved or declined, so explainability is a key part of responsible AI in finance.

### What is SHAP?

SHAP answers one question: **"Why did the model give this person this score?"**

Think of a group project where the final grade is 79. SHAP works out how much credit or blame each member deserves for that grade.

For a loan:

- The model starts from the **average risk** across all loans, say 20%.
- Each fact about the applicant then pushes the risk up or down from that average.
  - A high credit score pushes it **down**.
  - A high interest rate pushes it **up**.
  - A high debt-to-income ratio pushes it **up**.
- Add up all the pushes and you land on this person's final score.

SHAP gives each fact a number for its push. Positive means more risk, negative means less risk, and a bigger number means a bigger influence. The "Decision Drivers (SHAP)" chart in the loan officer dashboard is just those numbers drawn as bars.

It matters because a lender can't just say "the computer said no". They need to say which facts drove the decision.

---

## 6. Quick glossary

| Term | Meaning |
|---|---|
| **Model** | A program that learned patterns from past data |
| **Feature** | One input fact about a loan (a column) |
| **Target** | The thing we predict (`is_default`) |
| **Training / test set** | Data the model studies / data we hide to grade it fairly |
| **Overfitting** | Memorizing the training data instead of learning general patterns, so it fails on new loans |
| **AUC** | Chance the model ranks a defaulter as riskier than a non-defaulter (0.5 is a coin flip, 1.0 is perfect) |
| **Precision / Recall / F1** | Accuracy of alarms / share of defaults caught / balance of both |
| **AutoML** | Software that tries many models automatically and picks the best |
| **Endpoint** | A web address that returns a prediction on request |
| **MLflow** | A tool that logs and compares training runs |
| **SHAP** | A way to explain one prediction by showing how much each fact pushed the risk up or down |
| **One-hot encoding** | Turning a category (for example "grade C") into yes/no number columns |
