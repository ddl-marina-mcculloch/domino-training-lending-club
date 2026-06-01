# LendingClub Credit Risk — Training Delivery Guide

> **Audience:** Domino employees delivering the LendingClub Credit Risk training.
> This guide explains what the project does, how it is structured, what each model does, what the parameters mean, and how to interpret and discuss experiment results with trainees.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What the Experiment Does](#what-the-experiment-does)
3. [The Three Models](#the-three-models)
4. [The Class Imbalance Problem](#the-class-imbalance-problem)
5. [Understanding the 4 Evaluation Metrics](#understanding-the-4-evaluation-metrics)
6. [Experiment Results](#experiment-results)
7. [Key Talking Points for Trainers](#key-talking-points-for-trainers)
8. [Domino Platform Features Demonstrated](#domino-platform-features-demonstrated)

---

## Project Overview

The **domino-training-lending-club** project demonstrates how Domino can be used to train, track, and compare multiple machine learning models for a credit risk use case.

| Attribute | Detail |
|---|---|
| **Dataset** | LendingClub (peer-to-peer lending platform) |
| **Goal** | Predict whether a borrower is likely to default on a loan |
| **Problem type** | Binary classification (high risk / low risk) |
| **Project stage** | Ideation — model exploration and experimentation phase |

This is a highly realistic financial ML use case, making it an excellent vehicle for demonstrating Domino's experiment tracking, model comparison, and reproducibility features.

---

## What the Experiment Does

Three separate training scripts were run as **Domino Jobs** within a single experiment called **LendingClub-CreditRisk**. Each script trains a different type of model on the same dataset and logs its parameters, metrics, and output artefacts (confusion matrix and feature importance charts) to the experiment.

Domino's experiment tracker captures all of this automatically, enabling side-by-side comparison across frameworks.

---

## The Three Models

### 1. H2O AutoML — `train_h2o.py`

H2O AutoML automatically searches for the best model architecture and hyperparameters within a time budget. It tries multiple algorithms internally and selects the best performer. It is the most **hands-off** approach — you give it a time limit and it does the searching for you.

| Parameter | Value | Meaning |
|---|---|---|
| `max_runtime_secs` | `120` | Time budget for the full search |
| `max_models` | `15` | Maximum number of candidate models to try |

**Runtime:** 2 minutes 11 seconds

---

### 2. XGBoost Classifier — `train_xgboost.py`

XGBoost (Extreme Gradient Boosting) is a powerful ensemble method that builds trees **sequentially**, each one correcting the errors of the previous. It is widely used in industry for tabular data and tends to perform well out of the box.

| Parameter | Value | Meaning |
|---|---|---|
| `learning_rate` | `0.1` | How much each tree contributes to the final prediction |
| `max_depth` | `12` | How deep each tree can grow |
| `colsample_bytree` | `0.8` | Fraction of features sampled per tree |
| `subsample` | `0.8` | Fraction of training data sampled per tree |
| `scale_pos_weight` | `3.94` | Upweights the minority (default) class to handle imbalance |
| `n_estimators` | `200` | Number of trees to build |

**Runtime:** 39 seconds

---

### 3. Sklearn Random Forest — `train_sklearn.py`

A Random Forest builds many independent decision trees **in parallel** and averages their predictions. It is robust, interpretable, and a strong baseline model.

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | `150` | Number of trees to build |
| `max_depth` | `6` | Shallower trees to prevent overfitting |
| `min_samples_leaf` | `20` | Minimum data points required at each leaf node |
| `class_weight` | `balanced` | Automatically adjusts weights to compensate for class imbalance |
| `random_state` | `42` | Ensures reproducible results |

**Runtime:** 14 seconds

---

## The Class Imbalance Problem

> **This is a critical concept to explain to trainees.**

In a real-world lending dataset, the vast majority of loans are repaid — defaults are relatively rare. A model that simply predicts "not default" for every loan can achieve very high accuracy while being **completely useless** as a risk filter.

Each model handles class imbalance differently:

| Model | Imbalance Strategy |
|---|---|
| **H2O AutoML** | Handled internally as part of its search process; no explicit weighting applied |
| **XGBoost** | `scale_pos_weight: 3.94` — treats each default example as ~4× more important than a non-default |
| **Sklearn RF** | `class_weight: balanced` — automatically inversely weights classes proportional to their frequency |

This is precisely why **the choice of evaluation metric matters so much** — accuracy alone would be deeply misleading here.

---

## Understanding the 4 Evaluation Metrics

These are the four metrics logged by each run and visible in the Domino experiment comparison view.

### AUC (Area Under the ROC Curve)

AUC measures the model's overall ability to **rank** high-risk borrowers above low-risk ones, regardless of the decision threshold applied.

- **Range:** 0.5 (random guessing) → 1.0 (perfect)
- **Interpretation:** An AUC of 0.71 means that if you randomly pick one defaulter and one non-defaulter, the model will correctly rank the defaulter as riskier **71% of the time**
- **Limitation:** Does not describe real-world behaviour at a specific decision threshold

### F1 Score

F1 is the **harmonic mean of Precision and Recall**, combining both into a single number.

- Particularly useful for imbalanced classes — a model that rarely predicts the minority class will have low recall and therefore a low F1, even if accuracy looks acceptable
- A high F1 indicates the model is doing well at **both** identifying actual defaults and avoiding false alarms
- **Trainers should emphasise:** F1 is often a better headline metric than accuracy for credit risk

### Precision

> *"Of all the loans the model flagged as high-risk, how many actually were?"*

- A **high precision** model is selective — when it raises an alarm, it is usually right
- **Low precision** means many creditworthy borrowers are incorrectly rejected → lost revenue and reputational risk

### Recall

> *"Of all the loans that actually defaulted, how many did the model catch?"*

- A **high recall** model misses fewer actual defaults
- **Low recall** means dangerous loans slip through undetected
- In lending, a **missed default is typically more costly than a missed opportunity** — this is why recall is often prioritised in credit risk

### The Precision–Recall Trade-off

These two metrics are in tension:

- A model that predicts "high risk" very **aggressively** → high recall, lower precision (flags good borrowers incorrectly)
- A model that predicts "high risk" very **conservatively** → high precision, lower recall (misses real defaults)

The right balance depends on **business priorities**.

---

## Experiment Results

### Results Summary

| Metric | H2O AutoML | XGBoost | Sklearn RF |
|---|---|---|---|
| **AUC** | 0.7129 | 0.6975 | 0.7090 |
| **F1** | 0.1129 | 0.4219 | **0.4305** |
| **Precision** | **0.4692** | 0.3384 | 0.3453 |
| **Recall** | 0.0641 | 0.5599 | **0.5715** |

### Interpreting Each Model's Results

#### H2O AutoML
- Achieved the **highest AUC** and **highest precision**
- But with recall of only **0.064**, it is missing ~94% of actual defaults
- This reflects very **conservative** prediction behaviour — only flags loans as high-risk when extremely confident
- F1 of **0.11** reflects this severe imbalance between precision and recall
- Despite its strong AUC, it would **not be fit for purpose** as a practical risk filter

#### XGBoost
- Sits in the middle on AUC (0.698)
- Delivers **much more balanced** overall performance
- The `scale_pos_weight` parameter is clearly working — catches **56% of defaults** (recall) while maintaining reasonable precision
- F1 of **0.42** reflects a real, usable trade-off

#### Sklearn Random Forest ✅ *Recommended candidate*
- **Best overall balance:** second-highest AUC, best F1 (0.43), best recall (0.57)
- `class_weight: balanced` is doing similar work to XGBoost's `scale_pos_weight`
- **Fastest to train** (14 seconds) and **easiest to explain** to a business audience
- Strongest candidate from this comparison for most credit risk applications

---

## Key Talking Points for Trainers

### Why are we comparing three different model types?
To demonstrate that Domino can track **any ML framework** — H2O, XGBoost, Sklearn — in a single experiment, making cross-framework comparison straightforward without any special setup.

### Why does H2O AutoML have such low recall despite a good AUC?
AUC and recall measure **different things**:
- AUC measures ranking ability **across all thresholds**
- Recall is measured at a **specific decision threshold** (usually 0.5)

H2O may rank borrowers well in relative terms but applies a threshold that results in very few positive predictions. This is an excellent opportunity to discuss **threshold tuning** with trainees.

### Why does training time vary so much?
| Model | Why it takes that long |
|---|---|
| H2O AutoML (2m 11s) | Runs an internal model search across many algorithms and hyperparameter combinations |
| XGBoost (39s) | Builds 200 trees **sequentially**, each dependent on the previous |
| Sklearn RF (14s) | Builds 150 trees **in parallel**, each independent |

Domino logs the duration of every run automatically, making it easy to factor **training cost** into model selection decisions.

### What would the next step be?
After identifying the Sklearn Random Forest as the leading candidate, a team would typically:

1. **Tune hyperparameters** further (e.g., grid search over `max_depth`, `n_estimators`)
2. **Validate** on a hold-out test set
3. **Register the model** in Domino's Model Registry
4. **Move to deployment** via the Domino deployment workflow

The Domino Experiments view supports this workflow natively — runs can be compared, promoted, and linked to governance records.

---

## Domino Platform Features Demonstrated

Trainers should **explicitly call out** each of the following during the session:

| Feature | What to show |
|---|---|
| **Experiment Tracking** | Automatic logging of parameters, metrics, and artefacts from each run |
| **Multi-framework support** | H2O, XGBoost, and Sklearn all tracked in the same experiment without special setup |
| **Parallel Coordinates Plot** | Interactive visualisation showing how parameters relate to metrics across runs |
| **Run Comparison** | Side-by-side table view of all three runs |
| **Reproducibility** | Each run records its source script, environment, hardware tier, duration, and Git commit |
| **Project lifecycle stages** | The Ideation stage maps directly to real ML development workflows |

---

*Document maintained by Domino Training Team. For questions or updates, contact the training programme lead.*
