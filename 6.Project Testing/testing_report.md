# Phase 6: Project Testing

This document contains verification records and testing configurations for the Credit Card Approval Prediction System.

## 1. Automated Unit & Pipeline Verification
We validated the training pipeline logic by running `train_models.py` in the workspace virtual environment. The classification metrics generated are:

*   **Decision Tree**:
    *   Accuracy: **65.20%**
    *   Precision: **53.60%**
    *   Recall: **36.61%**
    *   F1-Score: **43.51%**
    *   ROC-AUC: **0.6786**
*   **XGBoost (Gradient Boosting)**:
    *   Accuracy: **65.00%**
    *   Precision: **52.60%**
    *   Recall: **44.26%**
    *   F1-Score: **48.07%**
    *   ROC-AUC: **0.6994**
*   **Random Forest**:
    *   Accuracy: **64.80%**
    *   Precision: **53.27%**
    *   Recall: **31.15%**
    *   F1-Score: **39.31%**
    *   ROC-AUC: **0.6970**
*   **Logistic Regression**:
    *   Accuracy: **64.60%**
    *   Precision: **52.34%**
    *   Recall: **36.61%**
    *   F1-Score: **43.09%**
    *   ROC-AUC: **0.6918**

The model with the highest test accuracy (**Decision Tree**) was successfully serialized to `model.pkl`, and its preprocessor parameters were serialized to `preprocessor.pkl`.

---

## 2. Interactive Scenario Verification (Browser Subagent Logs)

### Scenario 1: Analyst Credit Card Application Screening
*   **Method**: Form entry parameters were submitted via the Analyst Console.
*   **Verification**: The UI successfully loaded categories, calculated a pre-approval probability, changed the dial circle offset dynamically, and displayed positive / negative scoring explainers. The entry was logged inside the Evaluation History table.
*   **Result**: PASS.

### Scenario 2: High-Risk Delinquency Screening
*   **Method**: Loaded the 100-record applicant file (`sample_applicants.csv`) containing past-due delinquency history status codes (`1`–`5`). Clicked "Run Batch Compliance Scan".
*   **Verification**: The feature engineering controller successfully mapped codes `1` to `5` to binary `High_Risk` flags, generated auto-rejection tags, and updated batch stats (flag rate, approved count).
*   **Result**: PASS.

### Scenario 4: Customer Pre-Qualification Eligibility Check
*   **Method**: Went through the stepped consumer questionnaire wizard, submitting a soft check inquiry.
*   **Verification**: The loading spinner simulated wait times before displaying the shield status (Green Shield for high odds, Red warning indicator for low odds) and rendering credit improvements advice.
*   **Result**: PASS.
