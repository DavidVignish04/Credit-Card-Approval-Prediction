# Phase 1: Brainstorming & Ideation

## 1. Project Background
Banks receive thousands of credit card applications daily. Manual review is slow, expensive, and error-prone. This project automates the screening process using a predictive machine learning classifier evaluating financial and demographic profiles, mirroring real-world banking decisions.

## 2. Business Use Case Scenarios
*   **Scenario 1: Analyst Credit Card Application Screening**
    *   *Problem*: Credit analysts are overwhelmed with processing backlogs.
    *   *Solution*: An interactive interface that yields instant pre-approval likelihood scores, warning indicators, and explanation factors.
*   **Scenario 2: Delinquency Compliance Auditing**
    *   *Problem*: Customers with past-due loan records are high risk and must be filtered out for compliance.
    *   *Solution*: Batch-screening compliance audits that ingest applicant files, map delinquency status codes to binary flags, and auto-reject high-risk profiles.
*   **Scenario 4: Customer Pre-Qualification Eligibility Check**
    *   *Problem*: Users receive unnecessary hard-inquiry rejections.
    *   *Solution*: Stepped customer questionnaire wizard that simulates soft inquiries and outlines credit advice.

## 3. Algorithm Brainstorming
To evaluate the optimal predictive performance, we select 4 classification algorithms:
1.  **Logistic Regression**: Quick baseline linear probability classifier.
2.  **Decision Tree**: Non-linear, interpretable decision boundary logic.
3.  **Random Forest**: Ensemble bagging classifier reducing overfitting.
4.  **XGBoost (Gradient Boosting)**: Tree boosting classifier for optimized tabular scoring accuracy.
