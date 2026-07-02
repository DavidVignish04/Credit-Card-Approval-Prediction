# Phase 2: Requirement Analysis

## 1. Functional Requirements
*   **Predictive Model Pipeline**:
    *   Train 4 classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost).
    *   Select active model automatically based on **Highest Accuracy**.
    *   Separate model scoring payload (`model.pkl`) and preprocessing object (`preprocessor.pkl`).
*   **Web API Endpoints**:
    *   `GET /`: Serves the primary web dashboard.
    *   `GET /api/model_stats`: Serves performance comparison metrics for all 4 models.
    *   `POST /api/predict`: Scores single applicant profiles and details risk contributions.
    *   `POST /api/predict_batch`: Processes multi-record files, converts delinquency status to binary, and flags high-risk applications.
*   **UI Presentation**:
    *   Glassmorphic, ambient-glow dark theme.
    *   Interactive bar graph comparing metrics.
    *   Analyst scoring entry panel with sliding numeric synchronizers and session Evaluation History Log.
    *   Compliance batch-file upload simulator.
    *   Stepped consumer pre-qualification wizard.

## 2. Non-Functional Requirements
*   **Response Time**: Real-time single prediction responses within < 200ms.
*   **Aesthetics**: Glassmorphism, smooth CSS transitions, unified color tokens (Emerald, Crimson, Violet).
*   **Portability & Local Offline Mode**: Full frontend simulated fallback mode in JavaScript if the Flask server is offline.
*   **Scalability**: Script pipeline prepared for cloud-hosting on IBM Watson Machine Learning.
