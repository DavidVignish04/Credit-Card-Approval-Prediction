# Phase 3: Project Design Phase

This document defines the architectural specification of the Credit Card Approval Prediction System.

## 🏢 Architectural Specification
The system utilizes a 7-tier architecture to decouple data gathering, modeling, serving, and interface rendering:

### 1. User Layer
*   **Access Channel**: Desktop or Mobile Web Browsers.
*   **Actors**: Credit Analyst, Compliance Auditor, Prospective Customer.

### 2. Presentation Layer (Web UI)
*   **Framework**: HTML5, Vanilla CSS3 (Custom Glassmorphism), Vanilla JavaScript, Chart.js.
*   **Screens**:
    *   *Model Performance Dashboard*: Comparing Accuracy, F1, and AUC.
    *   *Analyst Console*: Form with sliding input syncs, prediction outcome panel, and session-persistent history table.
    *   *Compliance Audit Portal*: Drag-and-drop batch upload, flag KPI metrics, and flag violations ledger.
    *   *Customer Pre-Qualification*: Stepped check wizard with shield indicators and tips.

### 3. Application Layer (Flask Web API)
*   **Routing Controller**: Python Flask framework maps web pages and exposes API routing points.
*   **Pipeline Controller**: Loads the preprocessing artifacts, encodes categories, transforms numeric values, forwards inference vectors to the model, and formats JSON outputs.

### 4. Machine Learning Layer
*   **Modeling Framework**: Scikit-Learn (`sklearn`).
*   **Algorithms**: Logistic Regression, Decision Tree, Random Forest, XGBoost.
*   **Selector**: Automates model comparison and flags the model with the **Highest Accuracy** for serialization.

### 5. Data Layer
*   **Data Sources**: Synthetic generator compiling Age, Income, job tenure, loan counts, inquiry counts, and payment status history.
*   **Feature Engineering**: Delinquency codes (`1`-`5` denoting delinquency over 30 days) are engineered into binary `High_Risk` flags.

### 6. Model Storage
*   **Model Artifact**: `model.pkl` (serialized classifier).
*   **Preprocessing Artifact**: `preprocessor.pkl` (serialized StandardScaler and category mapping encoders).

### 7. Deployment Layer
*   **Cloud Hosting**: Integrated connection client script demonstrating uploading model packages to IBM Watson Machine Learning Service and creating online REST scoring API endpoints.
