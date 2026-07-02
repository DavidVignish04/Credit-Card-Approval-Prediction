# Project Documentation & Walkthrough: Vanguard Credit Automation

This document outlines the final system configuration, model performance results, and local deployment steps for the Credit Card Approval Prediction System.

## 📁 Repository Lifecycle Restructuring
Per your mentor's repository template instructions, the project codebase and artifacts have been reorganized into 8 sequential project lifecycle directories:

1.  **[1. Brainstorming & Ideation](file:///home/david/credit_card_approval/1.%20Brainstorming%20&%20Ideation/)**: Documenting the problem statements, classification algorithms, and use case scenarios.
2.  **[2. Requirement Analysis](file:///home/david/credit_card_approval/2.%20Requirement%20Analysis/)**: Defining functional/non-functional requirements.
3.  **[3. Project Design Phase](file:///home/david/credit_card_approval/3.%20Project%20Design%20Phase/)**: Stating the 7-tier decoupling architecture design (User, Presentation, Application, ML, Data, Storage, Deployment).
4.  **[4. Project Planning Phase](file:///home/david/credit_card_approval/4.%20Project%20Planning%20Phase/)**: Storing the approved `implementation_plan.md` and `task.md` checklist.
5.  **[5. Project Development Phase](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/)**: The complete working application code (`app.py`, templates, static assets, serialized models, preprocessors, metrics, and dataset files).
6.  **[6. Project Testing](file:///home/david/credit_card_approval/6.Project%20Testing/)**: Verification report logging classifier performance metrics and browser subagent audit results.
7.  **[7. Project Documentation](file:///home/david/credit_card_approval/7.Project%20Documentation/)**: General user documentation and this walkthrough guide.
8.  **[8. Project Demonstration](file:///home/david/credit_card_approval/8.Project%20Demonstration/)**: Reference index containing browser-automated screenshots of the dashboard and analyst console log.

---

## 📊 Model Performance Matrix
The pipeline evaluated the classification models on applicant records and selected the model with the **Highest Accuracy** to serve real-time predictions:

*   **Decision Tree**: Accuracy = **65.20%** (Selected Active Model 🏆)
*   **XGBoost (Gradient Boosting)**: Accuracy = 65.00%
*   **Random Forest**: Accuracy = 64.80%
*   **Logistic Regression**: Accuracy = 64.60%

The model (`model.pkl`) and preprocessing object (`preprocessor.pkl`) are stored separately in the Model Storage tier.

---

## 🚀 Running the Web Application Locally

1.  **Navigate to the Development Directory**:
    ```bash
    cd "/home/david/credit_card_approval/5. Project Development Phase"
    ```
2.  **Activate the Virtual Environment**:
    ```bash
    source venv/bin/activate
    ```
3.  **Start the Flask Server**:
    ```bash
    python app.py
    ```
4.  **Access the Dashboard**:
    Open your browser and navigate to:
    👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🖼️ UI Demonstration screenshots (From Phase 8)
*   **Dashboard View**: [Model Dashboard Screenshot](file:///home/david/credit_card_approval/8.%20Project%20Demonstration/dashboard_view_1782972554239.png)
*   **Analyst Console & History Log**: [Analyst Console Screenshot](file:///home/david/credit_card_approval/8.%20Project%20Demonstration/analyst_evaluation_1782972654318.png)
