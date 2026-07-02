# Implementation Plan - Credit Card Approval Prediction System

This project implements an automated, end-to-end credit card approval system using Machine Learning. It trains four classification algorithms (Logistic Regression, Decision Tree, Random Forest, and XGBoost), compares their performance, saves the best model, and serves it through a premium Flask web application. It also provides an interactive user interface covering three distinct business scenarios and includes an IBM Watson ML deployment pipeline script.

## Workspace Recommendation
We will create the project inside `/home/david/.gemini/antigravity/scratch/credit_card_approval`. It is recommended to set this subdirectory as your active workspace.

## Proposed Changes

### Backend Component (Python / Flask)

#### [NEW] [train_models.py](file:///home/david/.gemini/antigravity/scratch/credit_card_approval/train_models.py)
A Python script that:
1. Generates a realistic synthetic credit card application dataset (e.g., 2,000+ records) with features like Age, Income, Employment Duration, Active Loans, Credit Inquiries, and Payment Status.
2. Performs feature engineering (e.g., mapping multi-class payment status codes to binary labels, scaling numerical features).
3. Trains four models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost (if `xgboost` is installed, otherwise a Gradient Boosting Classifier from `scikit-learn` as a highly compatible alternative).
4. Evaluates all models on accuracy, precision, recall, and F1-score.
5. Saves the best-performing model using `pickle` / `joblib`.
6. Saves evaluation metrics as a JSON file to be displayed on the frontend.

#### [NEW] [app.py](file:///home/david/.gemini/antigravity/scratch/credit_card_approval/app.py)
A Flask server that:
1. Loads the saved best-performing model.
2. Exposes API endpoints:
   - `/api/predict`: Runs inference on a single applicant profile.
   - `/api/predict_batch`: Runs inference on multiple profiles for batch compliance screening (Scenario 2).
   - `/api/model_stats`: Serves training metrics and model comparison stats to display on the dashboard.
3. Serves the frontend static files.

#### [NEW] [watson_pipeline.py](file:///home/david/.gemini/antigravity/scratch/credit_card_approval/watson_pipeline.py)
A script illustrating how to connect to IBM Watson Machine Learning, store the model in the Watson repository, and deploy it as an online scoring endpoint.

---

### Frontend Component (HTML, CSS, JS)

We will build a single-page dashboard with a premium, sleek banking aesthetic (glassmorphism, interactive animations, and responsive layout) using vanilla HTML5, CSS3, and JavaScript (with Chart.js loaded via CDN for beautiful charts).

#### [NEW] [templates/index.html](file:///home/david/.gemini/antigravity/scratch/credit_card_approval/templates/index.html)
The structure of our beautiful interface. It will contain:
- **Navbar**: Sleek logo, connection status, active workspace indicator, and view toggle buttons.
- **Section 1: Dashboard Overview**: Displays model performance comparison (a beautiful Chart.js chart comparing AUC-ROC, accuracy, etc.), dataset overview, and saved model metadata.
- **Section 2: Credit Analyst Console (Scenario 1)**: Interactive financial profile entry form with sliders, dropdowns, and an immediate visual prediction card (Approved/Rejected with probability details).
- **Section 3: Compliance Portal (Scenario 2)**: A section where the user can upload or generate mock batch customer files, see a table converting multi-class payment statuses to binary, and view high-risk flags.
- **Section 4: Customer Self-Service (Scenario 4)**: A consumer-friendly wizard with steps, credit health score estimation, and action items (personalized recommendations to improve approval odds).

#### [NEW] [static/css/style.css](file:///home/david/.gemini/antigravity/scratch/credit_card_approval/static/css/style.css)
A custom, premium styling system featuring:
- Sleek dark theme with gradients (deep navy, emerald green for approval, crimson red for rejection, amber for warnings, and gold/white text).
- Glassmorphism panels (`backdrop-filter: blur(12px)`).
- Custom progress steps for the Customer wizard.
- Smooth transitions and hover animations.

#### [NEW] [static/js/main.js](file:///home/david/.gemini/antigravity/scratch/credit_card_approval/static/js/main.js)
The logic layer that:
- Handles view transitions without page reloads.
- Performs API calls to the Flask backend for single and batch predictions.
- Renders Chart.js visualization of model metrics.
- Formats results, computes recommendations for customers, and handles CSV batch generation/parsing.
- Automatically falls back to a Javascript mock classifier client-side if the Flask server is not currently running (ensuring the website is fully functional and interactive in any static web environment).

---

## Verification Plan

### Automated Verification
1. Run `python train_models.py` to ensure dataset generation, training, evaluation, and model serialization function without errors.
2. Start Flask with `python app.py` and verify API endpoints using `curl` or by visiting `http://localhost:5000`.

### Manual Verification
1. Open the web interface.
2. Test **Scenario 1 (Analyst)**: Enter data, click "Evaluate Application", verify the model's approval status and percentage.
3. Test **Scenario 2 (Compliance)**: Click "Load Sample Compliance Batch", view binary classification conversions, and observe flagged high-risk records.
4. Test **Scenario 4 (Customer)**: Go through the stepped wizard, enter personal details, and see the eligibility score with dynamic improvement recommendations.
5. Check responsiveness on mobile/tablet viewports.
