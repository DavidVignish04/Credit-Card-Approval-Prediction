# AI-ML-and-GEN-AI-Track-Project-Template: Vanguard Credit Automation

This repository houses the end-to-end automated credit card approval prediction application, structured according to project lifecycle templates.

---

## 📁 Repository Structure

### [1. Brainstorming & Ideation](file:///home/david/credit_card_approval/1.%20Brainstorming%20&%20Ideation/)
*   [brainstorming_ideation.md](file:///home/david/credit_card_approval/1.%20Brainstorming%20&%20Ideation/brainstorming_ideation.md): Outlines the project background, 3 business use case scenarios (Analyst, Compliance, Customer wizard), and predictive ML classifiers selection.

### [2. Requirement Analysis](file:///home/david/credit_card_approval/2.%20Requirement%20Analysis/)
*   [requirement_analysis.md](file:///home/david/credit_card_approval/2.%20Requirement%20Analysis/requirement_analysis.md): Details functional requirements (API routes, ML scaling logic) and non-functional requirements (response limits, glassmorphic dark theme tokens).

### [3. Project Design Phase](file:///home/david/credit_card_approval/3.%20Project%20Design%20Phase/)
*   [project_design.md](file:///home/david/credit_card_approval/3.%20Project%20Design%20Phase/project_design.md): Specifies the 7-tier decoupling architecture (User, Presentation, Application, Machine Learning, Data, Model Storage, Deployment layers).

### [4. Project Planning Phase](file:///home/david/credit_card_approval/4.%20Project%20Planning%20Phase/)
*   [implementation_plan.md](file:///home/david/credit_card_approval/4.%20Project%20Planning%20Phase/implementation_plan.md): The detailed design specification plan approved for development.
*   [task.md](file:///home/david/credit_card_approval/4.%20Project%20Planning%20Phase/task.md): Project checklist and task tracker.

### [5. Project Development Phase](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/)
*   [app.py](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/app.py): Python Flask web server routing APIs and predictions.
*   [train_models.py](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/train_models.py): Dataset generator, trains 4 classifiers, auto-selects best model based on accuracy, and serializes artifacts.
*   [watson_pipeline.py](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/watson_pipeline.py): Cloud hosting demonstration client for IBM Watson ML.
*   [model.pkl](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/model.pkl): Serialized classifier model.
*   [preprocessor.pkl](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/preprocessor.pkl): Serialized preprocessor object.
*   [model_metrics.json](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/model_metrics.json): Exported model accuracy metrics.
*   [sample_applicants.csv](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/sample_applicants.csv): Delinquency audit mock records list.
*   [templates/index.html](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/templates/index.html): Responsive single-page dashboard frame.
*   [static/](file:///home/david/credit_card_approval/5.%20Project%20Development%20Phase/static/): CSS styling stylesheets and JS controllers.

### [6. Project Testing](file:///home/david/credit_card_approval/6.Project%20Testing/)
*   [testing_report.md](file:///home/david/credit_card_approval/6.Project%20Testing/testing_report.md): Log files detailing classifier accuracy scoring outputs and browser-agent scenario PASS audits.

### [7. Project Documentation](file:///home/david/credit_card_approval/7.Project%20Documentation/)
*   [walkthrough.md](file:///home/david/credit_card_approval/7.Project%20Documentation/walkthrough.md): Comprehensive project walkthrough and local execution instructions.

### [8. Project Demonstration](file:///home/david/credit_card_approval/8.Project%20Demonstration/)
*   [demonstration_guide.md](file:///home/david/credit_card_approval/8.Project%20Demonstration/demonstration_guide.md): Reference guide linking to model dashboard and analyst console screenshots.

---

## 🚀 Running the Web Application Locally

Navigate inside the development directory:
```bash
cd "/home/david/credit_card_approval/5. Project Development Phase"
```

Activate the environment and start the web server:
```bash
source venv/bin/activate
python app.py
```

Access the dashboard at:
👉 **[http://localhost:5000](http://localhost:5000)**