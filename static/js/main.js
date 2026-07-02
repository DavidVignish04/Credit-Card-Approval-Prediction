// Vanguard Credit Automation - Frontend Core Logic

document.addEventListener('DOMContentLoaded', () => {
    // State management
    let isServerOnline = false;
    let modelMetrics = null;
    let activeModel = 'Random Forest';
    let comparisonChartInstance = null;
    let currentComplianceBatch = [];
    const analystHistory = [];
    
    // Selectors
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const serverStatusDot = document.getElementById('server-status-dot');
    const serverStatusText = document.getElementById('server-status-text');
    const activeModelBadge = document.getElementById('active-model-badge');
    
    // KPI elements
    const kpiActiveModel = document.getElementById('kpi-active-model');
    const kpiAucRoc = document.getElementById('kpi-auc-roc');
    const kpiAccuracy = document.getElementById('kpi-accuracy');
    
    // Tab switching logic
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute('data-target');
            
            navItems.forEach(n => n.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            item.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Update Header Text based on Active Tab
            updateHeader(targetTab);
        });
    });
    
    function updateHeader(tabId) {
        if (tabId === 'dashboard-tab') {
            pageTitle.innerText = "Model Dashboard";
            pageSubtitle.innerText = "Real-time Credit Card Approval predictive modeling pipeline";
        } else if (tabId === 'analyst-tab') {
            pageTitle.innerText = "Analyst Console";
            pageSubtitle.innerText = "Screen credit card applicants using serialized ML models (Scenario 1)";
        } else if (tabId === 'compliance-tab') {
            pageTitle.innerText = "Compliance Portal";
            pageSubtitle.innerText = "Audit applicant lists and flag delinquencies (Scenario 2)";
        } else if (tabId === 'customer-tab') {
            pageTitle.innerText = "Eligibility Check";
            pageSubtitle.innerText = "Consumer self-service pre-qualification portal (Scenario 4)";
        }
    }

    // Connect to Flask backend
    async function checkBackendStatus() {
        serverStatusDot.className = 'status-indicator loading';
        serverStatusText.innerText = 'Connecting to Flask...';
        
        try {
            const response = await fetch('/api/model_stats');
            if (response.ok) {
                const data = await response.json();
                isServerOnline = true;
                modelMetrics = data.metrics;
                activeModel = data.best_model_name;
                
                serverStatusDot.className = 'status-indicator online';
                serverStatusText.innerText = 'Online (Flask Backend)';
                showToast('Connected to Python Flask Server!', 'success');
            } else {
                throw new Error('Server returned error status');
            }
        } catch (error) {
            console.warn("Flask backend offline or unreachable. Initializing client-side Simulation Mode.", error);
            isServerOnline = false;
            // Initialize with high-quality simulated metrics matching standard credit scores
            modelMetrics = {
                'Logistic Regression': {'accuracy': 0.8425, 'precision': 0.8115, 'recall': 0.7952, 'f1_score': 0.8033, 'roc_auc': 0.8841, 'confusion_matrix': [[312, 45], [54, 209]]},
                'Decision Tree': {'accuracy': 0.8654, 'precision': 0.8521, 'recall': 0.8123, 'f1_score': 0.8317, 'roc_auc': 0.9015, 'confusion_matrix': [[325, 32], [49, 214]]},
                'Random Forest': {'accuracy': 0.8988, 'precision': 0.8924, 'recall': 0.8485, 'f1_score': 0.8700, 'roc_auc': 0.9421, 'confusion_matrix': [[338, 19], [40, 223]]},
                'XGBoost': {'accuracy': 0.8951, 'precision': 0.8850, 'recall': 0.8522, 'f1_score': 0.8683, 'roc_auc': 0.9382, 'confusion_matrix': [[336, 21], [39, 224]]}
            };
            activeModel = 'XGBoost';
            
            serverStatusDot.className = 'status-indicator offline';
            serverStatusText.innerText = 'Offline (Simulation Mode)';
            showToast('Flask offline. Simulation Mode active.', 'info');
        }
        
        // Update dashboard KPIs and Badge
        activeModelBadge.innerText = activeModel.toUpperCase() + ' ACTIVE';
        kpiActiveModel.innerText = activeModel;
        
        const bestMetrics = modelMetrics[activeModel];
        if (bestMetrics) {
            kpiAucRoc.innerText = (bestMetrics.roc_auc * 100).toFixed(1) + '%';
            kpiAccuracy.innerText = (bestMetrics.accuracy * 100).toFixed(1) + '%';
        }
        
        // Render UI panels
        populateMetricsTable();
        renderComparisonChart();
    }
    
    // Fill metrics matrix table
    function populateMetricsTable() {
        const tbody = document.querySelector('#metrics-table tbody');
        tbody.innerHTML = '';
        
        Object.entries(modelMetrics).forEach(([name, data]) => {
            const isBest = name === activeModel;
            const tr = document.createElement('tr');
            if (isBest) tr.className = 'best-model-row';
            
            tr.innerHTML = `
                <td>
                    <strong>${name}</strong> 
                    ${isBest ? '<span class="badge badge-green" style="margin-left:5px">BEST</span>' : ''}
                </td>
                <td>${(data.accuracy * 100).toFixed(2)}%</td>
                <td>${(data.precision * 100).toFixed(2)}%</td>
                <td>${(data.recall * 100).toFixed(2)}%</td>
                <td>${(data.f1_score * 100).toFixed(2)}%</td>
                <td><strong class="text-indigo">${data.roc_auc.toFixed(4)}</strong></td>
                <td><span class="badge badge-green">Full Support</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Chart.js Performance Graph Rendering
    function renderComparisonChart() {
        const ctx = document.getElementById('modelComparisonChart').getContext('2d');
        
        const labels = Object.keys(modelMetrics);
        const accuracies = labels.map(l => (modelMetrics[l].accuracy * 100).toFixed(1));
        const precisions = labels.map(l => (modelMetrics[l].precision * 100).toFixed(1));
        const recalls = labels.map(l => (modelMetrics[l].recall * 100).toFixed(1));
        const f1s = labels.map(l => (modelMetrics[l].f1_score * 100).toFixed(1));
        const aucs = labels.map(l => (modelMetrics[l].roc_auc * 100).toFixed(1));
        
        if (comparisonChartInstance) {
            comparisonChartInstance.destroy();
        }
        
        comparisonChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: accuracies,
                        backgroundColor: 'rgba(99, 102, 241, 0.65)',
                        borderColor: '#6366f1',
                        borderWidth: 1
                    },
                    {
                        label: 'F1-Score (%)',
                        data: f1s,
                        backgroundColor: 'rgba(139, 92, 246, 0.65)',
                        borderColor: '#8b5cf6',
                        borderWidth: 1
                    },
                    {
                        label: 'ROC-AUC (%)',
                        data: aucs,
                        backgroundColor: 'rgba(59, 130, 246, 0.65)',
                        borderColor: '#3b82f6',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f8fafc',
                            font: { family: 'Outfit', size: 12 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    },
                    y: {
                        min: 50,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    }
                }
            }
        });
    }

    // ==========================================
    // SCENARIO 1: ANALYST CONSOLE
    // ==========================================
    const analystForm = document.getElementById('analyst-form');
    const btnLoadSampleAnalyst = document.getElementById('btn-load-sample-analyst');
    
    // Sliders syncing logic
    setupSliderSync('analyst-income', 'analyst-income-slider');
    setupSliderSync('analyst-employment', 'analyst-employment-slider');
    setupSliderSync('analyst-debt-ratio', 'analyst-debt-ratio-slider');
    
    function setupSliderSync(inputId, sliderId) {
        const input = document.getElementById(inputId);
        const slider = document.getElementById(sliderId);
        
        input.addEventListener('input', () => {
            slider.value = input.value;
        });
        
        slider.addEventListener('input', () => {
            input.value = slider.value;
        });
    }
    
    // Load mock applicant for analyst
    btnLoadSampleAnalyst.addEventListener('click', () => {
        const samples = [
            { income: 85000, employment: 6.5, debt: 0.18, loans: 1, inquiries: 0, status: 'C', car: true, prop: true, edu: 'Higher education', incType: 'Commercial associate' },
            { income: 28000, employment: 0.5, debt: 0.65, loans: 4, inquiries: 3, status: '1', car: false, prop: true, edu: 'Secondary / secondary special', incType: 'Working' },
            { income: 42000, employment: 2.0, debt: 0.32, loans: 2, inquiries: 1, status: '0', car: false, prop: false, edu: 'Secondary / secondary special', incType: 'Working' }
        ];
        
        const randomSample = samples[Math.floor(Math.random() * samples.length)];
        
        document.getElementById('analyst-income').value = randomSample.income;
        document.getElementById('analyst-income-slider').value = randomSample.income;
        document.getElementById('analyst-employment').value = randomSample.employment;
        document.getElementById('analyst-employment-slider').value = randomSample.employment;
        document.getElementById('analyst-debt-ratio').value = randomSample.debt;
        document.getElementById('analyst-debt-ratio-slider').value = randomSample.debt;
        document.getElementById('analyst-active-loans').value = randomSample.loans;
        document.getElementById('analyst-inquiries').value = randomSample.inquiries;
        document.getElementById('analyst-payment-status').value = randomSample.status;
        document.getElementById('analyst-own-car').checked = randomSample.car;
        document.getElementById('analyst-own-property').checked = randomSample.prop;
        document.getElementById('analyst-education-type').value = randomSample.edu;
        document.getElementById('analyst-income-type').value = randomSample.incType;
        
        showToast('Sample applicant profile loaded!', 'success');
    });
    
    // Analyst form submit scoring
    analystForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const payload = {
            Gender: 1, // default
            Age: 38,   // default
            Own_Car: document.getElementById('analyst-own-car').checked ? 1 : 0,
            Own_Property: document.getElementById('analyst-own-property').checked ? 1 : 0,
            Num_Children: 0,
            Annual_Income: parseFloat(document.getElementById('analyst-income').value),
            Income_Type: document.getElementById('analyst-income-type').value,
            Education_Type: document.getElementById('analyst-education-type').value,
            Employment_Duration_Years: parseFloat(document.getElementById('analyst-employment').value),
            Num_Active_Loans: parseInt(document.getElementById('analyst-active-loans').value),
            Credit_Inquiries_6M: parseInt(document.getElementById('analyst-inquiries').value),
            Debt_To_Income_Ratio: parseFloat(document.getElementById('analyst-debt-ratio').value),
            Payment_Status: document.getElementById('analyst-payment-status').value
        };
        
        const placeholder = document.getElementById('analyst-placeholder');
        const details = document.getElementById('analyst-details');
        
        placeholder.classList.add('hidden');
        details.classList.remove('hidden');
        
        let result = null;
        
        if (isServerOnline) {
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (response.ok) {
                    result = await response.json();
                } else {
                    throw new Error('Flask server prediction error');
                }
            } catch (err) {
                console.error("API Prediction failed, falling back to simulated logic", err);
                result = runSimulatedClassifier(payload);
            }
        } else {
            result = runSimulatedClassifier(payload);
        }
        
        // Display Analyst results
        displayAnalystResults(result);
    });
    
    function displayAnalystResults(result) {
        const resultCard = document.getElementById('analyst-result-card');
        const decisionBadge = document.getElementById('analyst-decision-badge');
        const decisionTitle = document.getElementById('analyst-decision-title');
        const scoringMethod = document.getElementById('analyst-scoring-method');
        const oddsPercent = document.getElementById('analyst-odds-percent');
        const progressCircle = document.getElementById('analyst-circle-progress');
        
        const posList = document.getElementById('analyst-positives-list');
        const negList = document.getElementById('analyst-negatives-list');
        
        // Set styling state
        if (result.approved === 1) {
            resultCard.className = 'card result-card approved-state';
            decisionBadge.innerText = 'APPROVED';
            decisionTitle.innerText = 'Credit Card Pre-Approved';
        } else {
            resultCard.className = 'card result-card rejected-state';
            decisionBadge.innerText = 'REJECTED';
            decisionTitle.innerText = 'Application Declined';
        }
        
        scoringMethod.innerText = `Scored by: ${result.model_used}`;
        oddsPercent.innerText = Math.round(result.probability * 100) + '%';
        
        // Circular progress dashoffset calculation
        const r = 45;
        const circumference = 2 * Math.PI * r; // ~282.7
        const offset = circumference - (result.probability * circumference);
        progressCircle.style.strokeDashoffset = offset;
        
        // Fill explanations
        posList.innerHTML = '';
        result.contributions.positives.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${item.feature}:</strong> ${item.desc}`;
            posList.appendChild(li);
        });
        
        negList.innerHTML = '';
        result.contributions.negatives.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${item.feature}:</strong> ${item.desc}`;
            negList.appendChild(li);
        });
        
        if (result.high_risk === 1) {
            showToast('Applicant flagged for delinquency record history!', 'error');
        } else {
            showToast('Evaluation completed successfully.', 'success');
        }
        
        // Log to session history (Scenario 1 & Technical Architecture)
        logPredictionToHistory(result);
    }
    
    function logPredictionToHistory(result) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        const incomeType = document.getElementById('analyst-income-type').value;
        const income = parseFloat(document.getElementById('analyst-income').value);
        const status = document.getElementById('analyst-payment-status').value;
        
        const historyItem = {
            timestamp: timeStr,
            incomeType: incomeType,
            annualIncome: income,
            paymentStatus: status,
            modelUsed: result.model_used,
            probability: result.probability,
            approved: result.approved
        };
        
        analystHistory.unshift(historyItem);
        if (analystHistory.length > 5) {
            analystHistory.pop();
        }
        
        updateHistoryTable();
    }
    
    function updateHistoryTable() {
        const tbody = document.querySelector('#analyst-history-table tbody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        if (analystHistory.length === 0) {
            tbody.innerHTML = `
                <tr id="history-empty-row">
                    <td colspan="7" class="text-center text-muted" style="padding: 2rem 0;">
                        <i class="fa-regular fa-folder-open" style="font-size: 1.5rem; display: block; margin-bottom: 0.5rem;"></i>
                        No evaluations logged in this session yet.
                    </td>
                </tr>
            `;
            return;
        }
        
        analystHistory.forEach(item => {
            const tr = document.createElement('tr');
            
            let statusMarkup = '';
            const status = String(item.paymentStatus);
            if (status === 'C') {
                statusMarkup = '<span class="badge badge-green">C (Paid)</span>';
            } else if (status === 'X') {
                statusMarkup = '<span class="badge badge-green">X (No debt)</span>';
            } else if (status === '0') {
                statusMarkup = '<span class="badge badge-amber">0 (Late &lt;30d)</span>';
            } else {
                statusMarkup = `<span class="badge badge-red">${status} (Overdue ${status}0d)</span>`;
            }
            
            const recommendation = item.approved === 1
                ? '<span class="badge badge-green"><i class="fa-solid fa-check"></i> Approved</span>'
                : '<span class="badge badge-red"><i class="fa-solid fa-times"></i> Rejected</span>';
                
            tr.innerHTML = `
                <td>${item.timestamp}</td>
                <td><strong>${item.incomeType}</strong></td>
                <td>$${item.annualIncome.toLocaleString()}</td>
                <td>${statusMarkup}</td>
                <td><small>${item.modelUsed}</small></td>
                <td><strong>${Math.round(item.probability * 100)}%</strong></td>
                <td>${recommendation}</td>
            `;
            
            tbody.appendChild(tr);
        });
    }

    // ==========================================
    // SCENARIO 2: COMPLIANCE PORTAL
    // ==========================================
    const btnLoadComplianceSample = document.getElementById('btn-load-compliance-sample');
    const btnRunComplianceScreening = document.getElementById('btn-run-compliance-screening');
    const complianceDragDrop = document.getElementById('compliance-drag-drop');
    const complianceResultsSection = document.getElementById('compliance-results-section');
    
    // Sample batch data reflecting Scenario 2 payment status conversions
    const sampleComplianceApplicants = [
        { Applicant_ID: "APP-4019", Name: "Robert Vance", Annual_Income: 52000, Payment_Status: "C", Employment_Duration_Years: 4, Credit_Inquiries_6M: 0, Debt_To_Income_Ratio: 0.15 },
        { Applicant_ID: "APP-8291", Name: "Angela Martin", Annual_Income: 34000, Payment_Status: "3", Employment_Duration_Years: 2, Credit_Inquiries_6M: 1, Debt_To_Income_Ratio: 0.38 }, // Delinquent 90d
        { Applicant_ID: "APP-5512", Name: "Oscar Martinez", Annual_Income: 95000, Payment_Status: "X", Employment_Duration_Years: 8, Credit_Inquiries_6M: 0, Debt_To_Income_Ratio: 0.11 },
        { Applicant_ID: "APP-1194", Name: "Ryan Howard", Annual_Income: 24000, Payment_Status: "1", Employment_Duration_Years: 0.5, Credit_Inquiries_6M: 4, Debt_To_Income_Ratio: 0.52 }, // Delinquent 30d
        { Applicant_ID: "APP-7703", Name: "Toby Flenderson", Annual_Income: 61000, Payment_Status: "0", Employment_Duration_Years: 5, Credit_Inquiries_6M: 1, Debt_To_Income_Ratio: 0.28 }, // Late <30d (normal risk)
        { Applicant_ID: "APP-9082", Name: "Stanley Hudson", Annual_Income: 78000, Payment_Status: "C", Employment_Duration_Years: 12, Credit_Inquiries_6M: 0, Debt_To_Income_Ratio: 0.21 },
        { Applicant_ID: "APP-2251", Name: "Kelly Kapoor", Annual_Income: 45000, Payment_Status: "4", Employment_Duration_Years: 1.5, Credit_Inquiries_6M: 3, Debt_To_Income_Ratio: 0.44 }, // Delinquent 120d
        { Applicant_ID: "APP-6102", Name: "Creed Bratton", Annual_Income: 21000, Payment_Status: "5", Employment_Duration_Years: 0.0, Credit_Inquiries_6M: 5, Debt_To_Income_Ratio: 0.85 }  // Delinquent 150d (written off)
    ];
    
    btnLoadComplianceSample.addEventListener('click', () => {
        currentComplianceBatch = sampleComplianceApplicants;
        btnRunComplianceScreening.removeAttribute('disabled');
        complianceDragDrop.classList.add('hidden');
        complianceResultsSection.classList.remove('hidden');
        
        // Initially show loaded records before scanning
        renderComplianceTable(false);
        showToast('Sample compliance list loaded. Ready for scan.', 'success');
    });
    
    // Simulate dropzone click
    document.getElementById('btn-simulate-upload').addEventListener('click', (e) => {
        e.stopPropagation();
        btnLoadComplianceSample.click();
    });
    
    complianceDragDrop.addEventListener('click', () => {
        btnLoadComplianceSample.click();
    });
    
    btnRunComplianceScreening.addEventListener('click', async () => {
        btnRunComplianceScreening.setAttribute('disabled', 'true');
        btnRunComplianceScreening.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Audits...';
        
        let batchResults = [];
        
        if (isServerOnline) {
            try {
                const response = await fetch('/api/predict_batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ applicants: currentComplianceBatch })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    batchResults = data.results;
                } else {
                    throw new Error('Flask server batch prediction error');
                }
            } catch (err) {
                console.error("Batch API call failed, running locally", err);
                batchResults = runSimulatedBatchClassifier(currentComplianceBatch);
            }
        } else {
            // Simulated delay for premium feel
            await new Promise(resolve => setTimeout(resolve, 800));
            batchResults = runSimulatedBatchClassifier(currentComplianceBatch);
        }
        
        // Render scanned results
        currentComplianceBatch = batchResults;
        renderComplianceTable(true);
        updateComplianceKPIs(batchResults);
        
        btnRunComplianceScreening.innerHTML = '<i class="fa-solid fa-filter"></i> Run Batch Compliance Scan';
        btnRunComplianceScreening.removeAttribute('disabled');
        showToast('Compliance screening completed!', 'success');
    });
    
    function renderComplianceTable(scanned) {
        const tbody = document.querySelector('#compliance-table tbody');
        tbody.innerHTML = '';
        
        currentComplianceBatch.forEach(app => {
            const tr = document.createElement('tr');
            
            // Payment Status Code Formatting
            let statusBadge = '';
            const status = String(app.payment_status);
            if (status === 'C') {
                statusBadge = '<span class="badge badge-green">C (Current)</span>';
            } else if (status === 'X') {
                statusBadge = '<span class="badge badge-green">X (No debt)</span>';
            } else if (status === '0') {
                statusBadge = '<span class="badge badge-amber">0 (Late &lt;30d)</span>';
            } else {
                statusBadge = `<span class="badge badge-red">${status} (Overdue ${status}0d)</span>`;
            }
            
            // High Risk Compliance conversion logic (Scenario 2 feature engineering)
            const isHighRisk = ['1', '2', '3', '4', '5'].includes(status) || app.high_risk === 1;
            const flagMarkup = isHighRisk 
                ? '<span class="badge badge-red"><i class="fa-solid fa-circle-exclamation"></i> HIGH RISK DELINQUENT</span>' 
                : '<span class="badge badge-green"><i class="fa-solid fa-circle-check"></i> COMPLIANT</span>';
                
            let likelihoodVal = '-';
            let decisionVal = '<span class="badge">Awaiting scan</span>';
            
            if (scanned) {
                likelihoodVal = `<strong>${Math.round(app.probability * 100)}%</strong>`;
                decisionVal = app.approved === 1 
                    ? '<span class="badge badge-green"><i class="fa-solid fa-check"></i> AUTO-APPROVED</span>' 
                    : `<span class="badge badge-red"><i class="fa-solid fa-times"></i> ${isHighRisk ? 'REJECT (COMPLIANCE)' : 'DECLINED (RISK)'}</span>`;
            }
            
            tr.innerHTML = `
                <td><strong>${app.Applicant_ID || 'APP-'}</strong></td>
                <td>${app.Name}</td>
                <td>$${parseFloat(app.annual_income).toLocaleString()}</td>
                <td>${statusBadge}</td>
                <td>${flagMarkup}</td>
                <td>${likelihoodVal}</td>
                <td>${decisionVal}</td>
            `;
            
            tbody.appendChild(tr);
        });
    }
    
    function updateComplianceKPIs(results) {
        const total = results.length;
        const highRisk = results.filter(r => r.high_risk === 1).length;
        const approved = results.filter(r => r.approved === 1).length;
        
        document.getElementById('batch-total-count').innerText = total;
        document.getElementById('batch-high-risk-count').innerText = highRisk;
        document.getElementById('batch-approved-count').innerText = approved;
        document.getElementById('batch-risk-rate').innerText = ((highRisk / total) * 100).toFixed(0) + '%';
    }

    // ==========================================
    // SCENARIO 4: CUSTOMER SELF-SERVICE WIZARD
    // ==========================================
    const wizardPanes = document.querySelectorAll('.wizard-step-pane');
    const wizardSteps = document.querySelectorAll('.progress-step');
    const wizardProgressBar = document.getElementById('wizard-progress-bar');
    
    // Wizard step buttons
    document.querySelectorAll('.btn-next').forEach(btn => {
        btn.addEventListener('click', () => {
            const nextStep = parseInt(btn.getAttribute('data-next'));
            switchStep(nextStep);
        });
    });
    
    document.querySelectorAll('.btn-prev').forEach(btn => {
        btn.addEventListener('click', () => {
            const prevStep = parseInt(btn.getAttribute('data-prev'));
            switchStep(prevStep);
        });
    });
    
    function switchStep(stepNum) {
        // Update panes
        wizardPanes.forEach(pane => pane.classList.remove('active'));
        document.getElementById(`customer-step-${stepNum}`).classList.remove('active');
        document.getElementById(`customer-step-${stepNum}`).classList.add('active');
        
        // Update Progress bar bubble styles
        wizardSteps.forEach(step => {
            const sIdx = parseInt(step.getAttribute('data-step'));
            step.className = 'progress-step';
            if (sIdx < stepNum) {
                step.classList.add('completed');
            } else if (sIdx === stepNum) {
                step.classList.add('active');
            }
        });
        
        // Progress line percentage
        const progressPct = ((stepNum - 1) / (wizardSteps.length - 1)) * 90;
        wizardProgressBar.style.width = progressPct + '%';
    }
    
    // Submit Wizard Evaluation
    document.getElementById('btn-submit-customer-wizard').addEventListener('click', async () => {
        // Go to Step 3 (Results Loading)
        switchStep(3);
        
        const loadingBox = document.getElementById('cust-loading');
        const resultsBox = document.getElementById('cust-results-final');
        
        loadingBox.classList.remove('hidden');
        resultsBox.classList.add('hidden');
        
        // Calculate dynamic inputs
        const name = document.getElementById('cust-name').value;
        const income = parseFloat(document.getElementById('cust-income').value);
        const debt = parseFloat(document.getElementById('cust-debt').value);
        const dti = income > 0 ? (debt * 12) / income : 0.5;
        
        const payload = {
            Gender: parseInt(document.getElementById('cust-gender').value),
            Age: parseFloat(document.getElementById('cust-age').value),
            Own_Car: document.getElementById('cust-own-car').checked ? 1 : 0,
            Own_Property: document.getElementById('cust-own-prop').checked ? 1 : 0,
            Num_Children: 0,
            Annual_Income: income,
            Income_Type: 'Working',
            Education_Type: document.getElementById('cust-education').value,
            Employment_Duration_Years: parseFloat(document.getElementById('cust-employment').value),
            Num_Active_Loans: 1, // assume average
            Credit_Inquiries_6M: parseInt(document.getElementById('cust-inquiries').value),
            Debt_To_Income_Ratio: dti,
            Payment_Status: document.getElementById('cust-payment-status').value
        };
        
        let result = null;
        
        // Simulated process duration for bank credit check scoring
        await new Promise(resolve => setTimeout(resolve, 1200));
        
        if (isServerOnline) {
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (response.ok) {
                    result = await response.json();
                } else {
                    throw new Error('Customer prediction API error');
                }
            } catch (err) {
                console.error("API Prediction failed, running locally for customer wizard", err);
                result = runSimulatedClassifier(payload);
            }
        } else {
            result = runSimulatedClassifier(payload);
        }
        
        // Display Customer results
        displayCustomerResults(name, result, payload);
    });
    
    function displayCustomerResults(name, result, input) {
        const loadingBox = document.getElementById('cust-loading');
        const resultsBox = document.getElementById('cust-results-final');
        
        loadingBox.classList.add('hidden');
        resultsBox.classList.remove('hidden');
        
        const resultsCard = document.querySelector('.customer-results-card');
        const customerShield = document.getElementById('customer-shield');
        const customerResultTitle = document.getElementById('customer-result-title');
        const customerOddsBar = document.getElementById('customer-odds-bar');
        const customerOddsLabel = document.getElementById('customer-odds-label');
        const customerCtaMsg = document.getElementById('customer-cta-msg');
        
        const probabilityPct = Math.round(result.probability * 100);
        
        customerOddsBar.style.width = probabilityPct + '%';
        customerOddsLabel.innerText = `Pre-Approval Probability: ${probabilityPct}%`;
        
        if (result.approved === 1) {
            resultsCard.className = 'card customer-results-card'; // clear classes
            customerShield.innerHTML = '<i class="fa-solid fa-shield-halved"></i>';
            customerResultTitle.innerText = `Congratulations ${name}, You are Pre-Approved!`;
            customerCtaMsg.innerText = "Based on our machine learning assessment, your financial profile meets the necessary approval benchmarks. You may proceed to submit a formal application.";
        } else {
            resultsCard.className = 'card customer-results-card rejected-odds';
            customerShield.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            customerResultTitle.innerText = `Hi ${name}, Your Eligibility Score is Low`;
            customerCtaMsg.innerText = "Unfortunately, based on the current predictive scoring models, your profile falls below our standard eligibility parameters. Check our feedback insights below to improve your chances.";
        }
        
        // Populate Customer Insights
        const insightsList = document.getElementById('customer-insights-list');
        insightsList.innerHTML = '';
        
        // Positive insights
        result.contributions.positives.forEach(item => {
            const div = document.createElement('div');
            div.className = 'insight-item positive';
            div.innerHTML = `
                <i class="fa-solid fa-circle-check"></i>
                <div>
                    <strong>${item.feature}</strong>: ${item.desc}
                </div>
            `;
            insightsList.appendChild(div);
        });
        
        // Risk factor alerts and constructive advice
        result.contributions.negatives.forEach(item => {
            const div = document.createElement('div');
            div.className = 'insight-item negative';
            
            // Generate helpful tips for customer rejection factors
            let tip = '';
            if (item.feature === 'Annual Income') {
                tip = ' Tip: Consider applying with a co-signer or exploring low-income card tiers.';
            } else if (item.feature === 'Recent Inquiries') {
                tip = ' Tip: Avoid running credit inquiries or applying for other loans for the next 3 to 6 months.';
            } else if (item.feature === 'Debt-to-Income') {
                tip = ' Tip: Try reducing existing personal loan or credit card balances to lower your debt-to-income ratio below 30%.';
            } else if (item.feature === 'Payment Status') {
                tip = ' Tip: Settle any overdue delinquencies immediately and establish 6-12 months of consistent on-time payments.';
            } else if (item.feature === 'Employment Tenure') {
                tip = ' Tip: Establishing at least 1-2 years of stable employment history significantly improves bank confidence metrics.';
            }
            
            div.innerHTML = `
                <i class="fa-solid fa-circle-exclamation text-red"></i>
                <div>
                    <strong>${item.feature}</strong>: ${item.desc}<br>
                    <span class="text-indigo" style="font-size:0.75rem">${tip}</span>
                </div>
            `;
            insightsList.appendChild(div);
        });
        
        showToast('Self-Service Assessment Loaded.', 'success');
    }
    
    // Restart customer wizard
    document.getElementById('btn-restart-customer').addEventListener('click', () => {
        switchStep(1);
    });

    // ==========================================
    // SIMULATED MACHINE LEARNING CLASSIFIER (JS Fallback)
    // ==========================================
    function runSimulatedClassifier(data) {
        const annualIncome = parseFloat(data.Annual_Income || 50000);
        const employmentDurationYears = parseFloat(data.Employment_Duration_Years || 2);
        const numActiveLoans = parseInt(data.Num_Active_Loans || 0);
        const creditInquiries6m = parseInt(data.Credit_Inquiries_6M || 0);
        const debtToIncomeRatio = parseFloat(data.Debt_To_Income_Ratio || 0.2);
        const paymentStatus = String(data.Payment_Status || 'C');
        
        const highRisk = ['1', '2', '3', '4', '5'].includes(paymentStatus) ? 1 : 0;
        
        // Construct approval probability
        const incomeScore = Math.min(1.0, Math.max(0.0, (annualIncome - 20000) / 100000));
        const employmentScore = Math.min(1.0, Math.max(0.0, employmentDurationYears / 15));
        const debtPenalty = debtToIncomeRatio * 1.5;
        const inquiryPenalty = creditInquiries6m * 0.15;
        
        let prob = 0.45 + 0.3 * incomeScore + 0.2 * employmentScore - debtPenalty - inquiryPenalty - 0.6 * highRisk;
        prob = Math.min(0.99, Math.max(0.01, prob));
        
        let approved = prob > 0.5 ? 1 : 0;
        
        // Strict business rules
        if (['4', '5'].includes(paymentStatus) || creditInquiries6m >= 5 || annualIncome < 22000) {
            approved = 0;
            prob = Math.min(prob, 0.05);
        }
        
        // Explanations generator
        const contributions = generateLocalContributions(data, prob);
        
        return {
            approved: approved,
            probability: prob,
            model_used: 'XGBoost Classifier (Local Simulation)',
            contributions: contributions,
            high_risk: highRisk
        };
    }
    
    function runSimulatedBatchClassifier(batch) {
        return batch.map(app => {
            const sim = runSimulatedClassifier(app);
            return {
                Applicant_ID: app.Applicant_ID,
                Name: app.Name,
                annual_income: app.Annual_Income,
                payment_status: app.Payment_Status,
                high_risk: sim.high_risk,
                approved: sim.approved,
                probability: sim.probability,
                risk_status: sim.high_risk === 1 ? 'High Risk Delinquent' : 'Normal credit history'
            };
        });
    }
    
    function generateLocalContributions(data, approvalProb) {
        const positives = [];
        const negatives = [];
        
        const annualIncome = parseFloat(data.Annual_Income || 50000);
        const employmentDurationYears = parseFloat(data.Employment_Duration_Years || 2);
        const creditInquiries6m = parseInt(data.Credit_Inquiries_6M || 0);
        const debtToIncomeRatio = parseFloat(data.Debt_To_Income_Ratio || 0.2);
        const paymentStatus = String(data.Payment_Status || 'C');
        
        // Income
        if (annualIncome >= 80000) {
            positives.push({feature: 'Annual Income', desc: `Robust annual earnings of $${annualIncome.toLocaleString()} demonstrates solid financial capacity.`});
        } else if (annualIncome >= 45000) {
            positives.push({feature: 'Annual Income', desc: 'Stable income levels within standard approval ranges.'});
        } else {
            negatives.push({feature: 'Annual Income', desc: `Income level ($${annualIncome.toLocaleString()}) is below preferred credit limits.`});
        }
        
        // Employment
        if (employmentDurationYears >= 5) {
            positives.push({feature: 'Employment Tenure', desc: `${employmentDurationYears} years in job demonstrates reliable job security.`});
        } else if (employmentDurationYears < 1) {
            negatives.push({feature: 'Employment Tenure', desc: 'Job tenure under 1 year indicates potential transition risk.'});
        }
        
        // DTI
        if (debtToIncomeRatio <= 0.25) {
            positives.push({feature: 'Debt-to-Income', desc: `Low debt utilization (${(debtToIncomeRatio*100).toFixed(1)}%) leaves room for credit lines.`});
        } else if (debtToIncomeRatio > 0.45) {
            negatives.push({feature: 'Debt-to-Income', desc: `High leverage (${(debtToIncomeRatio*100).toFixed(1)}%) significantly limits repayment capacity.`});
        }
        
        // Inquiries
        if (creditInquiries6m === 0) {
            positives.push({feature: 'Recent Inquiries', desc: 'No credit search inquiries in the past 6 months.'});
        } else if (creditInquiries6m >= 3) {
            negatives.push({feature: 'Recent Inquiries', desc: `${creditInquiries6m} hard credit pulls indicates high credit-seeking behavior.`});
        }
        
        // Payments
        if (['C', 'X'].includes(paymentStatus)) {
            positives.push({feature: 'Payment Status', desc: 'Flawless payment records with no history of default.'});
        } else if (paymentStatus === '0') {
            negatives.push({feature: 'Payment Status', desc: 'Minor short-term late payments (under 30 days) noticed.'});
        } else {
            negatives.push({feature: 'Payment Status', desc: 'Delinquent accounts (30+ days past due) flagged as high compliance risk.'});
        }
        
        if (positives.length === 0) positives.push({feature: 'Demographics', desc: 'Basic demographic factors align within acceptable parameters.'});
        if (negatives.length === 0) negatives.push({feature: 'Account Flags', desc: 'No risk alerts triggered on active accounts.'});
        
        return { positives, negatives };
    }

    // ==========================================
    // TOAST NOTIFICATIONS
    // ==========================================
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = '<i class="fa-solid fa-info-circle text-indigo"></i>';
        if (type === 'success') {
            icon = '<i class="fa-solid fa-circle-check text-green"></i>';
        } else if (type === 'error') {
            icon = '<i class="fa-solid fa-triangle-exclamation text-red"></i>';
        }
        
        toast.innerHTML = `
            ${icon}
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Auto remove
        setTimeout(() => {
            toast.style.animation = 'fadeIn 0.3s ease reverse';
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 3500);
    }

    // ==========================================
    // MODAL DIALOG MANAGEMENT (Watson Logs)
    // ==========================================
    const btnShowWatsonLogs = document.getElementById('btn-trigger-watson-pipeline');
    const watsonLogsModal = document.getElementById('watson-logs-modal');
    const btnCloseModalX = document.getElementById('btn-close-logs-modal');
    const btnCloseModalBottom = document.getElementById('btn-close-modal-bottom');
    const btnCopyLogs = document.getElementById('btn-copy-logs');
    
    btnShowWatsonLogs.addEventListener('click', () => {
        watsonLogsModal.classList.add('open');
    });
    
    function closeModal() {
        watsonLogsModal.classList.remove('open');
    }
    
    btnCloseModalX.addEventListener('click', closeModal);
    btnCloseModalBottom.addEventListener('click', closeModal);
    
    // Close on overlay click
    watsonLogsModal.addEventListener('click', (e) => {
        if (e.target === watsonLogsModal) {
            closeModal();
        }
    });
    
    btnCopyLogs.addEventListener('click', () => {
        const logContent = document.getElementById('terminal-logs-content').innerText;
        navigator.clipboard.writeText(logContent).then(() => {
            showToast('Logs copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy logs: ', err);
            showToast('Failed to copy logs.', 'error');
        });
    });

    // Start everything by checking Flask API connection status
    checkBackendStatus();
});
