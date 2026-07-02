import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Check if xgboost is installed, fallback to GradientBoostingClassifier if not
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGBOOST = False

print(f"XGBoost installation status: {'Found (XGBClassifier will be used)' if HAS_XGBOOST else 'Not Found (Using sklearn GradientBoostingClassifier as equivalent)'}")

def generate_synthetic_data(n_samples=2500, random_state=42):
    """
    Generates a realistic credit card applicant dataset.
    """
    np.random.seed(random_state)
    
    # Generate Demographics
    gender = np.random.binomial(1, 0.48, n_samples) # 0: Male, 1: Female
    age = np.random.uniform(20, 70, n_samples)
    own_car = np.random.binomial(1, 0.35, n_samples)
    own_property = np.random.binomial(1, 0.65, n_samples)
    num_children = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.18, 0.1, 0.02])
    
    # Income (log-normal distribution for realism)
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples) # Median around $50k
    
    # Income & Education Types
    income_types = np.random.choice(
        ['Working', 'Commercial associate', 'State servant', 'Pensioner', 'Student'],
        size=n_samples,
        p=[0.55, 0.25, 0.12, 0.075, 0.005]
    )
    education_types = np.random.choice(
        ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
        size=n_samples,
        p=[0.70, 0.24, 0.04, 0.018, 0.002]
    )
    
    # Employment duration: pensioners get 0 or very small, working depends on age
    employment_duration_years = []
    for idx, inc_type in enumerate(income_types):
        if inc_type == 'Pensioner':
            employment_duration_years.append(0.0)
        else:
            max_work = max(0.1, age[idx] - 18)
            duration = np.random.exponential(scale=6.0)
            duration = min(duration, max_work)
            employment_duration_years.append(round(duration, 1))
    employment_duration_years = np.array(employment_duration_years)
    
    # Financial metrics
    num_active_loans = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.18, 0.08, 0.03, 0.01])
    credit_inquiries_6m = np.random.choice([0, 1, 2, 3, 4, 5, 8], size=n_samples, p=[0.6, 0.2, 0.1, 0.06, 0.02, 0.01, 0.01])
    
    debt_to_income_ratio = np.random.beta(a=2, b=5, size=n_samples) * 0.9 # range 0 to 0.9, peak around 0.25
    
    # Payment status (delinquency codes)
    # C: paid off, X: no activity, 0: 1-29 days past due, 1: 30-59, 2: 60-89, 3: 90-119, 4: 120-149, 5: 150+
    payment_status_raw = np.random.choice(
        ['C', 'X', '0', '1', '2', '3', '4', '5'],
        size=n_samples,
        p=[0.50, 0.30, 0.12, 0.04, 0.02, 0.01, 0.005, 0.005]
    )
    
    # Build dataframe
    df = pd.DataFrame({
        'Gender': gender,
        'Age': np.round(age, 1),
        'Own_Car': own_car,
        'Own_Property': own_property,
        'Num_Children': num_children,
        'Annual_Income': np.round(annual_income, -2),
        'Income_Type': income_types,
        'Education_Type': education_types,
        'Employment_Duration_Years': employment_duration_years,
        'Num_Active_Loans': num_active_loans,
        'Credit_Inquiries_6M': credit_inquiries_6m,
        'Debt_To_Income_Ratio': np.round(debt_to_income_ratio, 3),
        'Payment_Status': payment_status_raw
    })
    
    # Define Target Logic (Credit Card Approval Decision)
    # Convert payment status to binary risk: 1, 2, 3, 4, 5 mean delinquency >= 30 days (High Risk)
    df['High_Risk'] = df['Payment_Status'].isin(['1', '2', '3', '4', '5']).astype(int)
    
    # Base approval probability based on financial parameters
    # High Income, long employment, low debt ratio, low inquiries, and NOT high risk favor approval.
    income_score = np.clip((df['Annual_Income'] - 20000) / 100000, 0, 1)
    employment_score = np.clip(df['Employment_Duration_Years'] / 15, 0, 1)
    debt_penalty = df['Debt_To_Income_Ratio'] * 2.0  # ratio > 0.5 starts hurting significantly
    inquiry_penalty = (df['Credit_Inquiries_6M'] / 3.0) # inquiries > 2 hurt a lot
    
    prob = 0.45 + 0.3 * income_score + 0.2 * employment_score - 0.3 * debt_penalty - 0.25 * inquiry_penalty - 0.7 * df['High_Risk']
    
    # Restrict to [0, 1]
    prob = np.clip(prob, 0.01, 0.99)
    
    # Sample actual approval binary label
    df['Approved'] = (np.random.rand(n_samples) < prob).astype(int)
    
    # Small rule-based overrides for realism:
    # 1. Extreme high risk (e.g. status 4 or 5) -> Rejected almost surely
    df.loc[df['Payment_Status'].isin(['4', '5']), 'Approved'] = 0
    # 2. Too many inquiries (e.g. 5 or more) -> Rejected
    df.loc[df['Credit_Inquiries_6M'] >= 5, 'Approved'] = 0
    # 3. Income too low (< 22,000) -> Rejected
    df.loc[df['Annual_Income'] < 22000, 'Approved'] = 0
    
    return df

def preprocess_and_split(df):
    """
    Encodes categories, scales data, splits into Train & Test sets.
    """
    # Copy dataframe
    data = df.copy()
    
    # Perform label encoding for categorical columns
    categorical_cols = ['Income_Type', 'Education_Type', 'Payment_Status']
    mappings = {}
    
    # We will map categorical types to integers explicitly to easily re-use in JS or backend
    for col in categorical_cols:
        unique_vals = sorted(data[col].unique())
        mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
        data[col] = data[col].map(mappings[col])
    
    # Define features and label
    feature_cols = [
        'Gender', 'Age', 'Own_Car', 'Own_Property', 'Num_Children', 
        'Annual_Income', 'Income_Type', 'Education_Type', 'Employment_Duration_Years',
        'Num_Active_Loans', 'Credit_Inquiries_6M', 'Debt_To_Income_Ratio', 'High_Risk'
    ]
    
    X = data[feature_cols]
    y = data['Approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Num_Children', 'Annual_Income', 'Employment_Duration_Years', 
                      'Num_Active_Loans', 'Credit_Inquiries_6M', 'Debt_To_Income_Ratio']
    
    # We fit the scaler on the training numeric features
    scaler.fit(X_train[numerical_cols])
    
    return X_train, X_test, y_train, y_test, scaler, numerical_cols, mappings, feature_cols

def train_and_evaluate(X_train, X_test, y_train, y_test, scaler, numerical_cols):
    """
    Trains 4 models and returns metrics.
    """
    # Scale the features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'confusion_matrix': cm
        }
        
        trained_models[name] = model
        print(f"  {name} - F1: {results[name]['f1_score']:.4f}, AUC: {results[name]['roc_auc']:.4f}")
        
    return trained_models, results, X_train_scaled, X_test_scaled

def main():
    # Ensure save paths exist
    os.makedirs('/home/david/credit_card_approval', exist_ok=True)
    
    print("Generating synthetic credit card application dataset...")
    df = generate_synthetic_data(n_samples=2500)
    
    # Save raw sample dataset to CSV for compliance portal testing
    df_sample = df.head(100).copy()
    df_sample.to_csv('/home/david/credit_card_approval/sample_applicants.csv', index=False)
    print("Saved sample CSV for compliance testing.")
    
    X_train, X_test, y_train, y_test, scaler, numerical_cols, mappings, feature_cols = preprocess_and_split(df)
    
    trained_models, results, X_train_scaled, X_test_scaled = train_and_evaluate(
        X_train, X_test, y_train, y_test, scaler, numerical_cols
    )
    
    # Select the best performing model based on Accuracy (matching technical architecture diagram)
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = trained_models[best_model_name]
    print(f"\nBest Model Selected: {best_model_name} with Accuracy {results[best_model_name]['accuracy']:.4f}")
    
    # Save Model File separately (model.pkl)
    model_path = '/home/david/credit_card_approval/model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved Model File to {model_path}")
    
    # Save Preprocessing Object separately (preprocessor.pkl wrapping scaler & mappings)
    preprocessor_payload = {
        'best_model_name': best_model_name,
        'scaler': scaler,
        'numerical_cols': numerical_cols,
        'feature_cols': feature_cols,
        'mappings': mappings,
        'xgboost_status': 'installed' if HAS_XGBOOST else 'simulated'
    }
    preprocessor_path = '/home/david/credit_card_approval/preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor_payload, f)
    print(f"Saved Preprocessing Object to {preprocessor_path}")
    
    # Save metrics JSON for frontend UI display
    metrics_path = '/home/david/credit_card_approval/model_metrics.json'
    metrics_payload = {
        'best_model_name': best_model_name,
        'metrics': results
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Exported metrics JSON to {metrics_path}")

if __name__ == '__main__':
    main()
