"""
IBM Watson Machine Learning Deployment Pipeline for Credit Card Approval Model.

This script demonstrates how to deploy the trained Random Forest / XGBoost model 
to IBM Watson Machine Learning (WML) on IBM Cloud for scalable, cloud-hosted real-time inference.

Requirements:
    pip install ibm-watson-machine-learning

To run this pipeline:
    1. Create an IBM Cloud Account and provision a Watson Machine Learning instance.
    2. Obtain an API Key (from IAM) and the Service Instance Location URL.
    3. Update the credentials dictionary below.
    4. Run: python watson_pipeline.py
"""

import os
import pickle
import numpy as np

def deploy_to_watson():
    # 1. Credentials Configuration
    # In production, these should be loaded from environment variables
    wml_credentials = {
        "url": os.getenv("WML_URL", "https://us-south.ml.cloud.ibm.com"),
        "apikey": os.getenv("WML_APIKEY", "YOUR_IBM_CLOUD_API_KEY")
    }
    
    SPACE_ID = os.getenv("WML_SPACE_ID", "YOUR_WML_DEPLOYMENT_SPACE_ID")
    
    print("Initiating IBM Watson Machine Learning pipeline...")
    
    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        print("\n[INFO] 'ibm-watson-machine-learning' package not installed.")
        print("To install, run: pip install ibm-watson-machine-learning")
        print("Simulating WML deployment pipeline workflow steps...\n")
        simulate_wml_workflow()
        return

    # Initialize client
    client = APIClient(wml_credentials)
    
    # Set default deployment space
    client.set.default_space(SPACE_ID)
    print(f"Connected to WML space: {SPACE_ID}")
    
    # 2. Load the best-performing model
    model_payload_path = "best_model.pkl"
    if not os.path.exists(model_payload_path):
        print(f"Error: {model_payload_path} not found. Please run train_models.py first.")
        return
        
    with open(model_payload_path, 'rb') as f:
        payload = pickle.load(f)
        
    model = payload['model']
    feature_cols = payload['feature_cols']
    print(f"Loaded {payload['best_model_name']} model for deployment.")
    
    # 3. Store model in IBM Watson Machine Learning Repository
    print("Storing model in Watson Machine Learning repository...")
    
    # Specify software specifications and model metadata
    # Use scikit-learn or xgboost software specifications depending on the model trained
    sw_spec_name = "runtime-23.1-py3.10" # Default python runtime in Watson ML
    sw_spec_id = client.software_specifications.get_uid_by_name(sw_spec_name)
    
    model_meta = {
        client.repository.ModelMetaNames.NAME: "Credit_Card_Approval_Classifier",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1", # adjustment based on version
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
        client.repository.ModelMetaNames.DESCRIPTION: f"Best credit card classifier model: {payload['best_model_name']}"
    }
    
    stored_model_details = client.repository.store_model(
        model=model,
        meta_props=model_meta,
        training_data=None, # Optional dataframe can go here
        feature_names=feature_cols
    )
    
    model_uid = client.repository.get_model_id(stored_model_details)
    print(f"Model stored successfully. Model UID: {model_uid}")
    
    # 4. Create an Online Deployment (for real-time API scoring)
    print("Creating real-time online deployment endpoint on IBM Cloud...")
    deployment_meta = {
        client.deployments.ConfigurationMetaNames.NAME: "Credit_Card_Approval_Online_API",
        client.deployments.ConfigurationMetaNames.DESCRIPTION: "Online deployment of Credit Card Approval prediction system.",
        client.deployments.ConfigurationMetaNames.ONLINE: {}
    }
    
    deployment_details = client.deployments.create(
        artifact_id=model_uid,
        meta_props=deployment_meta
    )
    
    deployment_uid = client.deployments.get_uid(deployment_details)
    scoring_endpoint = client.deployments.get_scoring_href(deployment_details)
    
    print("\n" + "="*50)
    print("SUCCESS: MODEL DEPLOYED TO IBM CLOUD")
    print(f"Deployment UID: {deployment_uid}")
    print(f"Scoring End-point URL: {scoring_endpoint}")
    print("="*50 + "\n")
    
    # 5. Testing the Scoring Endpoint with a sample payload
    print("Testing scoring endpoint with sample applicant profile...")
    sample_payload = {
        client.deployments.ScoringMetaNames.INPUT_DATA: [{
            "fields": feature_cols,
            # Sample applicant: Female, age 32, owns property, 0 kids, $65k income, Working, Higher Ed, etc.
            "values": [[1, 32.0, 0, 1, 0, 65000.0, 0, 1, 4.5, 1, 0, 0.15, 0]]
        }]
    }
    
    predictions = client.deployments.score(deployment_uid, sample_payload)
    print("Scoring Result output:")
    print(predictions)

def simulate_wml_workflow():
    """Simulates the steps for WML deployment log trace."""
    print("[STEP 1] Initializing WML API client using credentials...")
    print("  -> Authenticating with apikey (APIKEY-****************)... SUCCESS.")
    print("  -> Target space space-uuid-9f20c4-a128-48b2... CONNECTED.")
    print("\n[STEP 2] Loading model artifact: 'best_model.pkl'...")
    print("  -> Model detected: Random Forest Classifier")
    print("  -> Input Features: 13 dimensional numerical & encoded categories")
    print("\n[STEP 3] Registering model to IBM Watson Repository...")
    print("  -> Exporting pickle bytes and formatting metadata...")
    print("  -> Using software specification: 'runtime-23.1-py3.10'")
    print("  -> Storing model in Watson ML service registry...")
    print("  -> Saved model reference: Model-UID-cc78d10b-49ea-4cfb-b78c-02cf30a91f5a")
    print("\n[STEP 4] Deploying model for online real-time inference...")
    print("  -> Provisioning scoring container inside Watson runtime...")
    print("  -> Generating REST API endpoint routing URL...")
    print("  -> Deployment State: ACTIVE")
    print("\n" + "="*60)
    print("SIMULATION SUCCESS: MODEL DEPLOYED TO IBM WATSON ML CLOUD")
    print("Deployment UID: deploy-9a28b10f-2cba-4e92-bc12-d85c1bf0ac24")
    print("Scoring REST URL: https://us-south.ml.cloud.ibm.com/ml/v4/deployments/deploy-9a28b10f-2cba-4e92-bc12-d85c1bf0ac24/score?version=2024-03-01")
    print("="*60 + "\n")
    print("[STEP 5] Testing REST API scoring endpoint with a mock HTTP POST payload:")
    print("  Post payload: {'input_data': [{'fields': [...], 'values': [[1, 32.0, 0, 1, 0, 65000, 0, 1, 4.5, 1, 0, 0.15, 0]]}]}")
    print("  Scoring REST Response:")
    print("  {")
    print("    'predictions': [{")
    print("      'fields': ['prediction', 'probability'],")
    print("      'values': [[1, [0.12, 0.88]]]")
    print("    }]")
    print("  } -> Prediction: Approved (88.0% probability)")

if __name__ == '__main__':
    deploy_to_watson()
