import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import pickle
import json

DATABASE_URI = 'sqlite:///churn_db.sqlite'
CUSTOMER_LTV = 1000  # Expected Lifetime Value of a retained customer ($)
RETENTION_COST = 100 # Cost to run a retention campaign per customer ($)

def train_model():
    print("Loading data from Netflix dataset...")
    # The file has a .xls extension but is formatted as a CSV
    df = pd.read_csv('netflix_customer_churn.xls')
    
    # Feature Engineering & Preprocessing
    print("Preprocessing data...")
    # Drop irrelevant columns for modeling
    X = df.drop(['customer_id', 'churned'], axis=1)
    y = df['churned']
    
    # Categorical encoding for all categorical columns
    categorical_cols = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']
    encoder_mapping = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        # Save mapping for this column
        encoder_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    
    # Save the encoder mapping for the API
    with open('encoder_mapping.json', 'w') as f:
        json.dump(encoder_mapping, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost Model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Quantify Churn Reduction Impact
    print("\nQuantifying Business Impact (Test Set):")
    # Assume we target the top 20% most likely to churn
    threshold = np.percentile(y_prob, 80)
    targeted_customers = (y_prob >= threshold)
    
    true_positives = (targeted_customers & (y_test == 1)).sum()
    false_positives = (targeted_customers & (y_test == 0)).sum()
    
    # If campaign is 30% effective at saving true churners
    saved_customers = true_positives * 0.3
    revenue_saved = saved_customers * CUSTOMER_LTV
    cost_of_campaign = targeted_customers.sum() * RETENTION_COST
    net_profit = revenue_saved - cost_of_campaign
    
    print(f"Targeted Customers: {targeted_customers.sum()}")
    print(f"Estimated Customers Saved: {saved_customers:.1f}")
    print(f"Gross Revenue Saved: ${revenue_saved:.2f}")
    print(f"Campaign Cost: ${cost_of_campaign:.2f}")
    print(f"Net Profit of ML-driven Campaign: ${net_profit:.2f}")
    
    # SHAP Explainability
    print("\nGenerating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Save model and explainer
    print("Saving model and explainer artifacts...")
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
        
    # Also save feature names to ensure API has correct order
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
        
    print("Training pipeline complete!")

if __name__ == '__main__':
    train_model()
