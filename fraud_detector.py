import pandas as pd
import numpy as np
import joblib
import os
import time
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- CONFIGURATION & MODEL PATH ---
MODEL_FILE = "fraud_model.pkl"
DATA_FILE = "credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv"

# --- BACKEND: TRAINING FUNCTION ---
def train_model():
    """Trains the model and saves it to a pkl file."""
    if not os.path.exists(DATA_FILE):
        return False, "Dataset CSV not found. Please ensure it's in the directory."
    
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=['TransactionID'], errors='ignore')
    
    categorical_features = ['Location', 'MerchantCategory']
    numerical_features = ['Amount', 'Time', 'CardHolderAge']

    # Impute missing values
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X = df.drop('IsFraud', axis=1)
    y = df['IsFraud']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Best performing model from your analysis
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])

    rf_pipeline.fit(X_train, y_train)
    joblib.dump(rf_pipeline, MODEL_FILE)
    return True, "Model trained successfully!"

# --- FRONTEND: STREAMLIT UI ---
def main():
    st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

    # Custom CSS for a professional look
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
        .result-card { padding: 20px; border-radius: 10px; border: 1px solid #ddd; background-color: white; }
        </style>
        """, unsafe_allow_html=True)

    st.title("🛡️ FraudShield: Enterprise Detection System")
    st.markdown("Automated Real-Time Transaction Risk Assessment")

    # Ensure model exists
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Initializing AI Engine for the first time..."):
            success, msg = train_model()
            if not success:
                st.error(msg)
                return
            st.success(msg)

    # Load Model
    model = joblib.load(MODEL_FILE)

    # Layout
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Transaction Parameters")
        with st.form("prediction_form"):
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=250.0)
            txn_time = st.number_input("Time (Seconds)", min_value=0, value=86400)
            age = st.slider("Cardholder Age", 18, 100, 45)
            
            # Use columns for selectboxes to keep it compact
            loc_list = ["New York", "London", "California", "Online", "Chicago", "Paris", "Tokyo"]
            cat_list = ["Retail", "Entertainment", "Food", "Travel", "Electronics", "Groceries"]
            
            location = st.selectbox("Transaction Location", sorted(loc_list))
            category = st.selectbox("Merchant Category", sorted(cat_list))
            
            submit = st.form_submit_button("Run Risk Analysis")

    with col2:
        st.subheader("Analysis Report")
        if submit:
            # Prepare Input Data
            input_df = pd.DataFrame({
                'Amount': [amount],
                'Time': [txn_time],
                'CardHolderAge': [age],
                'Location': [location],
                'MerchantCategory': [category]
            })

            # Inference
            with st.spinner("Analyzing patterns..."):
                time.sleep(0.8) # Realism delay
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

            # Results Display
            if prediction == 1:
                st.error("### 🚨 HIGH RISK DETECTED")
                st.write(f"This transaction shows a **{(probability*100):.1f}%** correlation with known fraudulent patterns.")
                st.progress(probability)
                st.warning("**Action Required:** Block transaction and notify cardholder.")
            else:
                st.success("### ✅ TRANSACTION SECURE")
                st.write(f"Security score is high. Fraud probability is only **{(probability*100):.1f}%**.")
                st.progress(probability)
                st.info("**Action:** Proceed with processing.")

        else:
            st.info("Enter transaction data and click 'Run Risk Analysis' to view the security report.")

    # Model Performance Stats in Sidebar
    st.sidebar.title("Model Metrics")
    st.sidebar.metric("Algorithm", "Random Forest")
    st.sidebar.metric("Sampling Method", "SMOTE (Oversampling)")
    st.sidebar.markdown("---")
    st.sidebar.caption("System Version: 2.0.4-Stable")

if __name__ == "__main__":
    main()
