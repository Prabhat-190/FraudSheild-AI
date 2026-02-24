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

MODEL_FILE = "fraud_model.pkl"
DATA_FILE = "credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv"

def train_model():
    if not os.path.exists(DATA_FILE): return False, "Data Source Missing"
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=['TransactionID'], errors='ignore')
    cat_cols = ['Location', 'MerchantCategory']
    num_cols = ['Amount', 'Time', 'CardHolderAge']
    for col in num_cols: df[col] = df[col].fillna(df[col].median())
    for col in cat_cols: df[col] = df[col].fillna(df[col].mode()[0])
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    X = df.drop('IsFraud', axis=1)
    y = df['IsFraud']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_FILE)
    return True, "Core Initialized"

def main():
    st.set_page_config(page_title="FRAUD_SHIELD_V2", page_icon="🏦", layout="wide")

    st.markdown("""
        <style>
        .stApp { background: #050505; color: #e0e0e0; }
        [data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #1f1f1f; }
        div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
            background: #0f1115;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #2d2d2d;
        }
        .stButton>button {
            background: #ffffff;
            color: #000000;
            border-radius: 4px;
            font-weight: 700;
            letter-spacing: 1px;
            height: 45px;
            border: none;
        }
        .stButton>button:hover { background: #00ffcc; color: #000000; border: none; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -1px; }
        .report-box { padding: 24px; border-radius: 8px; margin-bottom: 20px; }
        .stProgress > div > div > div > div { background-color: #00ffcc; }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.caption("OPERATIONAL STATUS")
        st.title("CORE_V2")
        st.status("ENCRYPTED", state="complete")
        st.divider()
        if st.button("RE-SYNC DATABASE"):
            train_model()
            st.rerun()

    st.title("🏦 FRAUD_SHIELD / ANALYST_PRO")
    st.caption("AI-POWERED TRANSACTION AUDIT SYSTEM")
    
    if not os.path.exists(MODEL_FILE):
        with st.spinner("BUILDING NEURAL PIPELINE..."):
            train_model()

    model = joblib.load(MODEL_FILE)

    col1, col2 = st.columns([0.4, 0.6], gap="large")

    with col1:
        with st.form("audit_form"):
            st.subheader("TXN_METRICS")
            amt = st.number_input("TOTAL_AMOUNT (USD)", min_value=0.0, step=0.01, value=1250.00)
            
            st.divider()
            st.subheader("ENTITY_PROFILE")
            age_val = st.slider("ACCOUNT_HOLDER_AGE", 18, 95, 34)
            cat = st.selectbox("MERCHANT_CLASS", sorted(["Retail", "Electronics", "Crypto_Exchange", "Travel", "Gambling", "Groceries", "Entertainment"]))
            loc = st.selectbox("GEOGRAPHICAL_ZONE", sorted(["North America", "European Union", "Asia-Pacific", "LATAM", "Middle East", "Offshore"]))
            
            st.divider()
            st.subheader("NETWORK_DATA")
            t_val = st.number_input("TIMESTAMP_SEQUENCE (SEC)", value=86400)
            
            analyze = st.form_submit_button("EXECUTE AUDIT")

    with col2:
        if analyze:
            input_data = pd.DataFrame([[amt, t_val, age_val, loc, cat]], 
                                     columns=['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory'])
            
            with st.spinner("CALCULATING RISK VECTORS..."):
                time.sleep(0.6)
                prob = model.predict_proba(input_data)[0][1]
                is_fraud = prob > 0.5

            if is_fraud:
                st.markdown(f"""
                    <div class="report-box" style="border: 1px solid #ff4b4b; background: rgba(255, 75, 75, 0.05);">
                        <h1 style="color: #ff4b4b; margin:0;">CRITICAL_THREAT</h1>
                        <p style="font-size: 24px;">RISK_INDEX: {(prob*100):.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.error("ACTION: TRANSACTION_HALT_REQUIRED")
            else:
                st.markdown(f"""
                    <div class="report-box" style="border: 1px solid #00ffcc; background: rgba(0, 255, 204, 0.05);">
                        <h1 style="color: #00ffcc; margin:0;">VALIDATED</h1>
                        <p style="font-size: 24px;">RISK_INDEX: {(prob*100):.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.success("ACTION: PROCEED_TO_CLEARANCE")

            st.subheader("RISK_CONFIDENCE_INTERVAL")
            st.progress(prob)
            
            with st.expander("VIEW AUDIT_LOG"):
                st.json({
                    "timestamp": time.time(),
                    "input_vector": [amt, t_val, age_val, loc, cat],
                    "model_confidence": prob,
                    "status": "FLAGGED" if is_fraud else "CLEARED"
                })
        else:
            st.markdown("""
                <div style="border: 1px dashed #333; padding: 100px; text-align: center; border-radius: 12px;">
                    <p style="color: #666; letter-spacing: 2px;">AWAITING_INPUT_SIGNAL</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
