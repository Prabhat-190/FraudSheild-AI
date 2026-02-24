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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        
        .stApp {
            background: #050505;
            color: #e0e0e0;
            animation: fadeIn 1.5s ease-in;
        }

        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }

        div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
            background: #0f1115;
            padding: 2.5rem;
            border-radius: 20px;
            border: 1px solid #2d2d2d;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        div[data-testid="stVerticalBlock"] > div:has(div.stForm):hover {
            border-color: #00ffcc;
            box-shadow: 0 10px 30px rgba(0, 255, 204, 0.1);
            transform: scale(1.01);
        }

        .stButton>button {
            background: linear-gradient(90deg, #00ffcc, #00d2ff);
            color: #000000 !important;
            border-radius: 12px;
            font-weight: 700;
            border: none;
            padding: 0.75rem 2rem;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 255, 204, 0.4);
            filter: brightness(1.1);
        }

        .pulse-red {
            animation: pulse-red 2s infinite;
        }
        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(255, 75, 75, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
        }
        </style>
        """, unsafe_allow_html=True)

    if not os.path.exists(MODEL_FILE):
        with st.spinner("🧠 SYNCING NEURAL NETWORK..."):
            train_model()

    model = joblib.load(MODEL_FILE)

    st.title("🏦 FRAUD_SHIELD / AUDIT_PRO")
    st.markdown("<p style='opacity:0.6; letter-spacing:3px;'>ADVANCED RISK MITIGATION ENGINE</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([0.45, 0.55], gap="large")

    with col1:
        with st.form("audit_form"):
            st.markdown("### 📥 INPUT_STREAM")
            amt = st.number_input("TXN_AMOUNT (USD)", min_value=0.0, step=10.0, value=250.00)
            t_val = st.number_input("TIMESTAMP_SEC", value=86400)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 👤 SUBJECT_PROFILE")
            age_val = st.slider("HOLDER_AGE", 18, 95, 30)
            
            loc = st.selectbox("GEO_LOCATION", sorted(["California", "New York", "London", "Online", "Tokyo", "Berlin", "Paris"]))
            cat = st.selectbox("MERCHANT_CLASS", sorted(["Food", "Retail", "Electronics", "Crypto", "Entertainment", "Travel"]))
            
            st.markdown("<br>", unsafe_allow_html=True)
            analyze = st.form_submit_button("RUN NEURAL AUDIT")

    with col2:
        st.markdown("### 📊 ANALYSIS_REPORT")
        if analyze:
            input_data = pd.DataFrame([[amt, t_val, age_val, loc, cat]], 
                                     columns=['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory'])
            
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            prob = model.predict_proba(input_data)[0][1]
            is_fraud = prob > 0.5

            if is_fraud:
                st.markdown(f"""
                    <div style="background: rgba(255, 75, 75, 0.1); padding: 30px; border-radius: 15px; border: 1px solid #ff4b4b;" class="pulse-red">
                        <h1 style="color: #ff4b4b; margin:0;">🚨 FRAUD_DETECTED</h1>
                        <p style="font-size: 20px; color: #fff; margin-top: 10px;">RISK_PROBABILITY: <b>{(prob*100):.1f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                st.error("THREAT_LEVEL: CRITICAL / ACTION: REJECT_TRANSACTION")
            else:
                st.markdown(f"""
                    <div style="background: rgba(0, 255, 204, 0.1); padding: 30px; border-radius: 15px; border: 1px solid #00ffcc;">
                        <h1 style="color: #00ffcc; margin:0;">✅ TRANSACTION_SECURE</h1>
                        <p style="font-size: 20px; color: #fff; margin-top: 10px;">RISK_PROBABILITY: <b>{(prob*100):.1f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                st.success("THREAT_LEVEL: LOW / ACTION: AUTHORIZE_TXN")

            st.markdown("### RISK_INDEX_GAUGE")
            st.progress(prob)
        else:
            st.markdown("""
                <div style="border: 2px dashed #2d2d2d; padding: 120px; text-align: center; border-radius: 20px; opacity: 0.5;">
                    <p style="letter-spacing: 5px;">AWAITING_SIGNAL_INPUT</p>
                </div>
            """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("System Version: 2.1.0-Stable")
    st.sidebar.caption("Last Sync: " + time.strftime("%H:%M:%S"))

if __name__ == "__main__":
    main()
