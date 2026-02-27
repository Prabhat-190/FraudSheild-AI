import pandas as pd
import numpy as np
import joblib
import os
import time
import streamlit as st
import ccxt
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
    st.set_page_config(page_title="CRYPTO_SHIELD_PRO", page_icon="₿", layout="wide")

    st.markdown("""
        <style>
        .stApp { background: #050505; color: #e0e0e0; }
        .live-tag { color: #00ffcc; font-weight: bold; animation: blinker 1.2s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }
        div[data-testid="stMetricValue"] { color: #00ffcc !important; }
        </style>
        """, unsafe_allow_html=True)

    if not os.path.exists(MODEL_FILE): train_model()
    model = joblib.load(MODEL_FILE)

    st.title("₿ CRYPTO_SHIELD / LIVE_EXCHANGE")
    
    with st.sidebar:
        st.header("CONNECTION_SETTINGS")
        exchange_id = st.selectbox("EXCHANGE", ["binance", "coinbase", "kraken"])
        crypto_mode = st.toggle("ACTIVATE LIVE CRYPTO FEED", value=False)
        st.divider()
        st.caption("Using CCXT Unified API Library")

    if crypto_mode:
        st.markdown('<p class="live-tag">● CONNECTED_TO_EXCHANGE_STREAM</p>', unsafe_allow_html=True)
        
        # Initialize Exchange
        client = getattr(ccxt, exchange_id)()
        placeholder = st.empty()

        while crypto_mode:
            try:
                # 1. Ingest: Fetch live ticker data
                ticker = client.fetch_ticker('BTC/USDT')
                
                # 2. Transform: Map crypto data to your ML features
                live_txn = {
                    'Amount': ticker['last'],
                    'Time': int(time.time() % 86400),
                    'CardHolderAge': 30,
                    'Location': 'Online',
                    'MerchantCategory': 'Crypto'
                }
                input_df = pd.DataFrame([live_txn])

                # 3. Inference: Run through saved pipeline
                prob = model.predict_proba(input_df)[0][1]
                
                with placeholder.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("BTC_PRICE", f"${ticker['last']:,}")
                    c2.metric("VOL_24H", f"{ticker['quoteVolume']:.2f}")
                    c3.metric("RISK_INDEX", f"{prob*100:.2f}%")

                    if prob > 0.5:
                        st.error(f"🚨 ANOMALY: High Risk Pattern detected in Trade Stream")
                    else:
                        st.success("✅ TRANSACTION_FLOW: SECURE")
                    
                    st.dataframe(input_df, use_container_width=True)
                
                time.sleep(2) # Stream interval
            except Exception as e:
                st.error(f"Connection Lost: {e}")
                break
    else:
        st.info("Select an exchange and toggle 'ACTIVATE LIVE FEED' to start the neural audit.")

if __name__ == "__main__":
    main()
