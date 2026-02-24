# 🏦 FRAUD_SHIELD / AUDIT_PRO  
### AI-Based Transaction Risk Detection System  

🌐 **Live Application:**  
👉 https://frauddetectorpy-hx8n54x28xuyzes4jrei9b.streamlit.app/

---

## 📌 About This Project

FRAUD_SHIELD is a machine learning based fraud detection system that analyzes transaction details and predicts whether a transaction is likely to be fraudulent or legitimate.

The goal of this project was not just to build a model, but to simulate a real-world fintech risk monitoring dashboard with a clean, professional UI and structured ML pipeline.

It combines backend ML engineering with frontend user experience design.

---

## 🧠 How It Works

The system follows a proper ML workflow:

1. Data cleaning and preprocessing  
2. Handling missing values  
3. Encoding categorical features  
4. Scaling numerical features  
5. Handling class imbalance using SMOTE  
6. Training a Random Forest classifier  
7. Saving the full pipeline for real-time inference  

The complete preprocessing + model logic is bundled inside a single pipeline using `ImbPipeline`, ensuring consistent transformations during prediction.

---

## ⚙ Model Details

- Algorithm: Random Forest Classifier  
- Class imbalance handled using SMOTE  
- Stratified train-test split  
- Probability-based risk scoring  
- Threshold-based fraud detection  

The model outputs a fraud probability score, which is visualized in the dashboard as a risk index.

---

## 🎨 User Interface

The UI is designed with a fintech-inspired dark theme.

Features include:
- Smooth fade-in animation  
- Hover glow effects  
- Risk gauge visualization  
- Animated progress loader  
- Pulse alert for fraud detection  
- Clean dashboard layout  

The idea was to make it feel like a real enterprise security monitoring tool rather than a simple ML demo.

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Pandas / NumPy  
- Joblib  

---

## 📂 Project Structure

```
FraudSheild-AI
│
├── fraud_detector.py
├── fraud_model.pkl
├── requirements.txt
└── README.md
```

---

## ▶ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run fraud_detector.py
```

---




<img width="1469" height="819" alt="Screenshot 2026-02-24 at 4 14 53 PM" src="https://github.com/user-attachments/assets/e35eb606-0b4f-45fc-a54a-b7bc50625c9b" />

## 👨‍💻 Author
Prabhat Kumar

Mathematics & Computing  
IIT Kharagpur  

---

If you found this project interesting, feel free to ⭐ the repository.
