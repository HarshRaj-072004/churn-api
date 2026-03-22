# Customer Churn Prediction System

##  Overview

This project builds an end-to-end **Customer Churn Prediction System** using Machine Learning.
It predicts whether a customer will leave a bank based on historical data and provides **actionable insights** to reduce churn.

---

##  Key Features

* 🔍 Predict customer churn using ML models
* ⚖️ Handle class imbalance using SMOTE
* 📈 Compare multiple models (Logistic Regression, Random Forest, XGBoost)
* 📊 Evaluate using Accuracy, F1-score, ROC-AUC
* 🧠 Identify key churn drivers using feature importance
* 💰 Simulate business impact (potential revenue savings)
* 🌐 Deploy using FastAPI + Streamlit

---

##  Dataset

* **Source:** Customer Churn Modelling Dataset
* **Records:** 10,000+ customers
* **Features:** Credit Score, Geography, Gender, Age, Balance, etc.
* **Target Variable:** `Exited` (1 = Churn, 0 = Stay)

---

##  Tech Stack

* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **ML Techniques:** SMOTE, Feature Importance, Model Evaluation
* **Deployment:** FastAPI, Streamlit
* **Tools:** Joblib

---

##  Model Performance

| Model               | Accuracy  | ROC-AUC  | Recall (Churn) |
| ------------------- | --------- | -------- | -------------- |
| Logistic Regression | 80.0%     | 0.76     | 39%            |
| Random Forest ✅     | **84.5%** | **0.85** | **54%**        |
| XGBoost             | 84.3%     | 0.84     | 56%            |

👉 **Random Forest selected as best model**

---

##  Key Insights

Top factors influencing churn:

* Age
* Balance
* IsActiveMember
* EstimatedSalary
* NumOfProducts

---

##  Business Impact

* Identified high-risk customers using ML predictions
* Simulated **$5.5M+ potential annual savings** through targeted retention strategies

---

##  Project Workflow

1. Data Loading & Exploration
2. Data Cleaning & Encoding
3. Train-Test Split
4. Handling Imbalance (SMOTE)
5. Model Training (LR, RF, XGBoost)
6. Model Evaluation (Accuracy, F1, ROC-AUC)
7. Feature Importance Analysis
8. Deployment (FastAPI + Streamlit)

---

##  Deployment

* Backend: FastAPI (REST API for predictions)
* Frontend: Streamlit (interactive UI)

---

##  Installation

```bash
git clone (https://github.com/HarshRaj-072004/churn-api.git)
cd churn-prediction
pip install -r requirements.txt
```

---

##  Run Locally

### Run API

```bash
uvicorn app:app --reload
```

### Run Streamlit UI

```bash
streamlit run app.py
```

---

##  Future Improvements

* Hyperparameter tuning for better performance
* Real-time data integration
* Dashboard using Power BI
* Model monitoring (MLOps)

---

##  Author

**Harsh Raj**

* LinkedIn: https://www.linkedin.com/in/harsh-raj-3537342a2
* GitHub: https://github.com/HarshRaj-072004

---
