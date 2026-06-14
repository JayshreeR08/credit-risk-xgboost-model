# credit-risk-xgboost-model
Credit risk prediction system using XGBoost to classify loan applicants as high or low risk based on financial features

# Credit Risk Prediction using XGBoost

## 📌 Overview
This project is a machine learning model that predicts credit risk (loan approval risk) based on applicant financial and personal data.  
The model is trained using XGBoost and optimized for classification performance.

---

## 🚀 Problem Statement
Financial institutions need to assess whether a loan applicant is high-risk or low-risk.  
This model helps automate that decision using historical data patterns.

---

## 🧠 Model Details
- Algorithm: XGBoost Classifier
- Type: Binary Classification
- Output: Risk category (0 = Low Risk, 1 = High Risk)
- Preprocessing:
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling 
  - Class imbalance handling

---

## 📊 Dataset
- Source: Kaggle dataset https://www.kaggle.com/datasets/hemanthsai7/loandefault
- Features include:
  - Income
  - Credit history
  - Loan amount
  - Employment details
  - Other financial indicators

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Jupyter Notebook (Kaggle)

---

## 🔗 Kaggle Notebook
Training notebook can be found here:  
👉https://www.kaggle.com/code/jayshreerathore/credit-risk-prediction-system-using-xgboost

---

## ⚙️ How to Use the Model

```python
import pickle

model = pickle.load(open("loan_risk_model.pkl", "rb"))

# Example prediction
prediction = model.predict([[/* input features */]])
print(prediction)
