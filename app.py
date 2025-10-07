# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Model, Encoders, Scaler
# -------------------------------
model = joblib.load("models/attrition_model.pkl")
le_dict = joblib.load("models/label_encoders.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("🏢 Employee Attrition Predictor")
st.write("""
This app predicts the probability of an employee leaving the company and provides **AI-driven retention suggestions**.
""")

# -------------------------------
# Input Fields
# -------------------------------
st.sidebar.header("Employee Details")

def user_input_features():
    Age = st.sidebar.slider("Age", 18, 60, 30)
    Department = st.sidebar.selectbox("Department", le_dict['Department'].classes_)
    JobRole = st.sidebar.selectbox("Job Role", le_dict['JobRole'].classes_)
    MonthlyIncome = st.sidebar.number_input("Monthly Income", 1000, 200000, 5000)
    PercentSalaryHike = st.sidebar.slider("Percent Salary Hike", 0, 50, 15)
    JobSatisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
    WorkLifeBalance = st.sidebar.slider("Work Life Balance (1-4)", 1, 4, 3)
    YearsAtCompany = st.sidebar.slider("Years at Company", 0, 40, 5)
    OverTime = st.sidebar.selectbox("OverTime", le_dict['OverTime'].classes_)
    DistanceFromHome = st.sidebar.slider("Distance From Home (km)", 0, 60, 10)
    WorkMode = st.sidebar.selectbox("Work Mode", le_dict['WorkMode'].classes_)

    data = {
        'Age': Age,
        'Department': Department,
        'JobRole': JobRole,
        'MonthlyIncome': MonthlyIncome,
        'PercentSalaryHike': PercentSalaryHike,
        'JobSatisfaction': JobSatisfaction,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'OverTime': OverTime,
        'DistanceFromHome': DistanceFromHome,
        'WorkMode': WorkMode
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -------------------------------
# Preprocess Input
# -------------------------------
def preprocess_input(df):
    # Encode categorical features
    cat_cols = ['Department', 'JobRole', 'OverTime', 'WorkMode']
    for col in cat_cols:
        df[col] = le_dict[col].transform(df[col])

    # Scale numeric columns
    num_cols = ['Age', 'MonthlyIncome', 'PercentSalaryHike', 'YearsAtCompany', 'DistanceFromHome']
    df[num_cols] = scaler.transform(df[num_cols])

    return df

processed_input = preprocess_input(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Attrition"):
    pred_prob = model.predict_proba(processed_input)[0][1]  # Probability of leaving
    pred_class = model.predict(processed_input)[0]

    if pred_class == 1:
        st.error(f"⚠️ This employee is likely to **leave** with probability: {pred_prob*100:.2f}%")
    else:
        st.success(f"✅ This employee is likely to **stay** with probability: {(1-pred_prob)*100:.2f}%")

    # -------------------------------
    # AI-driven Retention Suggestions
    # -------------------------------
    st.subheader("💡 Retention Suggestions")

    suggestions = []

    if pred_class == 1:
        if input_df['OverTime'][0] == 'Yes':
            suggestions.append("Reduce overtime workload or offer flexible hours.")
        if input_df['WorkLifeBalance'][0] < 3:
            suggestions.append("Improve work-life balance via remote/hybrid options or wellness programs.")
        if input_df['JobSatisfaction'][0] < 3:
            suggestions.append("Engage with the employee to understand dissatisfaction and provide incentives.")
        if input_df['YearsAtCompany'][0] < 2:
            suggestions.append("Provide mentorship and onboarding support for newer employees.")
    else:
        suggestions.append("Employee is stable. Continue monitoring and engagement.")

    for s in suggestions:
        st.write(f"- {s}")

# -------------------------------
# Feature Importance Display
# -------------------------------
st.subheader("📊 Feature Importance (Random Forest)")

import matplotlib.pyplot as plt

feat_importances = pd.Series(model.feature_importances_, index=processed_input.columns)
feat_importances.sort_values().plot(kind='barh', figsize=(8,5), color='skyblue')
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
st.pyplot(plt.gcf())
