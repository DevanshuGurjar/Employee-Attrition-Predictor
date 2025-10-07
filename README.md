# 🧠 Employee Attrition Predictor  
### Predict Employee Turnover Using Machine Learning  

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/ML-sklearn-F7931E.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

### 🚀 Overview
The **Employee Attrition Predictor** is a machine learning web app built with **Streamlit** that predicts the likelihood of employee attrition based on key HR features like job role, salary, satisfaction, overtime, and more.  
It helps organizations **analyze workforce trends** and take **proactive retention measures**.

---

### 🧩 Key Features
- 🔍 Predicts employee attrition probability in real-time  
- 📊 Visualizes employee demographics and salary insights  
- 🧠 Machine learning backend using Random Forest Classifier  
- 🧹 Modularized code with separate preprocessing and model files  
- 🇮🇳 Tuned dataset to reflect Indian salary structure for realism  

---

### 🧮 Tech Stack
- **Frontend:** Streamlit  
- **Backend:** Python (Scikit-learn, Pandas, NumPy)  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment-ready:** Works locally or via cloud (Streamlit Cloud / Render)  

---

## 📊 Dataset
The dataset used is based on IBM’s HR Analytics dataset, modified to include:
- Realistic monthly incomes (scaled ×10 for Indian salaries)
- Additional work mode correlations

---

## 🖼 Screenshots

### 📊 Prediction Result
![Prediction](./assets/image1.jpeg)

### 🔍 Feature Importance
![Feature Importance](./assets/image2.jpeg)

---

## 🧠 Model Insights
- Top influencing factors:
  - Monthly Income
  - Age
  - Years at Company
  - Distance from Home
  - Percent Salary Hike

---

## 🚀 Future Work
- Integrate SHAP explainability directly into the Streamlit app for better model interpretability.
- Add employee sentiment analysis using textual feedback data.
- Deploy the model on Streamlit Cloud or AWS for public access.
- Build an HR dashboard to monitor attrition trends in real time.
- Experiment with deep learning models like ANN or XGBoost for improved accuracy.

---

## ⚠️ Limitations
- The current dataset is limited and may not fully represent diverse industries or employee demographics.
- Model predictions depend on HR data quality — missing or biased data can affect accuracy.
- The salary distribution was scaled to better fit the Indian market and may not reflect global data patterns.
- Model retraining is manual; automation can be added for periodic updates.

---

## 💻 Run Locally
```bash
git clone https://github.com/DevanshuGurjar/Employee-Attrition-Predictor.git
cd Employee-Attrition-Predictor
pip install -r requirements.txt
streamlit run app.py
