# 🧠 Employee Attrition Prediction

A Machine Learning web app built with **Python, Streamlit, and scikit-learn** that predicts whether an employee is likely to leave or stay based on key HR factors.  
This project uses a Random Forest model trained on IBM’s HR Analytics dataset, customized for the Indian market.

---

## 🚀 Features
- Predicts employee attrition probability with detailed insights  
- Interactive **Streamlit dashboard**  
- Realistic Indian salary scaling  
- Displays **feature importance** using Random Forest  

---

## 🧩 Tech Stack
- **Python**
- **Pandas, NumPy, Scikit-learn**
- **Matplotlib, Seaborn**
- **Streamlit**

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
