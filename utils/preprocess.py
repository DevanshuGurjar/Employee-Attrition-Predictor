
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(csv_path):
    # Load dataset
    data = pd.read_csv(csv_path)

    # Select relevant features
    features = [
        'Age', 'Department', 'JobRole', 'MonthlyIncome', 'PercentSalaryHike',
        'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany',
        'OverTime', 'DistanceFromHome'
    ]
    data = data[features + ['Attrition']]

    # Encode target variable first (needed for WorkMode correlation)
    data['AttritionFlag'] = data['Attrition'].map({'Yes': 1, 'No': 0})

    # Create correlated 'WorkMode' feature
    np.random.seed(42)
    def assign_workmode(attrition_flag):
        if attrition_flag == 1:  # Employee left
            return np.random.choice(['On-site', 'Hybrid', 'Remote'], p=[0.6, 0.3, 0.1])
        else:  # Employee stayed
            return np.random.choice(['On-site', 'Hybrid', 'Remote'], p=[0.2, 0.4, 0.4])

    data['WorkMode'] = data['AttritionFlag'].apply(assign_workmode)

    # Drop helper column (we keep original 'Attrition' for clarity)
    data.drop('AttritionFlag', axis=1, inplace=True)

    # Encode categorical features
    le_dict = {}  # Store LabelEncoders to use later for Streamlit
    categorical_cols = ['Department', 'JobRole', 'OverTime', 'WorkMode']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le

    # Encode target variable
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

    # Split features and target
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']

    # Scale numeric features
    numeric_cols = ['Age', 'MonthlyIncome', 'PercentSalaryHike', 'YearsAtCompany', 'DistanceFromHome']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, le_dict, scaler