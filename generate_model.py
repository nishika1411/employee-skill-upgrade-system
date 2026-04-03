"""
Script to generate employee_attrition_test.csv and salary_model.pkl
Run this once before deploying or if files are missing.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

np.random.seed(42)
n = 1000

departments = ['Sales', 'Research & Development', 'Human Resources']
job_roles = [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative',
    'Manager', 'Human Resources', 'Research Director'
]

dept_salary_base = {
    'Sales': 5000,
    'Research & Development': 7000,
    'Human Resources': 4500
}
role_salary_base = {
    'Sales Executive': 5200,
    'Research Scientist': 7500,
    'Laboratory Technician': 4800,
    'Manufacturing Director': 9000,
    'Healthcare Representative': 6000,
    'Manager': 10000,
    'Human Resources': 4500,
    'Research Director': 12000
}

dept_col = np.random.choice(departments, n)
role_col = np.random.choice(job_roles, n)
age_col = np.random.randint(22, 60, n)
job_level_col = np.random.randint(1, 6, n)
satisfaction_col = np.random.randint(1, 5, n)
performance_col = np.random.randint(1, 5, n)
training_col = np.random.randint(0, 7, n)
worklife_col = np.random.randint(1, 5, n)
experience_col = np.clip(age_col - 22 + np.random.randint(-3, 5, n), 0, 40)
years_company_col = np.clip(experience_col - np.random.randint(0, 5, n), 0, experience_col)

# Salary based on role + level + experience + noise
monthly_income = np.array([
    role_salary_base[r] + job_level_col[i] * 1200
    + experience_col[i] * 80
    + performance_col[i] * 200
    + np.random.randint(-500, 500)
    for i, r in enumerate(role_col)
]).clip(2000, 20000)

df = pd.DataFrame({
    'Age': age_col,
    'Department': dept_col,
    'JobRole': role_col,
    'JobLevel': job_level_col,
    'JobSatisfaction': satisfaction_col,
    'PerformanceRating': performance_col,
    'TrainingTimesLastYear': training_col,
    'WorkLifeBalance': worklife_col,
    'TotalWorkingYears': experience_col,
    'YearsAtCompany': years_company_col,
    'MonthlyIncome': monthly_income.astype(int)
})

csv_path = 'employee_attrition_test.csv'
df.to_csv(csv_path, index=False)
print(f"[OK] Saved {csv_path} with {len(df)} rows")

# Train model
df_ml = df.copy()
le_dict = {}
for col in df_ml.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    le_dict[col] = le

X = df_ml.drop('MonthlyIncome', axis=1)
y = df_ml['MonthlyIncome']

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

with open('salary_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("[OK] Saved salary_model.pkl")
print("[OK] Done! You can now run: streamlit run app.py")
