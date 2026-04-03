import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Employee Skill Upgrade Recommendation System",
    page_icon="🚀",
    layout="wide"
)

# ── Helper: generate synthetic data if CSV missing ──────────────────────────
def generate_data():
    np.random.seed(42)
    n = 1000
    departments = ['Sales', 'Research & Development', 'Human Resources']
    job_roles = [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician',
        'Manufacturing Director', 'Healthcare Representative',
        'Manager', 'Human Resources', 'Research Director'
    ]
    role_salary_base = {
        'Sales Executive': 5200, 'Research Scientist': 7500,
        'Laboratory Technician': 4800, 'Manufacturing Director': 9000,
        'Healthcare Representative': 6000, 'Manager': 10000,
        'Human Resources': 4500, 'Research Director': 12000
    }
    dept_col  = np.random.choice(departments, n)
    role_col  = np.random.choice(job_roles, n)
    age_col   = np.random.randint(22, 60, n)
    jl_col    = np.random.randint(1, 6, n)
    sat_col   = np.random.randint(1, 5, n)
    perf_col  = np.random.randint(1, 5, n)
    train_col = np.random.randint(0, 7, n)
    wl_col    = np.random.randint(1, 5, n)
    exp_col   = np.clip(age_col - 22 + np.random.randint(-3, 5, n), 0, 40)
    yc_col    = np.clip(exp_col - np.random.randint(0, 5, n), 0, exp_col)
    income    = np.array([
        role_salary_base[r] + jl_col[i]*1200 + exp_col[i]*80
        + perf_col[i]*200 + np.random.randint(-500, 500)
        for i, r in enumerate(role_col)
    ]).clip(2000, 20000).astype(int)

    return pd.DataFrame({
        'Age': age_col, 'Department': dept_col, 'JobRole': role_col,
        'JobLevel': jl_col, 'JobSatisfaction': sat_col,
        'PerformanceRating': perf_col, 'TrainingTimesLastYear': train_col,
        'WorkLifeBalance': wl_col, 'TotalWorkingYears': exp_col,
        'YearsAtCompany': yc_col, 'MonthlyIncome': income
    })

# ── Load / generate data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = 'employee_attrition_test.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_data()
        df.to_csv(csv_path, index=False)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# ── Encode data & build label-encoder dict ───────────────────────────────────
@st.cache_resource
def load_encoders_and_model():
    df = load_data()
    le_dict = {}
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    model_path = 'salary_model.pkl'
    if os.path.exists(model_path):
        try:
            model = pickle.load(open(model_path, 'rb'))
        except Exception:
            model = None
    else:
        model = None

    if model is None:
        X = df_enc.drop('MonthlyIncome', axis=1)
        y = df_enc['MonthlyIncome']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    return df_enc, le_dict, model

df_enc, le_dict, model = load_encoders_and_model()
df_raw = load_data()

# ── Skill map ────────────────────────────────────────────────────────────────
role_skills = {
    'Sales Executive':          ['Communication', 'Negotiation', 'CRM'],
    'Research Scientist':       ['Python', 'Machine Learning', 'Statistics'],
    'Laboratory Technician':    ['Lab Skills', 'Data Analysis'],
    'Manufacturing Director':   ['Operations', 'Planning', 'Leadership'],
    'Healthcare Representative':['Medical Knowledge', 'Communication'],
    'Manager':                  ['Leadership', 'Strategy', 'People Management'],
    'Human Resources':          ['Recruitment', 'HR Management', 'Communication'],
    'Research Director':        ['Advanced ML', 'Deep Learning', 'Leadership'],
}

# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "💰 Salary Prediction", "🎯 Skill Recommendation"])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Dashboard
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Employee Skill Upgrade Recommendation System")
    st.markdown("### Dashboard – Overview of Employee Data")

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Employees", len(df_raw))
    col2.metric("💵 Avg Monthly Income", f"₹{int(df_raw['MonthlyIncome'].mean()):,}")
    col3.metric("📈 Avg Performance", f"{df_raw['PerformanceRating'].mean():.2f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_raw['MonthlyIncome'], kde=True, ax=ax, color='steelblue')
        ax.set_title("Monthly Income Distribution")
        ax.set_xlabel("Monthly Income (₹)")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        dept_avg = df_raw.groupby('Department')['MonthlyIncome'].mean().sort_values()
        sns.barplot(x=dept_avg.values, y=dept_avg.index, ax=ax2, palette='viridis')
        ax2.set_title("Avg Income by Department")
        ax2.set_xlabel("Avg Monthly Income (₹)")
        st.pyplot(fig2)
        plt.close(fig2)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Salary Prediction
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰 Salary Prediction":
    st.title("💰 Salary Prediction")
    st.markdown("Fill in employee details to predict their monthly income.")

    col1, col2 = st.columns(2)

    with col1:
        age        = st.slider("Age", 18, 60, 30)
        department = st.selectbox("Department", le_dict['Department'].classes_)
        job_role   = st.selectbox("Job Role", le_dict['JobRole'].classes_)
        job_level  = st.slider("Job Level", 1, 5, 2)
        satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)

    with col2:
        performance   = st.slider("Performance Rating (1–4)", 1, 4, 3)
        training      = st.slider("Training Times Last Year", 0, 10, 2)
        worklife      = st.slider("Work Life Balance (1–4)", 1, 4, 3)
        experience    = st.slider("Total Working Years", 0, 40, 5)
        years_company = st.slider("Years At Company", 0, 40, 3)

    if st.button("🔮 Predict Salary", use_container_width=True):
        dept = le_dict['Department'].transform([department])[0]
        role = le_dict['JobRole'].transform([job_role])[0]

        input_data = np.array([[
            age, dept, role, job_level,
            satisfaction, performance,
            training, worklife,
            experience, years_company
        ]])

        pred = model.predict(input_data)[0]
        st.success(f"### 💵 Predicted Monthly Salary: ₹{int(pred):,}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Skill Recommendation
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Skill Recommendation":
    st.title("🎯 Skill Upgrade Recommendations")
    st.markdown("Get personalised skill recommendations based on role and performance.")

    col1, col2, col3 = st.columns(3)
    with col1:
        role = st.selectbox("Job Role", list(role_skills.keys()))
    with col2:
        perf = st.slider("Performance Rating (1–4)", 1, 4, 3)
    with col3:
        train = st.slider("Training Times Last Year", 0, 10, 2)

    if st.button("🎯 Get Recommendations", use_container_width=True):
        skills = list(role_skills.get(role, ["General Skills"]))

        if perf < 3:
            skills.append("Performance Improvement Plan")
        if train < 2:
            skills.append("Enroll in Training Programs")

        st.markdown("### ✅ Recommended Skills / Actions")
        for i, s in enumerate(skills, 1):
            if "Improvement" in s or "Enroll" in s:
                st.warning(f"⚠️ {i}. {s}")
            else:
                st.success(f"✅ {i}. {s}")
