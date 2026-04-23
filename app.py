import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import tempfile
import json

st.set_page_config(
    page_title="Employee Skill Upgrade Recommendation System",
    page_icon="🚀",
    layout="wide"
)

# ── Required columns ─────────────────────────────────────────────────────────
REQUIRED_COLS = [
    'Age', 'Department', 'JobRole', 'JobLevel', 'JobSatisfaction',
    'PerformanceRating', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'TotalWorkingYears', 'YearsAtCompany', 'MonthlyIncome'
]

# ── Helper: generate synthetic data if CSV missing ───────────────────────────
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

# ── Helper: parse uploaded files ──────────────────────────────────────────────
def parse_uploaded_file(file_obj):
    ext = file_obj.name.split('.')[-1].lower()
    raw_bytes = file_obj.read()
    if ext == 'csv':
        return pd.read_csv(io.BytesIO(raw_bytes))
    elif ext == 'xlsx':
        return pd.read_excel(io.BytesIO(raw_bytes))
    elif ext == 'json':
        return pd.read_json(io.BytesIO(raw_bytes))
    elif ext in ['sqlite', 'db', 'sql']:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite') as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, conn)
        if not tables.empty:
            first_table = tables['name'].iloc[0]
            df = pd.read_sql(f"SELECT * FROM {first_table}", conn)
        else:
            conn.close()
            os.remove(tmp_path)
            raise ValueError("No tables found in the SQLite database")
        conn.close()
        os.remove(tmp_path)
        return df
    else:
        raise ValueError("Unsupported file format")

def build_download_templates(sample_df):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("CSV Template", csv_bytes, "template.csv", "text/csv", use_container_width=True)
    with c2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sample_df.to_excel(writer, index=False)
        xlsx_data = output.getvalue()
        st.download_button("Excel Template", xlsx_data, "template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c3:
        json_bytes = sample_df.to_json(orient="records").encode('utf-8')
        st.download_button("JSON Template", json_bytes, "template.json", "application/json", use_container_width=True)
    with c4:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite') as tmp:
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        sample_df.to_sql("employees", conn, index=False, if_exists="replace")
        conn.close()
        with open(tmp_path, "rb") as f:
            sql_bytes = f.read()
        os.remove(tmp_path)
        st.download_button("SQLite Template", sql_bytes, "template.sqlite", "application/x-sqlite3", use_container_width=True)

# ── Load / generate default data ─────────────────────────────────────────────
@st.cache_data
def load_default_data():
    csv_path = 'employee_attrition_test.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_data()
        df.to_csv(csv_path, index=False)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# ── Validate uploaded dataframe ───────────────────────────────────────────────
def validate_dataframe(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing

# ── Encode data & train model (works on any valid df) ────────────────────────
def build_encoders_and_model(df_raw, force_retrain=False):
    le_dict = {}
    df_enc = df_raw.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    model_path = 'salary_model.pkl'
    model = None

    if not force_retrain and os.path.exists(model_path):
        try:
            model = pickle.load(open(model_path, 'rb'))
        except Exception:
            model = None

    if model is None or force_retrain:
        X = df_enc.drop('MonthlyIncome', axis=1)
        y = df_enc['MonthlyIncome']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        if not force_retrain:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

    return df_enc, le_dict, model

# ── Skill map ─────────────────────────────────────────────────────────────────
role_skills = {
    'Sales Executive': [
        {"skill": "Advanced Negotiation", "reason": "Close high-value deals", "level": "Advanced", "course": "Negotiation Mastery", "provider": "Coursera", "duration": "4 weeks", "duration_weeks": 4, "impact": "High 💰", "cost": "₹4,000", "cost_inr": 4000, "prereq": "Basic Sales Strategy"},
        {"skill": "Salesforce/CRM", "reason": "Improve pipeline management", "level": "Intermediate", "course": "Salesforce Admin", "provider": "Trailhead", "duration": "2 weeks", "duration_weeks": 2, "impact": "Medium 🚀", "cost": "Free", "cost_inr": 0, "prereq": "None"},
        {"skill": "Public Speaking", "reason": "Boost confidence in pitches", "level": "Beginner", "course": "Dynamic Public Speaking", "provider": "Udemy", "duration": "3 weeks", "duration_weeks": 3, "impact": "Medium 🗣️", "cost": "₹1,200", "cost_inr": 1200, "prereq": "None"}
    ],
    'Research Scientist': [
        {"skill": "Deep Learning", "reason": "AI research focus", "level": "Advanced", "course": "Deep Learning Specialization", "provider": "Coursera", "duration": "12 weeks", "duration_weeks": 12, "impact": "High 💰", "cost": "₹11,800", "cost_inr": 11800, "prereq": "Python, Math"},
        {"skill": "Statistical Modeling", "reason": "Experimental design", "level": "Intermediate", "course": "Bayesian Statistics", "provider": "edX", "duration": "6 weeks", "duration_weeks": 6, "impact": "High 📈", "cost": "₹8,000", "cost_inr": 8000, "prereq": "Basic Stats"},
        {"skill": "Cloud Computing", "reason": "Scale experiments", "level": "Beginner", "course": "AWS Cloud Practitioner", "provider": "AWS", "duration": "2 weeks", "duration_weeks": 2, "impact": "Medium ☁️", "cost": "Free", "cost_inr": 0, "prereq": "None"}
    ],
    'Laboratory Technician': [
        {"skill": "Equipment Handling", "reason": "Lab safety/efficiency", "level": "Intermediate", "course": "Advanced Lab Techniques", "provider": "edX", "duration": "4 weeks", "duration_weeks": 4, "impact": "Medium 🔬", "cost": "₹4,000", "cost_inr": 4000, "prereq": "Safety Protocols"},
        {"skill": "Data Analysis (R/Python)", "reason": "Doc and Analysis", "level": "Intermediate", "course": "Data Analysis with Python", "provider": "DataCamp", "duration": "6 weeks", "duration_weeks": 6, "impact": "High 📊", "cost": "₹3,000/mo", "cost_inr": 3000, "prereq": "None"},
        {"skill": "Quality Control (GLP)", "reason": "Regulatory compliance", "level": "Beginner", "course": "Good Laboratory Practice", "provider": "Udemy", "duration": "1 week", "duration_weeks": 1, "impact": "Low 📋", "cost": "₹1,200", "cost_inr": 1200, "prereq": "None"}
    ],
    'Manufacturing Director': [
        {"skill": "Lean Six Sigma", "reason": "Process optimization", "level": "Advanced", "course": "Six Sigma Green Belt", "provider": "Coursera", "duration": "8 weeks", "duration_weeks": 8, "impact": "High 💰", "cost": "₹6,300", "cost_inr": 6300, "prereq": "Management Exp."},
        {"skill": "Supply Chain Mgmt", "reason": "Reduce delays/costs", "level": "Intermediate", "course": "Supply Chain Excellence", "provider": "edX", "duration": "5 weeks", "duration_weeks": 5, "impact": "High 📦", "cost": "₹12,000", "cost_inr": 12000, "prereq": "Basic Operations"},
        {"skill": "Strategic Leadership", "reason": "Plant operations", "level": "Advanced", "course": "Executive Leadership", "provider": "Harvard Online", "duration": "6 weeks", "duration_weeks": 6, "impact": "Very High 👑", "cost": "₹160,000", "cost_inr": 160000, "prereq": "5+ yrs Leadership"}
    ],
    'Healthcare Representative': [
        {"skill": "Medical Device Knowledge", "reason": "Accurate pitching", "level": "Intermediate", "course": "MedTech Sales", "provider": "Udemy", "duration": "3 weeks", "duration_weeks": 3, "impact": "High 🩺", "cost": "₹1,600", "cost_inr": 1600, "prereq": "Life Sciences Degree"},
        {"skill": "Healthcare Compliance", "reason": "Ethical sales", "level": "Beginner", "course": "HIPAA & Compliance", "provider": "CTG", "duration": "1 week", "duration_weeks": 1, "impact": "Medium ⚖️", "cost": "₹3,200", "cost_inr": 3200, "prereq": "None"},
        {"skill": "Relationship Mgmt", "reason": "Build trust", "level": "Intermediate", "course": "B2B Relationship Sales", "provider": "LinkedIn", "duration": "2 weeks", "duration_weeks": 2, "impact": "Medium 🤝", "cost": "₹2,400/mo", "cost_inr": 2400, "prereq": "Basic Sales"}
    ],
    'Manager': [
        {"skill": "Agile Project Mgmt", "reason": "Accelerate delivery", "level": "Intermediate", "course": "Agile Crash Course", "provider": "Udemy", "duration": "2 weeks", "duration_weeks": 2, "impact": "High 🚀", "cost": "₹1,200", "cost_inr": 1200, "prereq": "None"},
        {"skill": "Conflict Resolution", "reason": "Team harmony", "level": "Intermediate", "course": "Managing Team Conflict", "provider": "Coursera", "duration": "3 weeks", "duration_weeks": 3, "impact": "Medium 🕊️", "cost": "₹4,000", "cost_inr": 4000, "prereq": "None"},
        {"skill": "Financial Acumen", "reason": "Budget planning", "level": "Advanced", "course": "Finance for Managers", "provider": "Coursera", "duration": "4 weeks", "duration_weeks": 4, "impact": "High 💰", "cost": "₹4,000", "cost_inr": 4000, "prereq": "Basic Accounting"}
    ],
    'Human Resources': [
        {"skill": "Talent Analytics", "reason": "Improve hiring quality", "level": "Intermediate", "course": "People Analytics", "provider": "Wharton Online", "duration": "5 weeks", "duration_weeks": 5, "impact": "High 📈", "cost": "₹16,000", "cost_inr": 16000, "prereq": "HR Principles"},
        {"skill": "Employer Branding", "reason": "Attract candidates", "level": "Intermediate", "course": "Brand Strategy", "provider": "LinkedIn", "duration": "2 weeks", "duration_weeks": 2, "impact": "Medium ✨", "cost": "₹2,400", "cost_inr": 2400, "prereq": "None"},
        {"skill": "D&I Strategies", "reason": "Healthier workplace", "level": "Beginner", "course": "Diversity & Inclusion", "provider": "Coursera", "duration": "4 weeks", "duration_weeks": 4, "impact": "Medium 🌍", "cost": "₹4,000", "cost_inr": 4000, "prereq": "None"}
    ],
    'Research Director': [
        {"skill": "Grants Writing", "reason": "Secure budgeting", "level": "Advanced", "course": "Grant Writing 101", "provider": "edX", "duration": "4 weeks", "duration_weeks": 4, "impact": "Very High 💰", "cost": "₹12,000", "cost_inr": 12000, "prereq": "Research Exp."},
        {"skill": "R&D Strategy", "reason": "Align with business", "level": "Advanced", "course": "Innovation Strategy", "provider": "MIT OpenCourseWare", "duration": "6 weeks", "duration_weeks": 6, "impact": "High 🧠", "cost": "Free", "cost_inr": 0, "prereq": "PhD/Leadership"},
        {"skill": "Executive Comms", "reason": "Board presentations", "level": "Intermediate", "course": "Communicating for Impact", "provider": "LinkedIn", "duration": "2 weeks", "duration_weeks": 2, "impact": "Medium 🎙️", "cost": "₹2,400", "cost_inr": 2400, "prereq": "None"}
    ]
}

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.title(" Navigation")
page = st.sidebar.radio(
    "Go to",
    [" Dashboard", " Salary Prediction", "Skill Recommendation", " Upload Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Source")

# ── Upload widget in sidebar ──────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "json", "sqlite", "db"],
    help="Dataset must contain: " + ", ".join(REQUIRED_COLS) + ". Supported: CSV, Excel, JSON, SQLite."
)

# ── Session state for uploaded data ──────────────────────────────────────────
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df   = None
    st.session_state.upload_errors = []
    st.session_state.upload_name   = None
    st.session_state.upload_enc    = None
    st.session_state.upload_model  = None

# ── Process upload ────────────────────────────────────────────────────────────
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.upload_name:
        # New file uploaded — parse & validate
        try:
            df_up = parse_uploaded_file(uploaded_file)
            df_up.dropna(inplace=True)
            df_up.drop_duplicates(inplace=True)
            missing_cols = validate_dataframe(df_up)

            if missing_cols:
                st.session_state.upload_errors = missing_cols
                st.session_state.uploaded_df   = None
                st.session_state.upload_name   = uploaded_file.name
                st.session_state.upload_enc    = None
                st.session_state.upload_model  = None
            else:
                with st.spinner("🔄 Training model on uploaded data…"):
                    df_enc_up, le_dict_up, model_up = build_encoders_and_model(
                        df_up[REQUIRED_COLS], force_retrain=True
                    )
                st.session_state.uploaded_df   = df_up
                st.session_state.upload_errors = []
                st.session_state.upload_name   = uploaded_file.name
                st.session_state.upload_enc    = (df_enc_up, le_dict_up, model_up)
        except Exception as e:
            st.session_state.upload_errors = [str(e)]
            st.session_state.uploaded_df   = None
            st.session_state.upload_name   = uploaded_file.name

# ── Sidebar: show status ──────────────────────────────────────────────────────
if st.session_state.uploaded_df is not None:
    st.sidebar.success(f"Using: **{st.session_state.upload_name}**  \n"
                       f"({len(st.session_state.uploaded_df):,} rows)")
    if st.sidebar.button("Reset to Default Data"):
        st.session_state.uploaded_df   = None
        st.session_state.upload_errors = []
        st.session_state.upload_name   = None
        st.session_state.upload_enc    = None
        st.session_state.upload_model  = None
        st.rerun()
elif st.session_state.upload_errors:
    st.sidebar.error(" Upload failed")
else:
    st.sidebar.info("Using default dataset")

# ── Choose active data & model ────────────────────────────────────────────────
if st.session_state.uploaded_df is not None and st.session_state.upload_enc is not None:
    df_raw             = st.session_state.uploaded_df
    df_enc, le_dict, model = st.session_state.upload_enc
else:
    df_raw = load_default_data()
    df_enc, le_dict, model = build_encoders_and_model(df_raw)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Dashboard & Skill Gap Analysis
# ════════════════════════════════════════════════════════════════════════════
if page == " Dashboard":
    st.title(" Enterprise Skill & Performance Dashboard")
    st.markdown("Monitor company-wide employee data, analyze skill gaps, and track learning progress.")

    tab_overview, tab_gap, tab_progress = st.tabs([
        " Company Overview", "Skill Gap Analysis", " Training Progress"
    ])

    with tab_overview:
        st.markdown("### Overview of Employee Data")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(" Total Employees", f"{len(df_raw):,}")
        col2.metric(" Avg Monthly Income", f"₹{int(df_raw['MonthlyIncome'].mean()):,}")
        col3.metric(" Avg Performance", f"{df_raw['PerformanceRating'].mean():.2f}")
        col4.metric(" Avg Work-Life", f"{df_raw['WorkLifeBalance'].mean():.2f}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            fig1 = px.histogram(df_raw, x="MonthlyIncome", nbins=30, title="Monthly Income Distribution", color_discrete_sequence=['steelblue'])
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            dept_avg = df_raw.groupby('Department')['MonthlyIncome'].mean().sort_values().reset_index()
            fig2 = px.bar(dept_avg, x="MonthlyIncome", y="Department", orientation='h', title="Avg Income by Department", color="Department")
            st.plotly_chart(fig2, use_container_width=True)

    with tab_gap:
        st.markdown("###  Skill Gap Risk Analysis")
        st.info("A 'Skill Gap' is flagged when an employee's performance rating falls short of the target 4.0 standard.")
        
        df_gap = df_raw.copy()
        df_gap['TargetPerformance'] = 4.0
        df_gap['SkillGap'] = df_gap['TargetPerformance'] - df_gap['PerformanceRating']
        df_gap['SkillGap'] = df_gap['SkillGap'].clip(lower=0)
        
        gap_by_role = df_gap.groupby('JobRole')['SkillGap'].mean().sort_values(ascending=False).reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.bar(gap_by_role, x='SkillGap', y='JobRole', orientation='h', title="Average Skill Gap by Job Role", color="SkillGap", color_continuous_scale="Reds")
            st.plotly_chart(fig3, use_container_width=True)
            
        with c2:
            # Scatter to find at-risk employees (High exp, low perf)
            fig4 = px.scatter(df_gap, x='YearsAtCompany', y='PerformanceRating', color='SkillGap', size='SkillGap', title="Tenure vs Performance", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig4, use_container_width=True)
            
        high_risk_count = len(df_gap[(df_gap['SkillGap'] >= 1.5) & (df_gap['YearsAtCompany'] > 3)])
        st.warning(f"**Action Required:** We found **{high_risk_count}** tenured employees (>3 yrs) with a significant skill gap. Consider allocating bootcamp budgets to upskill them immediately.")

    with tab_progress:
        st.markdown("### Corporate Up-skilling Tracker")
        st.write("Track the adoption of learning initiatives across the enterprise.")
        
        # Simulate active enrollments based on TrainingTimesLastYear
        active_learners = len(df_raw[df_raw['TrainingTimesLastYear'] >= 2])
        total_emp = len(df_raw)
        adoption_rate = active_learners / total_emp
        
        c1, c2, c3 = st.columns(3)
        c1.metric(" Active Learners", f"{active_learners:,}")
        c2.metric(" Company Learning Goal", "80% Enrolled")
        c3.metric(" Current Adoption", f"{int(adoption_rate*100)}%")
        
        st.markdown("#### Progress towards Annual Up-skill Goal")
        st.progress(min(adoption_rate / 0.80, 1.0))
        
        st.markdown("---")
        st.markdown("####  Top Enrolled Courses (Simulated from Active Rules)")
        
        course_counts = {}
        for role, count in df_raw['JobRole'].value_counts().items():
            if role in role_skills and len(role_skills[role]) > 0:
                top_course = role_skills[role][0]['course']
                course_counts[top_course] = course_counts.get(top_course, 0) + int(count * 0.4)
                
        course_df = pd.DataFrame(list(course_counts.items()), columns=["Course", "Enrollees"]).sort_values(by="Enrollees", ascending=False).head(5)
        
        c_a, c_b = st.columns([1.5, 1])
        with c_a:
            st.dataframe(course_df, use_container_width=True, hide_index=True)
        with c_b:
            fig5 = px.pie(course_df, names='Course', values='Enrollees', hole=0.3)
            fig5.update_layout(showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Salary Prediction
# ════════════════════════════════════════════════════════════════════════════
elif page == " Salary Prediction":
    st.title(" Salary Prediction")
    st.markdown("Fill in employee details to predict their monthly income.")

    col1, col2 = st.columns(2)

    with col1:
        age          = st.slider("Age", 18, 60, 30)
        department   = st.selectbox("Department", le_dict['Department'].classes_)
        job_role     = st.selectbox("Job Role", le_dict['JobRole'].classes_)
        job_level    = st.slider("Job Level", 1, 5, 2)
        satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)

    with col2:
        performance   = st.slider("Performance Rating (1–4)", 1, 4, 3)
        training      = st.slider("Training Times Last Year", 0, 10, 2)
        worklife      = st.slider("Work Life Balance (1–4)", 1, 4, 3)
        experience    = st.slider("Total Working Years", 0, 40, 5)
        years_company = st.slider("Years At Company", 0, 40, 3)

    if st.button(" Predict Salary", use_container_width=True):
        dept = le_dict['Department'].transform([department])[0]
        role = le_dict['JobRole'].transform([job_role])[0]

        input_data = np.array([[
            age, dept, role, job_level,
            satisfaction, performance,
            training, worklife,
            experience, years_company
        ]])

        pred = model.predict(input_data)[0]
        st.success(f"### Predicted Monthly Salary: ₹{int(pred):,}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Skill Recommendation
# ════════════════════════════════════════════════════════════════════════════
elif page == "Skill Recommendation":
    st.title(" Skill Upgrade Recommendations")
    st.markdown("Get personalised skill recommendations based on role and performance.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#####  Employee Details")
        role = st.selectbox("Job Role", list(role_skills.keys()))
        perf = st.slider("Performance Rating (1–4)", 1, 4, 3)
    with col2:
        st.markdown("#####  Course Preferences")
        max_duration = st.slider(" Max Course Duration (Weeks)", 1, 16, 12)
        max_budget = st.slider(" Max Budget per Course (₹)", 0, 200000, 20000, step=1000)

    if "recommendation_active" not in st.session_state:
        st.session_state.recommendation_active = False
        st.session_state.recommendation_role = None
        st.session_state.recommendation_perf = None
        st.session_state.rec_duration = None
        st.session_state.rec_budget = None

    if st.button(" Find Matching Courses", use_container_width=True):
        st.session_state.recommendation_active = True
        st.session_state.recommendation_role = role
        st.session_state.recommendation_perf = perf
        st.session_state.rec_duration = max_duration
        st.session_state.rec_budget = max_budget

    if st.session_state.recommendation_active:
        active_role = st.session_state.recommendation_role
        active_perf = st.session_state.recommendation_perf
        active_duration = st.session_state.rec_duration
        active_budget = st.session_state.rec_budget
        
        all_skills = role_skills.get(active_role, [])
        
        # ── Dynamic Skill Filtering based on Sliders ──
        filtered_skills = [
            s for s in all_skills 
            if s['cost_inr'] <= active_budget and s['duration_weeks'] <= active_duration
        ]
        
        # Overlay performance logic
        if active_perf <= 2:
            st.info(" **Customized for you:** Highlighting essential foundational & core job skills.")
            skills = [s for s in filtered_skills if s['level'] in ['Beginner', 'Intermediate']]
        elif active_perf == 3:
            st.info(" **Customized for you:** Balanced mix of Intermediate and Advanced upskilling.")
            skills = [s for s in filtered_skills if s['level'] in ['Intermediate', 'Advanced']]
        else:
            st.info(" **Customized for you:** Advanced mastery and Leadership tracks for high performers.")
            skills = [s for s in filtered_skills if s['level'] == 'Advanced']
            
        # Fallback if filters are too restrictive
        if not skills:
            if not filtered_skills:
                st.warning(" No courses match your strict budget or duration filters. Showing all recommendations.")
                skills = all_skills
            else:
                skills = filtered_skills

        st.markdown("---")
        st.markdown(f"##  Learning Path for **{active_role}**")
        
        st.markdown("###  Urgent Action Items")
        if active_perf < 3:
            st.error("**Performance Improvement Plan:** Recommended based on recent performance rating.")
        else:
            st.info("No urgent action items. Keep up the good work!")

        st.markdown("###  Core Skills Checklist")
        if not skills:
            st.info("General Skills")
        else:
            total_skills = len(skills)
            checked_count = 0
            
            # Using columns for each checklist item
            for i, s in enumerate(skills, 1):
                col_chk, col_exp = st.columns([0.5, 9.5])
                with col_chk:
                    # Checkbox for tracking progress (keys must be unique across the app)
                    chk_key = f"chk_{active_role.replace(' ', '_')}_{i}"
                    is_done = st.checkbox("Done", key=chk_key, label_visibility="collapsed")
                    if is_done:
                        checked_count += 1
                
                with col_exp:
                    with st.expander(f"**Step {i}: {s['skill']}**  |  Level: {s['level']}  |  Impact: {s['impact']}", expanded=(not is_done)):
                        c1, c2 = st.columns(2)
                        c1.markdown(f"** Why?** {s['reason']}")
                        c1.markdown(f"** Salary Impact:** {s['impact']}")
                        c1.markdown(f"** Prerequisites:** {s.get('prereq', 'None')}")
                        
                        c2.markdown(f"**Suggested Course:** {s['course']} ({s['provider']})")
                        c2.markdown(f"** Duration:** {s['duration']}")
                        c2.markdown(f"** Estimated Cost:** {s.get('cost', 'Free')}")
            
            # Progress bar based on checks
            st.markdown("### Overall Progress")
            progress_frac = checked_count / total_skills
            st.progress(progress_frac)
            if progress_frac == 1.0:
                st.success(" **Amazing!** You've completed the recommended learning path for this role!")
            else:
                st.caption(f"**Current Progress:** {int(progress_frac * 100)}% ({checked_count}/{total_skills} skills attained)")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Upload Data (dedicated page)
# ════════════════════════════════════════════════════════════════════════════
elif page == " Upload Data":
    st.title(" Upload Your Employee Dataset")
    st.markdown(
        "Upload a custom CSV file to replace the default dataset. "
        "The model will automatically retrain on your data."
    )

    # ── Show required columns ─────────────────────────────────────────────
    with st.expander("Required CSV Columns", expanded=True):
        col_info = {
            "Column": REQUIRED_COLS,
            "Type": ["Integer", "String", "String", "Integer (1–5)",
                     "Integer (1–4)", "Integer (1–4)", "Integer (0–10)",
                     "Integer (1–4)", "Integer", "Integer", "Integer"],
            "Example": [30, "Sales", "Manager", 3, 2, 3, 2, 3, 8, 5, 8500]
        }
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Upload area ───────────────────────────────────────────────────────
    st.markdown("### Drop your file here or use the sidebar uploader")
    inline_file = st.file_uploader(
        "Choose a file (CSV, Excel, JSON, SQLite)",
        type=["csv", "xlsx", "json", "sqlite", "db"],
        key="inline_uploader",
        help="Must contain all required columns listed above"
    )

    active_file = inline_file if inline_file is not None else uploaded_file

    if active_file is not None:
        st.markdown("---")
        try:
            if inline_file:
                inline_file.seek(0)
                df_preview = parse_uploaded_file(inline_file)
            else:
                df_preview = df_raw.copy()

            missing_cols = validate_dataframe(df_preview)

            if missing_cols:
                st.error(f"Missing required columns: **{', '.join(missing_cols)}**")
                st.markdown("Please check your file and re-upload with the correct columns.")
            else:
                # ── Stats ─────────────────────────────────────────────────
                st.success(f"File validated successfully! **{len(df_preview):,}** rows loaded.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(" Total Rows",      f"{len(df_preview):,}")
                m2.metric(" Total Columns",   len(df_preview.columns))
                m3.metric(" Avg Income",      f"₹{int(df_preview['MonthlyIncome'].mean()):,}")
                m4.metric("Duplicate Rows",  df_preview.duplicated().sum())

                st.markdown("---")

                # ── Tabs: Preview, Stats, Charts ──────────────────────────
                tab1, tab2, tab3 = st.tabs(["🔍 Data Preview", "📊 Statistics", "📈 Charts"])

                with tab1:
                    search_query = st.text_input("🔎 Filter by Department or Job Role", "")
                    df_show = df_preview.copy()
                    if search_query:
                        mask = (
                            df_show['Department'].astype(str).str.contains(search_query, case=False, na=False) |
                            df_show['JobRole'].astype(str).str.contains(search_query, case=False, na=False)
                        )
                        df_show = df_show[mask]
                    st.dataframe(df_show.head(100), use_container_width=True)
                    st.caption(f"Showing up to 100 rows (filtered: {len(df_show):,} matches)")

                with tab2:
                    st.dataframe(df_preview.describe().T.style.format("{:.2f}"),
                                 use_container_width=True)

                with tab3:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_up1 = px.histogram(df_preview, x="MonthlyIncome", nbins=30, title="Monthly Income Distribution", color_discrete_sequence=['steelblue'])
                        st.plotly_chart(fig_up1, use_container_width=True)
                    with c2:
                        role_counts = df_preview['JobRole'].value_counts().reset_index()
                        role_counts.columns = ['JobRole', 'Count']
                        fig_up2 = px.bar(role_counts, x='Count', y='JobRole', orientation='h', title="Employees by Job Role", color="JobRole")
                        st.plotly_chart(fig_up2, use_container_width=True)

                st.markdown("---")

                # ── Download template ─────────────────────────────────────
                st.markdown("###  Download Sample Templates")
                sample_df = generate_data().head(10)
                build_download_templates(sample_df)

        except Exception as e:
            st.error(f" Error reading file: {e}")

    else:
        # ── Empty state ───────────────────────────────────────────────────
        st.info(" Upload a CSV file using the sidebar or the uploader above to get started.")

        st.markdown("### Don't have a file? Download our template!")
        sample_df = generate_data().head(10)
        build_download_templates(sample_df)

    # ── Show errors from sidebar upload ──────────────────────────────────
    if st.session_state.upload_errors and uploaded_file is not None:
        st.markdown("---")
        st.error(
            f" Last upload failed — missing columns: "
            f"**{', '.join(st.session_state.upload_errors)}**"
        )
