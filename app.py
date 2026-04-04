import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

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
    'Sales Executive':           ['Communication', 'Negotiation', 'CRM'],
    'Research Scientist':        ['Python', 'Machine Learning', 'Statistics'],
    'Laboratory Technician':     ['Lab Skills', 'Data Analysis'],
    'Manufacturing Director':    ['Operations', 'Planning', 'Leadership'],
    'Healthcare Representative': ['Medical Knowledge', 'Communication'],
    'Manager':                   ['Leadership', 'Strategy', 'People Management'],
    'Human Resources':           ['Recruitment', 'HR Management', 'Communication'],
    'Research Director':         ['Advanced ML', 'Deep Learning', 'Leadership'],
}

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "💰 Salary Prediction", "🎯 Skill Recommendation", "📂 Upload Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Data Source")

# ── Upload widget in sidebar ──────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="CSV must contain: " + ", ".join(REQUIRED_COLS)
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
            raw_bytes = uploaded_file.read()
            df_up = pd.read_csv(io.BytesIO(raw_bytes))
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
    st.sidebar.success(f"✅ Using: **{st.session_state.upload_name}**  \n"
                       f"({len(st.session_state.uploaded_df):,} rows)")
    if st.sidebar.button("🔄 Reset to Default Data"):
        st.session_state.uploaded_df   = None
        st.session_state.upload_errors = []
        st.session_state.upload_name   = None
        st.session_state.upload_enc    = None
        st.session_state.upload_model  = None
        st.rerun()
elif st.session_state.upload_errors:
    st.sidebar.error("❌ Upload failed")
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
# PAGE 1 – Dashboard
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Employee Skill Upgrade Recommendation System")
    st.markdown("### Dashboard – Overview of Employee Data")

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Employees", f"{len(df_raw):,}")
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
        dept_avg = df_raw.groupby('Department')['MonthlyIncome'].mean().sort_values().reset_index()
        sns.barplot(x='MonthlyIncome', y='Department', hue='Department', data=dept_avg, ax=ax2, palette='viridis', legend=False)
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

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Upload Data (dedicated page)
# ════════════════════════════════════════════════════════════════════════════
elif page == "📂 Upload Data":
    st.title("📂 Upload Your Employee Dataset")
    st.markdown(
        "Upload a custom CSV file to replace the default dataset. "
        "The model will automatically retrain on your data."
    )

    # ── Show required columns ─────────────────────────────────────────────
    with st.expander("📋 Required CSV Columns", expanded=True):
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
        "Choose a CSV file",
        type=["csv"],
        key="inline_uploader",
        help="Must contain all required columns listed above"
    )

    active_file = inline_file if inline_file is not None else uploaded_file

    if active_file is not None:
        st.markdown("---")
        try:
            raw_bytes = active_file.read() if inline_file else open(
                'employee_attrition_test.csv', 'rb').read()

            if inline_file:
                df_preview = pd.read_csv(io.BytesIO(raw_bytes))
            else:
                df_preview = df_raw.copy()

            missing_cols = validate_dataframe(df_preview)

            if missing_cols:
                st.error(f"❌ Missing required columns: **{', '.join(missing_cols)}**")
                st.markdown("Please check your file and re-upload with the correct columns.")
            else:
                # ── Stats ─────────────────────────────────────────────────
                st.success(f"✅ File validated successfully! **{len(df_preview):,}** rows loaded.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📄 Total Rows",      f"{len(df_preview):,}")
                m2.metric("📊 Total Columns",   len(df_preview.columns))
                m3.metric("💵 Avg Income",      f"₹{int(df_preview['MonthlyIncome'].mean()):,}")
                m4.metric("🧹 Duplicate Rows",  df_preview.duplicated().sum())

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
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.histplot(df_preview['MonthlyIncome'], kde=True, ax=ax, color='steelblue')
                        ax.set_title("Monthly Income Distribution")
                        ax.set_xlabel("Monthly Income (₹)")
                        st.pyplot(fig)
                        plt.close(fig)
                    with c2:
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        role_counts = df_preview['JobRole'].value_counts().reset_index()
                        role_counts.columns = ['JobRole', 'Count']
                        sns.barplot(x='Count', y='JobRole', hue='JobRole', data=role_counts, ax=ax2, palette='magma', legend=False)
                        ax2.set_title("Employees by Job Role")
                        ax2.set_xlabel("Count")
                        st.pyplot(fig2)
                        plt.close(fig2)

                st.markdown("---")

                # ── Download template ─────────────────────────────────────
                st.markdown("### 📥 Download Sample Template")
                sample_df = generate_data().head(10)
                csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download CSV Template (10 rows)",
                    data=csv_bytes,
                    file_name="employee_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    else:
        # ── Empty state ───────────────────────────────────────────────────
        st.info("👈 Upload a CSV file using the sidebar or the uploader above to get started.")

        st.markdown("### 📥 Don't have a file? Download our template!")
        sample_df = generate_data().head(10)
        csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV Template (10 rows)",
            data=csv_bytes,
            file_name="employee_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── Show errors from sidebar upload ──────────────────────────────────
    if st.session_state.upload_errors and uploaded_file is not None:
        st.markdown("---")
        st.error(
            f"❌ Last upload failed — missing columns: "
            f"**{', '.join(st.session_state.upload_errors)}**"
        )
