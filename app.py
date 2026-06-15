import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Gen Z Mental Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("student_mental_health_burnout.csv")

df = load_data()

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return joblib.load("models/burnout_model.pkl")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Control Panel")
st.sidebar.caption("Use filters and inputs to explore burnout patterns.")

st.sidebar.header("📊 Dashboard Filter")

burnout_filter = st.sidebar.selectbox(
    "Burnout Level",
    ["All"] + list(df["burnout_level"].unique())
)

filtered_df = df.copy()

if burnout_filter != "All":
    filtered_df = filtered_df[
        filtered_df["burnout_level"] == burnout_filter
    ]

st.sidebar.header("🧠 ML Burnout Prediction")

age = st.sidebar.number_input("Age", 18, 35, 21)

gender_text = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

if gender_text == "Male":
    gender = 1
elif gender_text == "Female":
    gender = 0
else:
    gender = 2

course_text = st.sidebar.selectbox(
    "Course",
    ["BCA", "BCom", "BSc", "BTech"]
)

course_map = {
    "BCA": 0,
    "BCom": 1,
    "BSc": 2,
    "BTech": 3
}

course = course_map[course_text]

year_text = st.sidebar.selectbox(
    "Year",
    ["1st", "2nd", "3rd", "4th"]
)

year_map = {
    "1st": 0,
    "2nd": 1,
    "3rd": 2,
    "4th": 3
}

year = year_map[year_text]

study_hours = st.sidebar.slider(
    "Daily Study Hours",
    0.0,
    12.0,
    4.0
)

sleep_hours = st.sidebar.slider(
    "Daily Sleep Hours",
    0.0,
    12.0,
    7.0
)

screen_time = st.sidebar.slider(
    "Screen Time Hours",
    0.0,
    12.0,
    5.0
)

cgpa = st.sidebar.slider(
    "CGPA",
    0.0,
    10.0,
    7.0
)

# =========================
# HEADER
# =========================

st.markdown("""
## 🎯 Project Goal

This dashboard analyzes the impact of sleep, screen time, academic pressure,
and lifestyle factors on student burnout and mental health.
""")

st.title("🧠 Gen Z Mental Health Dashboard")

st.markdown("""
Analyze student burnout, screen time, sleep patterns, and academic performance.
""")

# =========================
# KPI CARDS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Students", len(filtered_df))

with col2:
    st.metric(
        "Avg Sleep",
        round(filtered_df["daily_sleep_hours"].mean(), 2)
    )

with col3:
    st.metric(
        "Avg Screen Time",
        round(filtered_df["screen_time_hours"].mean(), 2)
    )

with col4:
    st.metric(
        "Avg CGPA",
        round(filtered_df["cgpa"].mean(), 2)
    )

st.markdown("---")

# =========================
# CHARTS
# =========================

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("📌 Burnout Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        data=filtered_df,
        x="burnout_level",
        ax=ax
    )

    ax.set_xlabel("Burnout Level")
    ax.set_ylabel("Count")

    st.pyplot(fig)

with chart2:
    st.subheader("📱 Screen Time Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        filtered_df["screen_time_hours"],
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Screen Time Hours")
    ax.set_ylabel("Count")

    st.pyplot(fig)

chart3, chart4 = st.columns(2)

with chart3:
    st.subheader("😴 Sleep vs Burnout")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        data=filtered_df,
        x="burnout_level",
        y="daily_sleep_hours",
        ax=ax
    )

    ax.set_xlabel("Burnout Level")
    ax.set_ylabel("Daily Sleep Hours")

    st.pyplot(fig)

with chart4:
    st.subheader("🎓 CGPA vs Burnout")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        data=filtered_df,
        x="burnout_level",
        y="cgpa",
        ax=ax
    )

    ax.set_xlabel("Burnout Level")
    ax.set_ylabel("CGPA")

    st.pyplot(fig)

# =========================
# HEATMAP
# =========================

st.subheader("📊 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False,
    ax=ax
)

st.pyplot(fig)

st.markdown("---")

# =========================
# ML PREDICTION
# =========================

st.subheader("🎯 ML Prediction Result")

input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "course": course,
    "year": year,
    "daily_study_hours": study_hours,
    "daily_sleep_hours": sleep_hours,
    "screen_time_hours": screen_time,
    "stress_level": 1,
    "anxiety_score": 5,
    "depression_score": 5,
    "academic_pressure_score": 5,
    "financial_stress_score": 5,
    "social_support_score": 5,
    "physical_activity_hours": 1.5,
    "sleep_quality": 1,
    "attendance_percentage": 75,
    "cgpa": cgpa,
    "internet_quality": 1
}])

if model_loaded:

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.error("🔴 High Burnout")

    elif prediction == 1:
        st.success("🟢 Low Burnout")

    else:
        st.warning("🟡 Medium Burnout")
# =========================
# RULE BASED RISK SCORE
# =========================

risk = 0

if sleep_hours < 6:
    risk += 30

if screen_time > 7:
    risk += 30

if study_hours > 8:
    risk += 20

if cgpa < 6:
    risk += 20

st.subheader("🎯 Burnout Risk Score")

if risk < 30:
    st.success("🟢 Low Burnout Risk")

elif risk < 60:
    st.warning("🟡 Medium Burnout Risk")

else:
    st.error("🔴 High Burnout Risk")

# =========================
# RECOMMENDATIONS
# =========================

st.subheader("🤖 Recommendations")

if risk < 30:

    st.success("""
    • Maintain your current routine  
    • Continue exercising  
    • Keep healthy sleep habits
    """)

elif risk < 60:

    st.warning("""
    • Improve sleep schedule  
    • Reduce screen time  
    • Take regular study breaks
    """)

else:

    st.error("""
    • Sleep 7–8 hours daily  
    • Reduce screen time  
    • Exercise regularly  
    • Manage stress effectively  
    • Consider speaking with a counselor
    """)

# =========================
# INSIGHTS
# =========================

st.subheader("📌 Key Insights")

st.info("""
• Higher screen time may increase burnout risk.  

• Better sleep habits support mental well-being.  

• Physical activity can help reduce stress.  

• Attendance and academic performance are linked with student wellness.
""")

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    "### 👨‍💻 Created by Zubair Ansari  \n"
    "Data Science Portfolio Project | Python • Streamlit • Machine Learning 🚀"
)