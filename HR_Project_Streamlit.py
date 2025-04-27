import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Advanced HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .metric-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .st-bq {
        border-left: 5px solid #4e79a7;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        background-image: none;
        color: #000000
    }
    .css-1v0mbdj {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_csv("HR_Employee_Enhanced.csv")

    # Decode numerical values to categorical where appropriate
    # (Assuming these mappings based on typical HR datasets)

    # Business Travel
    travel_map = {0: "Non-Travel", 1: "Travel_Rarely", 2: "Travel_Frequently"}
    df['BusinessTravel'] = df['BusinessTravel'].map(travel_map)

    # Department
    dept_map = {1: "Sales", 2: "Research & Development", 3: "Human Resources"}
    df['Department'] = df['Department'].map(dept_map)

    # Education Field
    edu_field_map = {
        0: "Other", 1: "Life Sciences", 2: "Medical",
        3: "Marketing", 4: "Technical Degree", 5: "Human Resources"
    }
    df['EducationField'] = df['EducationField'].map(edu_field_map)

    # Gender
    gender_map = {0: "Female", 1: "Male"}
    df['Gender'] = df['Gender'].map(gender_map)

    # Job Role
    job_role_map = {
        0: "Other", 1: "Sales Executive", 2: "Research Scientist",
        3: "Laboratory Technician", 4: "Manufacturing Director",
        5: "Healthcare Representative", 6: "Manager",
        7: "Sales Representative", 8: "Research Director"
    }
    df['JobRole'] = df['JobRole'].map(job_role_map)

    # Marital Status
    marital_map = {0: "Single", 1: "Married", 2: "Divorced"}
    df['MaritalStatus'] = df['MaritalStatus'].map(marital_map)

    # OverTime
    overtime_map = {0: "No", 1: "Yes"}
    df['OverTime'] = df['OverTime'].map(overtime_map)

    # Attrition
    attrition_map = {0: "No", 1: "Yes"}
    df['Attrition'] = df['Attrition'].map(attrition_map)

    # Convert normalized values back to original scales where possible
    # (These would need to be adjusted based on actual min/max values)
    df['Age'] = (df['Age'] * 20 + 30).round(0)  # Assuming age was normalized
    df['MonthlyIncome'] = (df['MonthlyIncome'] * 5000 + 2000).round(0)
    df['DistanceFromHome'] = (df['DistanceFromHome'] * 10 + 5).round(0)

    return df


df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Data Filters")

# Department filter
selected_department = st.sidebar.multiselect(
    "Select Department(s):",
    options=df["Department"].unique(),
    default=df["Department"].unique(),
    help="Filter by organizational department"
)

# Job Role filter
selected_job_role = st.sidebar.multiselect(
    "Select Job Role(s):",
    options=df["JobRole"].unique(),
    default=df["JobRole"].unique(),
    help="Filter by employee job role"
)

# Age range filter
age_range = st.sidebar.slider(
    "Select Age Range:",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=(int(df["Age"].min()), int(df["Age"].max())),
    help="Filter employees by age range"
)

# Attrition status filter
attrition_status = st.sidebar.multiselect(
    "Attrition Status:",
    options=df["Attrition"].unique(),
    default=df["Attrition"].unique(),
    help="Filter by attrition status"
)

# Monthly Income range filter
income_range = st.sidebar.slider(
    "Monthly Income Range ($):",
    min_value=int(df["MonthlyIncome"].min()),
    max_value=int(df["MonthlyIncome"].max()),
    value=(int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())),
    help="Filter by monthly income range"
)

# Job Level filter
job_level = st.sidebar.multiselect(
    "Job Level:",
    options=sorted(df["JobLevel"].unique()),
    default=sorted(df["JobLevel"].unique()),
    help="Filter by job level"
)

# Filter data based on selections
df_filtered = df[
    (df["Department"].isin(selected_department)) &
    (df["JobRole"].isin(selected_job_role)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Attrition"].isin(attrition_status)) &
    (df["MonthlyIncome"].between(income_range[0], income_range[1])) &
    (df["JobLevel"].isin(job_level))
    ]

# Main dashboard
st.title("📊 Advanced HR Analytics Dashboard")
st.markdown("""
This interactive dashboard provides comprehensive insights into employee attrition patterns, 
helping HR professionals identify risk factors and retention opportunities.
""")

# KPI cards
st.markdown("## 📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_employees = len(df_filtered)
    st.metric("Total Employees", f"{total_employees:,}")

with col2:
    attrition_count = df_filtered["Attrition"].value_counts().get("Yes", 0)
    st.metric("Attrition Count", attrition_count)

with col3:
    attrition_rate = (df_filtered["Attrition"].value_counts(normalize=True).get("Yes", 0) * 100).round(1)
    st.metric("Attrition Rate (%)", f"{attrition_rate}%")

with col4:
    avg_age = df_filtered["Age"].mean().round(1)
    st.metric("Average Age", avg_age)

with col5:
    avg_income = df_filtered["MonthlyIncome"].mean().round(0)
    st.metric("Avg Monthly Income", f"${avg_income:,}")

st.markdown("---")

# First row - Overview and Attrition Analysis
st.markdown("## 🔍 Attrition Overview")
col1, col2 = st.columns([1, 2])

with col1:
    # Attrition distribution pie chart
    fig = px.pie(df_filtered, names="Attrition", title="Attrition Distribution",
                 color="Attrition", color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"},
                 hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label',
                      marker=dict(line=dict(color='#FFFFFF', width=2)))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Attrition by multiple dimensions
    tab1, tab2, tab3, tab4 = st.tabs(["By Department", "By Job Role", "By Education", "By Age Group"])

    with tab1:
        # Attrition by Department
        dept_attrition = df_filtered.groupby(["Department", "Attrition"]).size().unstack().fillna(0)
        dept_attrition["Attrition Rate"] = (
                    dept_attrition["Yes"] / (dept_attrition["Yes"] + dept_attrition["No"]) * 100).round(1)
        dept_attrition = dept_attrition.sort_values("Attrition Rate", ascending=False)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=dept_attrition.index,
                y=dept_attrition["Yes"],
                name="Attrition Count",
                marker_color="#FF7F0E"
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=dept_attrition.index,
                y=dept_attrition["Attrition Rate"],
                name="Attrition Rate (%)",
                mode="lines+markers",
                line=dict(color="#2CA02C", width=3)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Attrition by Department",
            xaxis_title="Department",
            yaxis_title="Attrition Count",
            yaxis2_title="Attrition Rate (%)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Attrition by Job Role
        job_attrition = df_filtered.groupby(["JobRole", "Attrition"]).size().unstack().fillna(0)
        job_attrition["Attrition Rate"] = (
                    job_attrition["Yes"] / (job_attrition["Yes"] + job_attrition["No"]) * 100).round(1)
        job_attrition = job_attrition.sort_values("Attrition Rate", ascending=False)

        fig = px.bar(job_attrition, x=job_attrition.index, y="Yes",
                     title="Attrition by Job Role",
                     labels={"Yes": "Attrition Count", "index": "Job Role"},
                     color="Attrition Rate",
                     color_continuous_scale="OrRd")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Attrition by Education Field
        edu_attrition = df_filtered.groupby(["EducationField", "Attrition"]).size().unstack().fillna(0)
        edu_attrition["Attrition Rate"] = (
                    edu_attrition["Yes"] / (edu_attrition["Yes"] + edu_attrition["No"]) * 100).round(1)
        edu_attrition = edu_attrition.sort_values("Attrition Rate", ascending=False)

        fig = px.bar(edu_attrition, x=edu_attrition.index, y="Yes",
                     title="Attrition by Education Field",
                     labels={"Yes": "Attrition Count", "index": "Education Field"},
                     text="Attrition Rate",
                     color="Attrition Rate",
                     color_continuous_scale="Peach")
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Attrition by Age Group
        df_filtered["AgeGroup"] = pd.cut(df_filtered["Age"],
                                         bins=[20, 30, 40, 50, 60, 70],
                                         labels=["20-29", "30-39", "40-49", "50-59", "60+"])

        age_attrition = df_filtered.groupby(["AgeGroup", "Attrition"]).size().unstack().fillna(0)
        age_attrition["Attrition Rate"] = (
                    age_attrition["Yes"] / (age_attrition["Yes"] + age_attrition["No"]) * 100).round(1)

        fig = px.bar(age_attrition, x=age_attrition.index, y=["Yes", "No"],
                     title="Attrition by Age Group",
                     labels={"value": "Count", "index": "Age Group"},
                     barmode="stack",
                     color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Second row - Detailed Analysis
st.markdown("## 📊 Detailed Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Job Factors", "Satisfaction", "Advanced Insights"])

with tab1:
    # Demographics tab
    st.markdown("### Demographic Factors")
    col1, col2 = st.columns(2)

    with col1:
        # Age distribution by Attrition
        fig = px.histogram(df_filtered, x="Age", color="Attrition",
                           title="Age Distribution by Attrition",
                           nbins=20, barmode="overlay",
                           color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"},
                           opacity=0.7)
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Gender distribution by Attrition
        gender_attrition = df_filtered.groupby(["Gender", "Attrition"]).size().unstack()
        fig = px.bar(gender_attrition, x=gender_attrition.index, y=["Yes", "No"],
                     title="Attrition by Gender",
                     labels={"value": "Count", "index": "Gender"},
                     barmode="group",
                     color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

    # Marital Status and Business Travel
    col1, col2 = st.columns(2)

    with col1:
        # Marital Status distribution
        fig = px.sunburst(df_filtered, path=["MaritalStatus", "Attrition"],
                          title="Attrition by Marital Status",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Business Travel impact
        travel_attrition = df_filtered.groupby(["BusinessTravel", "Attrition"]).size().unstack()
        travel_attrition["Attrition Rate"] = (
                    travel_attrition["Yes"] / (travel_attrition["Yes"] + travel_attrition["No"]) * 100).round(1)

        fig = px.bar(travel_attrition, x=travel_attrition.index, y="Attrition Rate",
                     title="Attrition Rate by Business Travel Frequency",
                     labels={"Attrition Rate": "Attrition Rate (%)", "index": "Business Travel"},
                     text="Attrition Rate",
                     color="Attrition Rate",
                     color_continuous_scale="OrRd")
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Job Factors tab
    st.markdown("### Job-Related Factors")
    col1, col2 = st.columns(2)

    with col1:
        # Job Level vs Attrition
        fig = px.box(df_filtered, x="Attrition", y="JobLevel",
                     title="Job Level Distribution by Attrition",
                     color="Attrition",
                     color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Monthly Income vs Attrition
        fig = px.box(df_filtered, x="Attrition", y="MonthlyIncome",
                     title="Monthly Income Distribution by Attrition",
                     color="Attrition",
                     color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Years at Company vs Attrition
        fig = px.violin(df_filtered, x="Attrition", y="YearsAtCompany",
                        title="Years at Company by Attrition",
                        color="Attrition",
                        color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"},
                        box=True, points="all")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Overtime impact
        overtime_attrition = df_filtered.groupby(["OverTime", "Attrition"]).size().unstack()
        overtime_attrition["Attrition Rate"] = (
                    overtime_attrition["Yes"] / (overtime_attrition["Yes"] + overtime_attrition["No"]) * 100).round(1)

        fig = px.bar(overtime_attrition, x=overtime_attrition.index, y="Attrition Rate",
                     title="Attrition Rate by Overtime Status",
                     labels={"Attrition Rate": "Attrition Rate (%)", "index": "Overtime"},
                     text="Attrition Rate",
                     color="Attrition Rate",
                     color_continuous_scale="OrRd")
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Satisfaction tab
    st.markdown("### Employee Satisfaction Metrics")
    col1, col2 = st.columns(2)

    with col1:
        # Job Satisfaction
        fig = px.histogram(df_filtered, x="JobSatisfaction", color="Attrition",
                           title="Job Satisfaction by Attrition",
                           nbins=7, barmode="group",
                           color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Environment Satisfaction
        fig = px.histogram(df_filtered, x="EnvironmentSatisfaction", color="Attrition",
                           title="Environment Satisfaction by Attrition",
                           nbins=7, barmode="group",
                           color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Work-Life Balance
        fig = px.histogram(df_filtered, x="WorkLifeBalance", color="Attrition",
                           title="Work-Life Balance by Attrition",
                           nbins=7, barmode="group",
                           color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Relationship Satisfaction
        fig = px.histogram(df_filtered, x="RelationshipSatisfaction", color="Attrition",
                           title="Relationship Satisfaction by Attrition",
                           nbins=7, barmode="group",
                           color_discrete_map={"Yes": "#FF7F0E", "No": "#1F77B4"})
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Advanced Insights tab
    st.markdown("### Advanced Insights")

    # Correlation Analysis
    st.markdown("#### Correlation Heatmap")

    # Prepare data for correlation
    corr_df = df_filtered.copy()

    # Encode categorical variables
    le = LabelEncoder()
    cat_cols = ["Attrition", "BusinessTravel", "Department", "EducationField",
                "Gender", "JobRole", "MaritalStatus", "OverTime"]

    for col in cat_cols:
        corr_df[col] = le.fit_transform(corr_df[col])

    # Select numerical and encoded categorical columns
    corr_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
                 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                 'JobLevel', 'JobSatisfaction', 'EnvironmentSatisfaction',
                 'WorkLifeBalance', 'RelationshipSatisfaction', 'Attrition']

    corr_matrix = corr_df[corr_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False
    ))
    fig.update_layout(
        title="Correlation Matrix of Key Variables",
        xaxis_tickangle=-45,
        width=800,
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

    # Statistical Tests
    st.markdown("#### Statistical Significance Testing")

    # T-test for Monthly Income between Attrition groups
    yes_income = df_filtered[df_filtered["Attrition"] == "Yes"]["MonthlyIncome"]
    no_income = df_filtered[df_filtered["Attrition"] == "No"]["MonthlyIncome"]
    t_stat, p_value = stats.ttest_ind(yes_income, no_income, equal_var=False)

    st.markdown(f"""
    **Monthly Income Difference Between Attrition Groups:**
    - T-statistic: {t_stat:.2f}
    - P-value: {p_value:.4f}
    - {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'} at 95% confidence level
    """)

    # Chi-square test for categorical variables
    st.markdown("**Chi-square Tests for Categorical Variables:**")

    cat_vars = ["Department", "JobRole", "MaritalStatus", "OverTime"]

    for var in cat_vars:
        contingency_table = pd.crosstab(df_filtered["Attrition"], df_filtered[var])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        st.markdown(f"""
        - **{var} vs Attrition:**
          - Chi2: {chi2:.2f}
          - P-value: {p:.4f}
          - {'Statistically significant' if p < 0.05 else 'Not statistically significant'} at 95% confidence level
        """)

# Download filtered data
st.sidebar.markdown("---")
st.sidebar.markdown("### Export Data")
if st.sidebar.button("Download Filtered Data as CSV"):
    csv = df_filtered.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="filtered_employee_data.csv",
        mime="text/csv"
    )

# Data description
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Description")
st.sidebar.markdown("""
This dataset contains employee information including:
- Demographic details (Age, Gender, Marital Status)
- Job-related information (Department, Job Role, Level)
- Compensation (Monthly Income)
- Satisfaction metrics (Job, Environment, Work-Life Balance)
- Attrition status (Yes/No)
""")

# Hide Streamlit style
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
