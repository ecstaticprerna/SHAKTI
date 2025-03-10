# SHAKTI
SHAKTI (Strategic Human Attrition Knowledge Through Insights): The Science of Workforce Stability is a data-driven project that analyzes employee attrition trends and predicts workforce turnover using machine learning. By using HR analytics, visualization, and predictive modeling, it provides actionable insights to enhance employee retention strategies and improve organizational stability. 

Employee attrition is a critical challenge for organizations, impacting productivity, morale, and operational costs. Retention360 is a comprehensive HR analytics project designed to predict employee turnover and provide actionable insights using machine learning. By leveraging the IBM HR Analytics Employee Attrition Dataset, this project identifies key factors influencing employee retention and helps HR professionals make data-driven decisions to improve workforce stability.

# Dataset Used
Dataset: IBM HR Analytics Employee Attrition Dataset (Link : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
Features:
Employee demographics (Age, Gender, Marital Status)
Job-related factors (Job Role, Department, Work Environment)
Compensation & Benefits (Salary, Stock Options)
Work-life balance (Job Satisfaction, Overtime)
Career growth (Years at Company, Training)
Performance indicators (Job Involvement, Performance Rating)
Target Variable: Attrition (Yes/No)

# Methodologies & Implementation Steps
The project follows a structured data science lifecycle

_1. Data Preprocessing & Cleaning_
Handling missing values and duplicate records
Converting categorical variables using Label Encoding & One-Hot Encoding
Normalizing numerical features using StandardScaler

_2. Exploratory Data Analysis (EDA) & Insights_
Understanding attrition trends across departments, salaries, and job roles
Visualizations using Matplotlib, Seaborn, and Power BI/Tableau
Correlation heatmaps to find relationships between features
Employee segmentation based on job satisfaction and workload

_3. Machine Learning Modeling_
Multiple models are trained and compared for performance to select the best one:
Baseline Model: Logistic Regression
Tree-Based Models: Random Forest, Decision Tree
Boosting Models: XGBoost
Advanced ML Models: Multi-Layer Perceptron, Support-Vector Machines
Hyperparameter Tuning using GridSearchCV for optimization
Model Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC Curve

_4. Feature Importance & Explainability_
SHAP (SHapley Additive Explanations) for feature importance analysis
Permutation importance to understand key attrition drivers

_5. Interactive Dashboard & Visualization_
Power BI / Tableau / Streamlit Dashboard
Key metrics like Attrition Rate, Department-wise Turnover, Salary-Attrition Trends
Model predictions visualized with probability scores

_Results & Insights (To Be Updated After Implementation)_
[Placeholder for Final Results: Will include key takeaways, top features affecting attrition, model performance comparison, and business recommendations.]

# Final Deliverables
Cleaned dataset (HR_Employee_Cleaned.csv)

EDA Report with visual insights (EDA_Report.pdf)

Trained ML model (Retention360_Model.pkl)

Dashboard for HR professionals (Power BI / Tableau / Streamlit)

Final Report & Presentation (PDF & PPT)
