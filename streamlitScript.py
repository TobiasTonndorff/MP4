# 1. Install Streamlit (if not already installed)
# !pip install streamlit

# 2. Import necessary libraries
import streamlit as st
import numpy as np
from joblib import load
import pandas as pd

# 3. Load the trained classification model
nb_model = load('nb_model.joblib')  # Replace 'nb_model.joblib' with the path to your trained Naive Bayes model file

# 4. Design the user interface
st.title('Employee Attrition Prediction')

# Create input fields for user data
age = st.number_input('Age', min_value=18, max_value=100, value=30, key='age_input')
monthly_income = st.number_input('Monthly Income', min_value=0, value=5000, key='monthly_income_input')
years_at_company = st.number_input('Years at Company', min_value=0, value=3, key='years_at_company_input')

# Input fields for other features
daily_rate = st.number_input('Daily Rate', min_value=0, value=500, key='daily_rate_input')
distance_from_home = st.number_input('Distance From Home', min_value=0, value=10, key='distance_from_home_input')
education_field_hr = st.checkbox('Human Resources', key='education_field_hr_checkbox')
education_field_ls = st.checkbox('Life Sciences', key='education_field_ls_checkbox')
education_field_mktg = st.checkbox('Marketing', key='education_field_mktg_checkbox')
education_field_md = st.checkbox('Medical', key='education_field_md_checkbox')
education_field_ot = st.checkbox('Other', key='education_field_ot_checkbox')
education_field_tech = st.checkbox('Technical Degree', key='education_field_tech_checkbox')
environment_satisfaction = st.slider('Environment Satisfaction', min_value=1, max_value=4, value=2, key='enviroment_satisfaction_slider')
job_involvement = st.slider('Job Involvement', min_value=1, max_value=4, value=2, key='job_involvement_slider')
job_level = st.slider('Job Level', min_value=1, max_value=5, value=2, key='job_level_slider')
job_satisfaction = st.slider('Job Satisfaction', min_value=1, max_value=4, value=2, key='job_satisfaction_slider')
monthly_rate = st.number_input('Monthly Rate', min_value=0, value=5000, key='monthly_rate_input')
hourly_rate = st.number_input('Hourly Rate', min_value=0, value=50, key='hourly_rate_input')
overtime = st.checkbox('Overtime', key='overtime_checkbox')
gender = st.radio('Gender', options=['Male', 'Female'], key='gender_radio')
percent_salary_hike = st.number_input('Percent Salary Hike', min_value=0, value=15, key='percent_salary_hike_input')
performance_rating = st.slider('Performance Rating', min_value=1, max_value=4, value=3, key='performance_rating_slider')
standard_hours = st.number_input('Standard Hours', min_value=0, value=80, key='standard_hours_input')
work_life_balance = st.slider('Work Life Balance', min_value=1, max_value=4, value=2, key='work_life_balance_slider')
years_in_current_role = st.number_input('Years in Current Role', min_value=0, value=3, key='years_in_current_role_input')
years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, value=3, key='years_since_last_promotion_input')


# conver gender into numerical
gender_numeric = 1 if gender == 'Male' else 0


# Add more input fields for other features

# 5. Preprocess the input data to match the format expected by the model
# Create a DataFrame with the input data
input_data = np.array([[age, monthly_income, years_at_company, daily_rate, distance_from_home,
                        int(education_field_hr), int(education_field_ls), int(education_field_mktg),
                        int(education_field_md), int(education_field_ot), int(education_field_tech),
                        environment_satisfaction, job_involvement, job_level, job_satisfaction,
                        monthly_rate, hourly_rate, int(overtime), gender_numeric, percent_salary_hike,
                        performance_rating, standard_hours, work_life_balance,
                        years_in_current_role, years_since_last_promotion]])

# 6. Create a button to trigger prediction
if st.button('Predict'):
    # 7. Use the model to make predictions
    prediction = nb_model.predict(input_data)
    
    # 8. Display the prediction result
    if prediction == 1:
        st.error('The employee is predicted to leave the company (Attrition: Yes)')
    else:
        st.success('The employee is predicted to stay in the company (Attrition: No)')

# 9. Deploy the application on localhost
if __name__ == '__main__':
    st.write('To run the application, use the following command in your terminal:')
    st.code('streamlit run streamlitScript.py')  # Replace 'streamlitScript.py' with the filename of your Streamlit script
