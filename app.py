import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('model.pkl')
scaler = joblib.load('scaler (1).pkl')

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Student ki details bharo — Exam Score predict hoga!")

st.subheader("📚 Study Details")
hours = st.slider("Hours Studied per day", 1, 44, 20)
attendance = st.slider("Attendance %", 60, 100, 80)
sleep = st.slider("Sleep Hours", 4, 9, 7)
previous = st.slider("Previous Scores", 50, 100, 70)
tutoring = st.slider("Tutoring Sessions", 0, 8, 2)
physical = st.slider("Physical Activity (hours)", 0, 6, 3)

st.subheader("👨‍👩‍👧 Background Details")
parental = st.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
motivation = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
internet = st.selectbox("Internet Access", ['Yes', 'No'])
extra = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
disabilities = st.selectbox("Learning Disabilities", ['No', 'Yes'])
gender = st.selectbox("Gender", ['Male', 'Female'])
school = st.selectbox("School Type", ['Public', 'Private'])
peer = st.selectbox("Peer Influence", ['Positive', 'Neutral', 'Negative'])
income = st.selectbox("Family Income", ['Low', 'Medium', 'High'])
teacher = st.selectbox("Teacher Quality", ['Low', 'Medium', 'High'])
education = st.selectbox("Parental Education Level", ['High School', 'College', 'Postgraduate'])
distance = st.selectbox("Distance from Home", ['Near', 'Moderate', 'Far'])

if st.button("🎯 Predict Exam Score"):
    ord_map = {'Low': 0, 'Medium': 1, 'High': 2}
    bin_map = {'No': 0, 'Yes': 1}
    peer_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    dist_map = {'Near': 0, 'Moderate': 1, 'Far': 2}
    edu_map = {'High School': 0, 'College': 1, 'Postgraduate': 2}

    features = pd.DataFrame({
        'Hours_Studied': [hours],
        'Attendance': [attendance],
        'Parental_Involvement': [ord_map[parental]],
        'Access_to_Resources': [ord_map[resources]],
        'Extracurricular_Activities': [bin_map[extra]],
        'Sleep_Hours': [sleep],
        'Previous_Scores': [previous],
        'Motivation_Level': [ord_map[motivation]],
        'Internet_Access': [bin_map[internet]],
        'Tutoring_Sessions': [tutoring],
        'Family_Income': [ord_map[income]],
        'Teacher_Quality': [ord_map[teacher]],
        'Peer_Influence': [peer_map[peer]],
        'Physical_Activity': [physical],
        'Learning_Disabilities': [bin_map[disabilities]],
        'Parental_Education_Level': [edu_map[education]],
        'Distance_from_Home': [dist_map[distance]],
        'Gender': [1 if gender == 'Male' else 0],
        'School_Type_Public': [1 if school == 'Public' else 0],
        'Support_Score': [ord_map[parental] + ord_map[resources]]
    })

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.success(f"🎓 Predicted Exam Score: {prediction[0]:.1f} / 100")

    if prediction[0] >= 80:
        st.balloons()
        st.info("Excellent performance expected! 🌟")
    elif prediction[0] >= 70:
        st.info("Good performance expected! 👍")
    else:
        st.warning("Needs improvement! 📚")
