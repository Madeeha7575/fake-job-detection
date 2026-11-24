import streamlit as st
import pandas as pd
import joblib
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
st.title("Fake Job Detection System")
job_description= st.text_area("Paste a Job Description")
if st.button("Analyze"):
    transformed_text= vectorizer.transform([job_description])
    prediction = model.predict(transformed_text)
    if prediction[0]==1:
     st.error("Fake Job Posting!")
    else:
        st.success("Real Job Posting!")