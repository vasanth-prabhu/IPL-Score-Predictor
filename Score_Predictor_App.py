import streamlit as st
import numpy as np
import pandas as pd
import joblib

pipeline = joblib.load("best_ML_model_pipeline.joblib")

model = pipeline['model']
scaler = pipeline['scaler']
label_encoders = pipeline['label_encoders']
feature_cols = pipeline['feature_cols']

st.title("🏏 IPL Total Runs Predictor")

st.markdown("""
This app predicts the **total runs** in an IPL match given the match conditions.  
Select the inputs below and click **Predict**!
""")


encoded_inputs = []


venue = st.selectbox("Venue", label_encoders['venue'].classes_)
bat_team = st.selectbox("Batting Team", label_encoders['bat_team'].classes_)
bowl_team = st.selectbox("Bowling Team", label_encoders['bowl_team'].classes_)
striker = st.selectbox("Striker", label_encoders['batsman'].classes_)
bowler = st.selectbox("Bowler", label_encoders['bowler'].classes_)

runs = st.number_input("Current Runs", min_value=0, max_value=800, value=0)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=0)
overs = st.number_input("Overs Completed", min_value=0, max_value=20, value=0, step=1)
striker_ind = st.number_input("Striker Runs", min_value=0, value=0)

if st.button("Predict"):
    encoded_venue = label_encoders['venue'].transform([venue])[0]
    encoded_bat_team = label_encoders['bat_team'].transform([bat_team])[0]
    encoded_bowl_team = label_encoders['bowl_team'].transform([bowl_team])[0]
    encoded_striker = label_encoders['batsman'].transform([striker])[0]
    encoded_bowler = label_encoders['bowler'].transform([bowler])[0]

    input_features = [
        encoded_venue,
        encoded_bat_team,
        encoded_bowl_team,
        encoded_striker,
        encoded_bowler,
        runs,
        wickets,
        overs,
        striker_ind,
    ]

    input_df = pd.DataFrame([input_features], columns=feature_cols)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Total Runs: **{int(prediction)}**")
