import pandas as pd
import streamlit as st
import pickle
import os  # Add this import to handle file existence check

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Pune', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Rajkot', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Cuttack', 'Kanpur', 'Mohali', 'Bengaluru']

# Check if the pickle file exists
if os.path.exists('pipe.pkl'):
    pipe = pickle.load(open('pipe.pkl', 'rb'))
else:
    st.error("Error: 'pipe.pkl' file not found. Make sure the file exists in the same directory as this script.")
    st.stop()

st.title('IPL WIN PREDICTOR')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Select Bowling Team', teams)

selected_city = st.selectbox('Select City', cities)
target = st.number_input('Target')

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('score')
with col4:
    overs = st.number_input("Over Completed")
with col5:
    wickets = st.number_input("Wickets Out")
if st.button('PREDICT'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
