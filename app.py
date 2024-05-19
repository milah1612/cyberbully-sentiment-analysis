#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import pytz 

# Load pretrained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Streamlit app
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon=":bird:", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Twitter Sentiment Analysis")
st.header("Analyze the sentiment of tweets in real-time")

user_input = st.text_area("Enter the tweet for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        result = sentiment_pipeline(user_input)
        sentiment = result[0]['label']
        st.write(f"**Sentiment:** {sentiment}")
        
        myt = pytz.timezone('Asia/Kuala_Lumpur')
        timestamp = datetime.now(myt).strftime("%Y-%m-%d %H:%M:%S")

        if 'results' not in st.session_state:
            st.session_state.results = []

        st.session_state.results.append({
            'User Input': user_input,
            'Sentiment': sentiment,
            'Timestamp': timestamp
        })
    else:
        st.write("Please enter some text.")

st.sidebar.title("Export Report")
start_date = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input("End date", datetime.now())

if st.sidebar.button("Export Report"):
    if 'results' in st.session_state:
        df = pd.DataFrame(st.session_state.results)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        mask = (df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))
        export_df = df.loc[mask]

        if not export_df.empty:
            st.sidebar.download_button(
                label="Download CSV",
                data=export_df.to_csv(index=False),
                file_name=f"sentiment_report_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )
        else:
            st.sidebar.write("No data available for the selected date range.")
    else:
        st.sidebar.write("No sentiment analysis results available to export.")

st.markdown(
    """
    <footer style='text-align: center; margin-top: 2rem;'>
        <hr>
        <p>Developed by Your Name. Powered by Streamlit.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
