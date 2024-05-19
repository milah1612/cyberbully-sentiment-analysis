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

# Streamlit app configuration
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon=":bird:", layout="wide")

# Custom CSS for the app
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

# App title and header
st.title("Twitter Sentiment Analysis")
st.header("Analyze the sentiment of tweets in real-time")

# User input for sentiment analysis
user_input = st.text_area("Enter the tweet for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        try:
            result = sentiment_pipeline(user_input)
            sentiment = result[0]['label']
            st.write(f"**Sentiment:** {sentiment}")
            
            # Get current timestamp in specified timezone
            myt = pytz.timezone('Asia/Kuala_Lumpur')
            timestamp = datetime.now(myt).strftime("%Y-%m-%d %H:%M:%S")

            # Initialize session state for results
            if 'results' not in st.session_state:
                st.session_state.results = []

            # Append new result
            st.session_state.results.append({
                'User Input': user_input,
                'Sentiment': sentiment,
                'Timestamp': timestamp
            })

            # Clear the text area after analysis
            user_input = ""

        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
    else:
        st.write("Please enter some text.")

# Sidebar for exporting report
st.sidebar.title("Export Report")
start_date = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input("End date", datetime.now())

if st.sidebar.button("Export Report"):
    if 'results' in st.session_state:
        df = pd.DataFrame(st.session_state.results)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Filter results by selected date range
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

# Footer for the app
st.markdown(
    """
    <footer style='text-align: center; margin-top: 2rem;'>
        <hr>
        <p>Developed by Your Name. Powered by Streamlit.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
