#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re
import joblib
import streamlit as st
import contractions  
import pandas as pd
from datetime import datetime, timedelta

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove links, mentions, non-ASCII characters, and punctuations
    text = re.sub(r"https?\S+|www\.\S+|@[^\s]+|[^\w\s]|[\u0080-\uffff]", "", text)
    # Expand contractions
    text = contractions.fix(text)
    # Remove short words (optional, depends on your use case)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    # Join words back into text
    processed_text = ' '.join(words)
    return processed_text

# Load the SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a function to preprocess and vectorize text
def preprocess_and_vectorize(text):
    processed_text = preprocess_text(text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    return text_vector

# Define a function to make predictions
def predict_sentiment(text):
    text_vector = preprocess_and_vectorize(text)
    prediction = svm_model.predict(text_vector)
    return prediction[0]

# Streamlit app
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon=":bird:", layout="wide")

# Custom CSS for background color and header
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

# Header
st.title("Twitter Sentiment Analysis")
st.header("Analyze the sentiment of tweets in real-time")

# Get user input
user_input = st.text_area("Enter the tweet for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        prediction = predict_sentiment(user_input)
        processed_text = preprocess_text(user_input)
        
        # Determine sentiment
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        # Display results
        st.write(f"**Processed Text:** {processed_text}")
        st.write(f"**Sentiment:** {sentiment}")
        
        # Store the result with timestamp in MYT
        myt = pytz.timezone('Asia/Kuala_Lumpur')
        timestamp = datetime.now(myt).strftime("%Y-%m-%d %H:%M:%S")

        # Simulate storing results in a list (to be replaced with a database in a real app)
        if 'results' not in st.session_state:
            st.session_state.results = []

        st.session_state.results.append({
            'User Input': user_input,
            'Processed Text': processed_text,
            'Sentiment': sentiment,
            'Timestamp': timestamp
        })
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

        # Filter by date range
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

# Footer
st.markdown(
    """
    <footer style='text-align: center; margin-top: 2rem;'>
        <hr>
        <p>Developed by Your Name. Powered by Streamlit.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
