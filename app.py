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
st.title("Twitter Sentiment Analysis")
# Sidebar for export report
st.sidebar.header("Export Report")
start_date = st.sidebar.date_input("Start date", value=datetime.now() - timedelta(days=30), max_value=datetime.now() - timedelta(days=1))
end_date = st.sidebar.date_input("End date", value=datetime.now(), max_value=datetime.now())

if start_date and end_date:
    if (end_date - start_date).days > 30:
        st.sidebar.error("The date range cannot exceed 30 days.")
    else:
        if st.sidebar.button("Export Report"):
            report_df = generate_report_data(start_date, end_date)
            st.sidebar.download_button(
                label="Download CSV",
                data=report_df.to_csv(index=False),
                file_name=f"sentiment_report_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

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
    else:
        st.write("Please enter some text.")

# Custom CSS for background color and other styles
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("Sentiment Analysis Dashboard")
st.subheader("Analyze the sentiment of tweets in real-time")
