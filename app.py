#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re
import joblib
import streamlit as st
import contractions 
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz 


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


# Function to export report
def export_report(df, start_date, end_date):
    local_timezone = pytz.timezone('Asia/Kuala_Lumpur')
    current_time = datetime.now(local_timezone)
    filename = f"sentiment_report_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    df.to_csv(filename, index=False)
    return filename



# Streamlit app
st.title("Twitter Sentiment Analysis")
user_input = st.text_area("Enter the tweet for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        prediction = predict_sentiment(user_input)
        processed_text = preprocess_text(user_input)
        st.write(f"Processed Text: {processed_text}")
        
        # Check the prediction and handle it
        if prediction == 1:
            st.write("Sentiment: Positive")
        elif prediction == 0:
            st.write("Sentiment: Negative")
        else:
            st.write(f"Sentiment: {prediction}")
    else:
        st.write("Please enter some text.") 


# Sidebar for report export
st.sidebar.title("Report Export")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
max_date = datetime.now() - timedelta(days=30)
start_date = max(start_date, max_date)
end_date = min(end_date, datetime.now())
if st.sidebar.button("Export Report"):
    # Example DataFrame
    data = {'Text': ['Text 1', 'Text 2', 'Text 3'],
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Date': [datetime.now(), datetime.now(), datetime.now()]}
    df = pd.DataFrame(data)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    export_filename = export_report(df, start_date, end_date)
    st.sidebar.success(f"Report exported successfully as {export_filename}")
