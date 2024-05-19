#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Importing necessary libraries
import re
import joblib
import streamlit as st
import contractions
import pandas as pd
from datetime import datetime, timedelta 
import base64  # Add this line to import base64  
import os


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"https?\S+|www\.\S+|@[^\s]+|[^\w\s]|[\u0080-\uffff]", "", text)  # Remove links, mentions, non-ASCII characters, and punctuations
    text = contractions.fix(text)  # Expand contractions
    words = text.split()
    words = [word for word in words if len(word) > 2]  # Remove short words
    processed_text = ' '.join(words)
    return processed_text

# Load the SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess and vectorize text
def preprocess_and_vectorize(text):
    processed_text = preprocess_text(text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    return text_vector

# Function to make predictions
def predict_sentiment(text):
    text_vector = preprocess_and_vectorize(text)
    prediction = svm_model.predict(text_vector)
    return prediction[0]

def export_report(df):
    filename = f"sentiment_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
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
start_date_input = st.sidebar.date_input("Start Date")
end_date_input = st.sidebar.date_input("End Date")
max_date = datetime.now() - timedelta(days=30)

# Ensure start_date_input is a datetime object
start_date = datetime.combine(start_date_input, datetime.min.time()) if start_date_input else max_date
end_date = datetime.combine(end_date_input, datetime.max.time()) if end_date_input else datetime.now()

if st.sidebar.button("Export Report"):
    # Example DataFrame
    data = {'Text': ['Text 1', 'Text 2', 'Text 3'],
            'Sentiment': ['Positive', 'Negative'],
            'Date': [datetime.now(), datetime.now(), datetime.now()]}
    df = pd.DataFrame(data)

    # Get the file name for the exported report
    export_filename = export_report(df)

    # Provide a download link for the exported file
    with open(export_filename, "rb") as f:
        file_content = f.read()
    b64 = base64.b64encode(file_content).decode('utf-8')
    href = f'<a href="data:file/csv;base64,{b64}" download="{export_filename}">Click here to download the report</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Delete the temporary file
    os.remove(export_filename)
