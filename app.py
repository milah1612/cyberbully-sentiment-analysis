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
from langdetect import detect

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

if st.button("Analyze Sentiment") and user_input:
    # Detect the language of the input text
    language = detect(user_input)  

   # Add this line to display the detected language
    st.write(f"Detected Language: {language}")
    
    # Check if the language is English
    if language == 'en':
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

# Export report option
if st.button("Export Report"):
    # Example DataFrame
    data = {'Text': [processed_text],
            'Sentiment': ['Positive' if prediction == 1 else 'Negative'],
            'Date': [datetime.now()]}
    df = pd.DataFrame(data)
    export_filename = export_report(df)
    st.success(f"Report exported successfully as {export_filename}")

    # Provide a download link for the exported file
    st.markdown(f'<a href="data:file/csv;base64,{base64.b64encode(df.to_csv(index=False).encode()).decode()}" download="{export_filename}">Click here to download the report</a>', unsafe_allow_html=True) 
    
else:
    st.warning("Sorry, this tool currently supports only English language tweets.")
