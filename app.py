#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re
import joblib
import streamlit as st
import contractions  
import io 
from langdetect import detect  


# Function to load CSS file
def local_css(file_path):
    with io.open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
local_css("styles.css")  # Update with the actual filename    

# Twitter icon
st.image("twitter_icon.png", width=300, output_format='png', use_column_width=False)  # Replace "twitter_icon.png" with the actual filename and path  


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
user_input = st.text_area("Enter the tweet for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input: 
        # Perform language detection
        language = detect(user_input)
        if language == 'en':  # Only proceed if the text is detected as English
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
