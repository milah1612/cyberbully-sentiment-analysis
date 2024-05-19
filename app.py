#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import re
import joblib
import streamlit as st
import contractions   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io 
from langdetect import detect   
from collections import Counter


# Function to load CSS file
def local_css(file_path):
    with io.open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)  


# Load the CSS file
local_css("styles.css")  # Update with the actual filename    


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

# Function to check if text contains only English characters
def is_english(text):
    return all(ord(char) < 128 for char in text) 


# Initialize session state
if 'tweets' not in st.session_state:
    st.session_state.tweets = []  
    

# Streamlit app
# Sidebar configuration
st.sidebar.image("twitter_icon.png", width=200, output_format='png', use_column_width=False)  # Replace "twitter_icon.png" with the actual filename and path  
st.sidebar.title("TWITTER SENTIMENT ANALYSIS")
st.sidebar.write("This application performs sentiment analysis on the latest tweets based on the entered search term. The application can only predict positive or negative sentiment, and only English tweets are supported.")

# Add search parameter/tweet box
user_input = st.sidebar.text_area("Enter the search term or tweet for sentiment analysis:", height=200)

# Analyze sentiment button in the sidebar
if st.sidebar.button("Analyze Sentiment"):
    if user_input:
        try:
            # Perform language detection
            detected_language = detect(user_input)
            if detected_language != 'en':  # Check if the detected language is not English
                st.sidebar.write("Please enter text in English.")
            else:
                prediction = predict_sentiment(user_input)
                processed_text = preprocess_text(user_input)

                # Display the prediction result in the sidebar
                st.sidebar.subheader("Analysis Result")
                st.sidebar.write(f"Processed Text: {processed_text}")

                # Check the prediction and handle it
                if prediction == 1:
                    st.sidebar.write("Sentiment: Positive")
                elif prediction == 0:
                    st.sidebar.write("Sentiment: Negative")
                else:
                    st.sidebar.write(f"Sentiment: {prediction}")
        except Exception as e:
            st.sidebar.write("An error occurred. Please try again.")
    else:
        st.sidebar.write("Please enter some text.") 


# Add tabs for "All", "Positive", and "Negative" sentiments
tabs = st.button("All")
tabs_positive = st.button("Positive")
tabs_negative = st.button("Negative")

# Center the tabs
st.markdown(
    "<style>.btn-center { display: flex; justify-content: center; }</style>",
    unsafe_allow_html=True,
)
st.markdown('<div class="btn-center">', unsafe_allow_html=True)

# Determine which tab is active
if tabs:
    selected_tab = "All"
elif tabs_positive:
    selected_tab = "Positive"
elif tabs_negative:
    selected_tab = "Negative"
else:
    selected_tab = "All"  # Default to "All" if no tab is selected

# Display content based on selected tab
if selected_tab == "All":
    st.write("Content for All tab")
elif selected_tab == "Positive":
    st.write("Content for Positive tab")
elif selected_tab == "Negative":
    st.write("Content for Negative tab")

# End the centered div
st.markdown("</div>", unsafe_allow_html=True)
