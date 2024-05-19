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

# Streamlit app
# Twitter icon
st.sidebar.image("twitter_icon.png", width=200, output_format='png', use_column_width=False)  # Replace "twitter_icon.png" with the actual filename and path  

# Sidebar header
st.sidebar.title("TWITTER SENTIMENT ANALYSIS")
st.sidebar.write("This application performs sentiment analysis on the latest tweets based on the entered search term. The application can only predict positive or negative sentiment, and only English tweets are supported.")

# Add search parameter/tweet box
user_input = st.sidebar.text_area("Enter the search term or tweet for sentiment analysis:", height=200)
    
if st.sidebar.button("Analyze Sentiment"):
    if user_input:
        try:
            # Perform language detection
            detected_language = detect(user_input)
            if detected_language != 'en':  # Check if the detected language is not English
                st.write("Please enter text in English.")
            else:
                prediction = predict_sentiment(user_input)
                processed_text = preprocess_text(user_input)

                # Check the prediction and handle it
                if prediction == 1:
                    st.write("Sentiment: Positive")
                elif prediction == 0:
                    st.write("Sentiment: Negative")
                else:
                    st.write(f"Sentiment: {prediction}")
        except Exception as e:
            st.write("An error occurred. Please try again.")
    else:
        st.write("Please enter some text.")  

# Define navigation tabs
tabs = ["All", "Positive", "Negative"]
selected_tab = st.session_state.get("selected_tab", tabs[0])  # Initialize session state

# Render navigation tabs
tab_html = ""
for tab in tabs:
    tab_html += f'<li class="tab-item {"active" if tab == selected_tab else ""}" onclick="location.href=`#{tab.lower()}`">{tab}</li>'
st.markdown(f'<ul class="tabs">{tab_html}</ul>', unsafe_allow_html=True)

# Display corresponding content based on selected tab
if selected_tab == "All":
    st.write("All tweets will be displayed here.")
elif selected_tab == "Positive":
    st.write("Positive sentiment analysis will be displayed here.")
else:
    st.write("Negative sentiment analysis will be displayed here.")
