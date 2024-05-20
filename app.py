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
import plotly.express as px 
from hf import plot_sentiment, get_top_n_gram


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


# Function to load the dataset
@st.cache
def load_data():
    df = pd.read_csv('tweets.csv')  # Replace with your dataset path
    return df

# Load initial dataset
if 'df' not in st.session_state or st.session_state.df.empty:
    st.session_state.df = load_data()
    st.session_state.df['Processed Text'] = st.session_state.df['tweet_text'].apply(preprocess_text)
    st.session_state.df['Sentiment'] = st.session_state.df['tweet_text'].apply(predict_sentiment)

# Initialize session state for tweets
if 'tweets' not in st.session_state:
    st.session_state.tweets = []  
    
# Streamlit app
# Sidebar configuration
st.sidebar.image("twitter_icon.png", width=200, output_format='png', use_column_width=False)  # Replace with actual filename and path  
st.sidebar.title("TWITTER SENTIMENT ANALYSIS")
st.sidebar.write("This application performs sentiment analysis on the latest tweets based on the entered search term. The application can only predict positive or negative sentiment, and only English tweets are supported.")

# Add search parameter/tweet box
user_input = st.sidebar.text_area("Enter the search term or tweet for sentiment analysis:", height=200)

# Analyze sentiment button in the sidebar
if st.sidebar.button("Analyze Sentiment"):
    if user_input:
        try:
            detected_language = detect(user_input)
            if detected_language != 'en':  # Check if the detected language is not English
                st.sidebar.write("Please enter text in English.")
            else:
                prediction = predict_sentiment(user_input)
                processed_text = preprocess_text(user_input)

                # Add new tweet to the dataset
                new_tweet = {'tweet_text': user_input, 'cyberbullying_type': 'unknown', 'Processed Text': processed_text, 'Sentiment': prediction}
                st.session_state.df = st.session_state.df.append(new_tweet, ignore_index=True)

                # Display the prediction result in the sidebar
                st.sidebar.subheader("Analysis Result")
                st.sidebar.write(f"Processed Text: {processed_text}")
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

# Function to make the dashboard
def make_dashboard(tweet_df, bar_color):
    # Make 2 columns for the first row of the dashboard
    col1, col2 = st.columns([50, 50])
    with col1:
        # Plot the sentiment distribution
        sentiment_plot = hf.plot_sentiment(tweet_df)
        sentiment_plot.update_layout(height=350, title_x=0.5)
        st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)

    # Make 2 columns for the second row of the dashboard
    col1, col2 = st.columns([50, 50])
    with col1:
        # Plot the top 10 occurring words 
        top_unigram = hf.get_top_n_gram(tweet_df, ngram_range=(1, 1), n=10)
        unigram_plot = hf.plot_n_gram(
            top_unigram, title="Top 10 Occurring Words", color=bar_color
        )
        unigram_plot.update_layout(height=350)
        st.plotly_chart(unigram_plot, theme=None, use_container_width=True)

    with col2:
        # Plot the top 10 occurring bigrams
        top_bigram = hf.get_top_n_gram(tweet_df, ngram_range=(2, 2), n=10)
        bigram_plot = hf.plot_n_gram(
            top_bigram, title="Top 10 Occurring Bigrams", color=bar_color
        )
        bigram_plot.update_layout(height=350)
        st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

# Increase the font size of text inside the tabs
adjust_tab_font = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
}
</style>
"""
st.write(adjust_tab_font, unsafe_allow_html=True)

# Get the current page URL and parameters
params = st.experimental_get_query_params()

# Set the default tab if no tab parameter is provided
selected_tab = params.get("tab", ["All"])[0]

# Create a radio button widget to select the sentiment tab
selected_tab = st.radio("Select sentiment:", ["All", "Positive 😊", "Negative ☹️"])

# Display content based on the selected tab
if selected_tab == "All":
    # Show the table with sentiment and tweet column
    if not st.session_state.df.empty:
        st.dataframe(st.session_state.df[['Sentiment', 'tweet_text']])
    else:
        st.write("No tweets to display.")

elif selected_tab == "Positive 😊":
    # Make dashboard for tweets with positive sentiment
    if not st.session_state.df.empty:
        tweet_df = st.session_state.df.query("Sentiment == 1")
        make_dashboard(tweet_df, bar_color="#1F77B4")
    else:
        st.write("No positive tweets to display.")

else:
    # Make dashboard for tweets with negative sentiment
    if not st.session_state.df.empty:
        tweet_df = st.session_state.df.query("Sentiment == 0")
        make_dashboard(tweet_df, bar_color="#FF7F0E")
    else:
        st.write("No negative tweets to display.")
