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
import helper_functions as hf  # Assuming your helper functions are in this module


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
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Sentiment', 'Tweet'])

# Function to add new tweet and update the dataset
def add_new_tweet(tweet_text, df):
    # Perform sentiment analysis on the new tweet
    sentiment = predict_sentiment(tweet_text)

    # Create a new row with the tweet and its sentiment
    new_row = pd.DataFrame({"Sentiment": [sentiment], "Tweet": [tweet_text], "Processed Text": [preprocess_text(tweet_text)]})

    # Append the new row to the dataframe
    updated_df = pd.concat([df, new_row], ignore_index=True)

    # Update the session state
    st.session_state.df = updated_df

    return updated_df

# Read the initial dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('tweets.csv')  # Replace 'tweets.csv' with your dataset filename
    return df


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


# Function to make the dashboard
def make_dashboard(tweet_df, bar_color):
    # Make 2 columns for the first row of the dashboard
    col1, col2 = st.columns([50, 50])
    with col1:
        # Plot the sentiment distribution
        sentiment_plot = hf.plot_sentiment(tweet_df)
        sentiment_plot.update_layout(height=350, title_x=0.5)
        st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)

    with col2:
        # Plot the top 10 occurring words
        top_unigram = hf.get_top_n_gram(tweet_df, ngram_range=(1, 1), n=10)
        unigram_plot = hf.plot_n_gram(
            top_unigram, title="Top 10 Occurring Words", color=bar_color
        )
        unigram_plot.update_layout(height=350)
        st.plotly_chart(unigram_plot, theme=None, use_container_width=True)

    # Make 2 columns for the second row of the dashboard
    col1, col2 = st.columns([50, 50])
    with col1:
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
    # Show a table with sentiment and tweet column
    if "df" in st.session_state:
        tweet_df = st.session_state.df
        st.dataframe(tweet_df[["Sentiment", "Tweet"]])
    else:
        st.write("No tweets to display.")

elif selected_tab == "Positive 😊":
    # Make dashboard for tweets with positive sentiment
    if "df" in st.session_state:
        tweet_df = st.session_state.df.query("Sentiment == 1")
        st.write("### Positive Sentiment Analysis")

        # Plot the number of positive tweets in a pie chart
        sentiment_counts = tweet_df['Sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts, names=['Positive'], title="Positive Sentiment Distribution")
        st.plotly_chart(fig)

        make_dashboard(tweet_df, bar_color="#1F77B4")
    else:
        st.write("No positive tweets to display.")

else:
    # Make dashboard for tweets with negative sentiment
    if "df" in st.session_state:
        tweet_df = st.session_state.df.query("Sentiment == 0")
        st.write("### Negative Sentiment Analysis")

        # Plot the number of negative tweets in a pie chart
        sentiment_counts = tweet_df['Sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts, names=['Negative'], title="Negative Sentiment Distribution")
        st.plotly_chart(fig)

        make_dashboard(tweet_df, bar_color="#FF7F0E")
    else:
        st.write("No negative tweets to display.")
