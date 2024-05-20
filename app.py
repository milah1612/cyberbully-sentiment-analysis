#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import re
import joblib
import streamlit as st 
import requests
import contractions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from langdetect import detect
from collections import Counter
import plotly.express as px 
import pathlib  
import io



# Function to load CSS file
def local_css(file_path):
    with io.open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https?\S+|www\.\S+|@[^\s]+|[^\w\s]|[\u0080-\uffff]", "", text)
    text = contractions.fix(text)
    words = text.split()
    words = [word for word in words if len(word) > 2]
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
    st.session_state.df = pd.DataFrame(columns=['Sentiment', 'tweet_text', 'Processed Text'])

# URL of the CSV file hosted on GitHub
csv_url = 'https://raw.githubusercontent.com/milah1612/cyberbully-sentiment-analysis/main/tweets.csv' 

# Load initial dataset into session state
if st.session_state.df.empty:
    st.session_state.df = load_data(csv_url)

# Function to load data from a URL
@st.cache(allow_output_mutation=True)
def load_data(url):
    try:
        response = requests.get(url)
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_url)
        df['Sentiment'] = df['tweet_text'].apply(predict_sentiment)
        df['Processed Text'] = df['tweet_text'].apply(preprocess_text)
        return df 
    except requests.HTTPError as e: 
        st.error(f"Failed to fetch data from URL: {url}. Error: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error  


# Function to add new tweet and update the dataset
def add_new_tweet(tweet_text, df):
    sentiment = predict_sentiment(tweet_text)
    new_row = pd.DataFrame({"Sentiment": [sentiment], "tweet_text": [tweet_text], "Processed Text": [preprocess_text(tweet_text)]})
    updated_df = pd.concat([df, new_row], ignore_index=True)
    st.session_state.df = updated_df
    return updated_df

# Load the CSS file
local_css("styles.css")

# Streamlit app
# Sidebar configuration
st.sidebar.image("twitter_icon.png", width=200, output_format='png', use_column_width=False)
st.sidebar.title("TWITTER SENTIMENT ANALYSIS")
st.sidebar.write("This application performs sentiment analysis on the latest tweets based on the entered search term. The application can only predict positive or negative sentiment, and only English tweets are supported.")

# Add search parameter/tweet box
user_input = st.sidebar.text_area("Enter the search term or tweet for sentiment analysis:", height=200)

# Analyze sentiment button in the sidebar
if st.sidebar.button("Analyze Sentiment"):
    if user_input:
        try:
            detected_language = detect(user_input)
            if detected_language != 'en':
                st.sidebar.write("Please enter text in English.")
            else:
                prediction = predict_sentiment(user_input)
                processed_text = preprocess_text(user_input)

                # Display the prediction result in the sidebar
                st.sidebar.subheader("Analysis Result")
                st.sidebar.write(f"Processed Text: {processed_text}")

                if prediction == 1:
                    st.sidebar.write("Sentiment: Positive")
                elif prediction == 0:
                    st.sidebar.write("Sentiment: Negative")
                else:
                    st.sidebar.write(f"Sentiment: {prediction}")

                # Add the new tweet to the dataset
                add_new_tweet(user_input, st.session_state.df)
        except Exception as e:
            st.sidebar.write("An error occurred. Please try again.")
    else:
        st.sidebar.write("Please enter some text.")

# Function to make the dashboard
def make_dashboard(tweet_df, bar_color):
    col1, col2 = st.columns([50, 50])
    with col1:
        if not tweet_df.empty:
            sentiment_plot = px.histogram(tweet_df, x='Sentiment', color='Sentiment', title='Sentiment Distribution')
            sentiment_plot.update_layout(height=350, title_x=0.5)
            st.plotly_chart(sentiment_plot, use_container_width=True)
        else:
            st.write("No data available to display sentiment distribution.")

    with col2:
        if not tweet_df.empty:
            top_unigram = Counter(" ".join(tweet_df['Processed Text']).split()).most_common(10)
            if top_unigram:
                unigram_plot = px.bar(x=[item[0] for item in top_unigram], y=[item[1] for item in top_unigram], title="Top 10 Occurring Words", color_discrete_sequence=[bar_color])
                unigram_plot.update_layout(height=350)
                st.plotly_chart(unigram_plot, use_container_width=True)
            else:
                st.write("No words to display.")
        else:
            st.write("No data available to display top occurring words.")

    col1, col2 = st.columns([50, 50])
    with col1:
        if not tweet_df.empty:
            bigrams = Counter([" ".join(item) for item in zip(tweet_df['Processed Text'].str.split().explode(), tweet_df['Processed Text'].str.split().explode().shift(-1)) if item[1] is not None]).most_common(10)
            if bigrams:
                bigram_plot = px.bar(x=[item[0] for item in bigrams], y=[item[1] for item in bigrams], title="Top 10 Occurring Bigrams", color_discrete_sequence=[bar_color])
                bigram_plot.update_layout(height=350)
                st.plotly_chart(bigram_plot, use_container_width=True)
            else:
                st.write("No bigrams to display.")
        else:
            st.write("No data available to display top occurring bigrams.")

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
    if "df" in st.session_state and not st.session_state.df.empty:
        tweet_df = st.session_state.df
        st.dataframe(tweet_df[["Sentiment", "tweet_text"]])
    else:
        st.write("No tweets to display.")

elif selected_tab == "Positive 😊":
    if "df" in st.session_state and not st.session_state.df.empty:
        tweet_df = st.session_state.df.query("Sentiment == 1")
        st.write("### Positive Sentiment Analysis")
        make_dashboard(tweet_df, bar_color="#1F77B4")
    else:
        st.write("No positive tweets to display.")

else:
    if "df" in st.session_state and not st.session_state.df.empty:
        tweet_df = st.session_state.df.query("Sentiment == 0")
        st.write("### Negative Sentiment Analysis")
        make_dashboard(tweet_df, bar_color="#FF7F0E")
    else:
        st.write("No negative tweets to display.")
