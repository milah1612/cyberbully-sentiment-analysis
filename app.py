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
import io 
import plotly.graph_objects as go

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

# Function to load data from a URL
@st.experimental_singleton
def load_data(url):
    try:
        response = requests.get(url)  # Fixed the parameter name to url
        response.raise_for_status()  # Raise an HTTPError for bad status codes
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        df['Sentiment'] = df['tweet_text'].apply(predict_sentiment)
        df['Processed Text'] = df['tweet_text'].apply(preprocess_text) 
        # Save DataFrame to CSV
        df.to_csv("tweets.csv", index=False) 
        return df 
    except requests.HTTPError as e: 
        st.error(f"Failed to fetch data from URL: {url}. Error: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error  
    except Exception as ex:
        st.error(f"An unexpected error occurred: {ex}")
        return pd.DataFrame()  # Return empty DataFrame for any other exceptions   

# URL of the CSV file hosted on GitHub
csv_url = 'https://raw.githubusercontent.com/milah1612/cyberbully-sentiment-analysis/main/tweets.csv'   

# Load initial dataset into session state if it's empty
if "df" not in st.session_state or st.session_state.df.empty:
    st.session_state.df = load_data(csv_url)

def add_new_tweet(tweet_text, df):
    # Predict sentiment and preprocess text
    sentiment = predict_sentiment(tweet_text)
    processed_text = preprocess_text(tweet_text)
    
    # Create a new row DataFrame for the new tweet
    new_row = pd.DataFrame({"Sentiment": [sentiment], "tweet_text": [tweet_text], "Processed Text": [processed_text]})
    
    # Concatenate the new row with the existing DataFrame to create an updated DataFrame
    updated_df = pd.concat([df, new_row], ignore_index=True)
    
    # Update the session state DataFrame with the updated DataFrame
    st.session_state.df = updated_df

    return updated_df

def make_dashboard(tweet_df, bar_color):
    if tweet_df.empty or 'Sentiment' not in tweet_df:
        st.write("No data available to display.")
        return

    # Center-align all components
    st.markdown("<h1 style='text-align: center;'>Dashboard</h1>", unsafe_allow_html=True)

    # Display sentiment distribution 
    st.write("**Sentiment Label Count**")
    col1, col2 = st.columns(2)
    with col1:
        # Calculate sentiment counts and display in a DataFrame
        sentiment_counts = tweet_df['Sentiment'].value_counts()
        st.write(sentiment_counts) 

        # Create bar plot for sentiment distribution
        fig_bar = go.Figure(data=[go.Bar(x=sentiment_counts.index, y=sentiment_counts, marker_color=bar_color)])
        fig_bar.update_layout(title='Sentiment Distribution', xaxis_title='Sentiment', yaxis_title='Count')
        st.plotly_chart(fig_bar, use_container_width=True)    

    # Display top occurring words
    with col2:
        top_unigram = Counter(" ".join(tweet_df['Processed Text']).split()).most_common(10)
        if top_unigram:
            words = [item[0] for item in top_unigram]
            counts = [item[1] for item in top_unigram]
            fig_unigram = go.Figure(data=[go.Bar(x=words, y=counts, marker_color=bar_color)])
            fig_unigram.update_layout(title='Top 10 Ocurring Keyword', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#333"))
            st.plotly_chart(fig_unigram, use_container_width=False)
        else:
            st.write("No words to display.") 

    # Display top occurring bigrams
    col1, col2 = st.columns(2)
    with col1:
        bigrams = Counter([" ".join(item) for item in zip(tweet_df['Processed Text'].str.split().explode(), tweet_df['Processed Text'].str.split().explode().shift(-1)) if item[1] is not None]).most_common(10)
        if bigrams:
            bigram_words = [item[0] for item in bigrams]
            bigram_counts = [item[1] for item in bigrams]
            fig_bigram = go.Figure(data=[go.Bar(x=bigram_words, y=bigram_counts, marker_color=bar_color)])
            fig_bigram.update_layout(title='Top 10 Occurring Bigrams', height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#333"))
            st.plotly_chart(fig_bigram, use_container_width=False)
        else:
            st.write("No bigrams to display.")

    # Set the display width for DataFrame columns
    pd.options.display.max_colwidth = 1000  # Adjust the value as needed

    # Display sentiment and processed text table 
    st.write("**Sentiment and Processed Text**")
    st.dataframe(tweet_df[["Sentiment", "Processed Text"]])
 
# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Sentiment', 'Processed Text'])

# Load initial dataset into session state if it's empty
if st.session_state.df.empty:
    st.session_state.df = load_data(csv_url)

# Increase the font size of text inside the tabs
adjust_tab_font = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
}
</style>
"""
st.write(adjust_tab_font, unsafe_allow_html=True)

# Call make_dashboard function
make_dashboard(st.session_state.df, bar_color="#1F77B4")

# Load the CSS file
local_css("styles.css")

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
            st.sidebar.write(f"An error occurred: {e}")
    else:
        st.sidebar.write("Please enter some text.")
