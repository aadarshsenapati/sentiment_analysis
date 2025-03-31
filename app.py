import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Your YouTube API Key
API_KEY = "AIzaSyCedwduzFeJWNKvhLxu2SWgnnlrOBkm3Sc"

# Function to fetch comments
def get_comments(video_id, max_comments=100):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    )
    response = request.execute()
    
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    
    return comments

# Function for sentiment analysis
def analyze_sentiments(comments):
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    
    for comment in comments:
        score = sia.polarity_scores(comment)["compound"]
        
        if score > 0.05:
            sentiments.append("Positive")
        elif score < -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    
    return sentiments

# Streamlit UI
st.title("YouTube Sentiment Analysis")

video_input = st.text_input("Enter YouTube Video ID:")

if st.button("Analyze Sentiment"):
    if not video_input:
        st.warning("Please provide a Video ID/URL")
    else:
        video_id = video_input.split("v=")[-1] if "youtube.com" in video_input else video_input
        comments = get_comments(video_id)
        
        if not comments:
            st.error("No comments found or invalid video ID")
        else:
            sentiments = analyze_sentiments(comments)
            df = pd.DataFrame({"Comments": comments, "Sentiment": sentiments})
            
            # Plot sentiment distribution
            fig, ax = plt.subplots()
            df["Sentiment"].value_counts().plot(kind="bar", color=["red", "blue", "green"], ax=ax)
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
            
            # Display negative comments
            negative_comments = df[df["Sentiment"] == "Negative"]["Comments"].head(5)
            if not negative_comments.empty:
                st.subheader("Some Negative Comments:")
                for comment in negative_comments:
                    st.write(f"- {comment}")
