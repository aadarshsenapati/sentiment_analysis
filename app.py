import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Append NLTK data path if needed
# nltk.data.path.append('C:/Users/chdnv/AppData/Roaming/nltk_data')

api_key = "AIzaSyCedwduzFeJWNKvhLxu2SWgnnlrOBkm3Sc"  
youtube = build("youtube", "v3", developerKey=api_key)

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    contractions = {
        "don't": "do not", "can't": "cannot", "i'm": "i am", "you're": "you are",
        "it's": "it is", "they're": "they are", "we're": "we are", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not", "won't": "will not",
        "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
        "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "doesn't": "does not",
        "didn't": "did not", "mightn't": "might not", "mustn't": "must not"
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def get_comments(video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
    return comments

@st.cache_resource
def train_model():
    imdb_df = pd.read_excel("train.xlsx")
    imdb_df = imdb_df.dropna(subset=["Reviews", "Sentiment"]).copy()
    imdb_df["Sentiment"] = imdb_df["Sentiment"].map({"pos": 1, "neg": 0})
    imdb_df["Processed_Review"] = imdb_df["Reviews"].fillna("").apply(preprocess_text)

    X = imdb_df["Processed_Review"]
    y = imdb_df["Sentiment"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)
    hate_df = pd.read_csv("labeled_data.csv")
    hate_df["Processed_tweet"] = hate_df["tweet"].apply(preprocess_text)

    X_hate = hate_df["Processed_tweet"]
    y_hate = hate_df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X_hate, y_hate, test_size=0.2, random_state=42)

    hate_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LinearSVC(max_iter=10000))
    ])

    hate_model.fit(X_train, y_train)

    return model, vectorizer,hate_model

model, vectorizer,hate_model = train_model()

st.title("🎬 YouTube Comment Sentiment Analyzer")
video_id = st.text_input("Enter YouTube Video ID", "")

if st.button("Analyze") and video_id:
    with st.spinner("Fetching and analyzing comments..."):
        comments = get_comments(video_id)
        if comments:
            df = pd.DataFrame(comments, columns=["Comment"])
            df["Processed_Review"] = df["Comment"].fillna("").apply(preprocess_text)
            X_youtube_tfidf = vectorizer.transform(df["Processed_Review"])
            df["Sentiment"] = model.predict(X_youtube_tfidf)
            df["Sentiment"] = df["Sentiment"].map({1: "Positive", 0: "Negative"})
            df["Speech Type"] = hate_model.predict(df["Processed_Review"])            
            df["Speech Type"] = df["Speech Type"].map({
                0: "Hate Speech",
                1: "Offensive",
                2: "Normal"
            })

            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=df["Sentiment"], palette="coolwarm", ax=ax)
            st.pyplot(fig)

            st.subheader("😊 Top 10 Positive Comments")
            for i, comment in enumerate(df[df["Sentiment"] == "Positive"]["Comment"].head(10), start=1):
                st.write(f"{i}. {comment}")

            st.subheader("😠 Top 10 Negative Comments")
            for i, comment in enumerate(df[df["Sentiment"] == "Negative"]["Comment"].head(10), start=1):
                st.write(f"{i}. {comment}")
            st.subheader(" Hate, Offensive Comments Detected are:")
            st.write(df[df["Speech Type"] != "Normal"][["Comment", "Speech Type"]].head(5))
        else:
            st.warning("No comments found or unable to fetch comments.")
