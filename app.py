import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

model = tf.keras.models.load_model("movie_sentiment_bilstm.keras")

with open("movie_sentiment_tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_sentiment(review):
    cleaned_review = clean_text(review)

    sequence = tokenizer.texts_to_sequences([cleaned_review])

    padded_review = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    probability = model.predict(
        padded_review,
        verbose=0
    )[0][0]

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability * 100
    else:
        sentiment = "Negative"
        confidence = (1 - probability) * 100

    return sentiment, confidence


st.title("🎬 Movie Sentiment Analyzer")

st.write(
    "Analyze movie reviews using a Bidirectional LSTM "
    "sentiment classification model."
)


st.header("🔍 Single Review Analysis")

movie_name = st.text_input(
    "Movie Name",
    placeholder="Enter movie name"
)

review = st.text_area(
    "Movie Review",
    placeholder="Write your movie review here...",
    height=150
)

if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if movie_name.strip():
            st.write(f"**Movie:** {movie_name}")

        if sentiment == "Positive":
            st.success(f"**Sentiment: {sentiment}**")
        else:
            st.error(f"**Sentiment: {sentiment}**")

        st.write(f"**Confidence: {confidence:.2f}%**")


st.header("📊 Multiple Review Analysis")

st.write(
    "Enter multiple reviews, one review per line, "
    "to calculate the overall sentiment."
)

reviews_text = st.text_area(
    "Reviews",
    placeholder="Enter one review per line...",
    height=200
)

if st.button("📊 Analyze Overall Sentiment"):

    reviews = [
        review.strip()
        for review in reviews_text.split("\n")
        if review.strip()
    ]

    if len(reviews) == 0:
        st.warning("Please enter at least one review.")

    else:
        positive_count = 0
        negative_count = 0

        st.subheader("Individual Predictions")

        for i, review in enumerate(reviews, start=1):

            sentiment, confidence = predict_sentiment(review)

            if sentiment == "Positive":
                positive_count += 1
            else:
                negative_count += 1

            st.write(
                f"**Review {i}:** {sentiment} "
                f"({confidence:.2f}%)"
            )

        total_reviews = len(reviews)

        positive_percentage = (
            positive_count / total_reviews
        ) * 100

        negative_percentage = (
            negative_count / total_reviews
        ) * 100

        st.subheader("Overall Result")

        st.write(
            f"**Positive Reviews:** {positive_count} "
            f"({positive_percentage:.2f}%)"
        )

        st.write(
            f"**Negative Reviews:** {negative_count} "
            f"({negative_percentage:.2f}%)"
        )

        if positive_count > negative_count:
            st.success("🎬 Overall Sentiment: POSITIVE")

        elif negative_count > positive_count:
            st.error("🎬 Overall Sentiment: NEGATIVE")

        else:
            st.warning("🎬 Overall Sentiment: MIXED")
