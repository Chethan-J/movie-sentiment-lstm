
import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Movie Review Sentiment AI",
    page_icon="🎬",
    layout="centered"
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = tf.keras.models.load_model(
    "movie_sentiment_lstm.keras"
)

vocab_size = 10000
max_length = 200


# ==============================
# LOAD IMDB WORD INDEX
# ==============================

word_index = imdb.get_word_index()

word_index = {
    word: index + 3
    for word, index in word_index.items()
}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2


# ==============================
# CONVERT REVIEW TO NUMBERS
# ==============================

def encode_review(text):

    words = text.lower().split()

    encoded = []

    for word in words:

        if word in word_index:

            index = word_index[word]

            if index < vocab_size:
                encoded.append(index)
            else:
                encoded.append(2)

        else:
            encoded.append(2)

    return encoded


# ==============================
# PREDICT SENTIMENT
# ==============================

def predict_sentiment(text):

    encoded = encode_review(text)

    padded = pad_sequences(
        [encoded],
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    probability = model.predict(
        padded,
        verbose=0
    )[0][0]

    if probability >= 0.5:

        sentiment = "Positive 😊"
        confidence = probability * 100

    else:

        sentiment = "Negative 😞"
        confidence = (1 - probability) * 100

    return sentiment, confidence


# ==============================
# WEBSITE UI
# ==============================

st.title("🎬 Movie Review Sentiment AI")

st.write(
    "Enter a movie review and the LSTM model "
    "will predict whether the sentiment is positive or negative."
)

st.divider()


review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: The movie was amazing and the acting was excellent!",
    height=150
)


if st.button(
    "🔍 Analyze Review",
    use_container_width=True
):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if "Positive" in sentiment:

            st.success(sentiment)

        else:

            st.error(sentiment)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )


st.divider()

st.subheader("About the Model")

st.write(
    """
    This application uses Natural Language Processing
    and a Long Short-Term Memory (LSTM) neural network
    to classify movie reviews as positive or negative.
    """
)

st.info(
    "Model: Embedding + LSTM + Dropout + Dense"
)
