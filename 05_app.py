import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

from warnings import simplefilter

simplefilter('ignore')

# ✅ Make the app full-width
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="wide")

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Cache model loading for efficiency
@st.cache_resource
def load_sentiment_model():
    return load_model('03_simple_rnn_imdb.h5')

model = load_sentiment_model()

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review, and our AI will classify it as **Positive** or **Negative**.')

# Text input area
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Autofill sample review button
if st.button('Use Sample Review'):
    st.session_state.user_input = "This movie was good! I don't know what else I can say it was very good."

user_input = st.text_area('📝 Enter your movie review below:', height=150, value=st.session_state.user_input)

# Classification button
if st.button('Classify'):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)[0][0]
            sentiment = '😊 Positive' if prediction > 0.5 else '😞 Negative'

        # Display result
        st.success(f'**Sentiment:** {sentiment}')
        st.metric(label="Confidence Score", value=f"{prediction:.2%}")
        st.progress(int(prediction * 100))
    else:
        st.warning("⚠️ Please enter a valid movie review.")

# Footer
st.markdown("---")
st.caption("🔍 AI-powered sentiment analysis using an RNN model trained on IMDB reviews.")
