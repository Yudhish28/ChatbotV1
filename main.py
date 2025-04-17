import streamlit as st
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.data import find
from streamlit_chat import message

# Ensure NLTK resources are downloaded
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('wordnet')
try:
    nltk.download('punkt_tab')  # Optional/safe for your environment
except:
    pass  # Not required but included because it worked for you

# Load chatbot files
model = tf.keras.models.load_model("chatbotmodel.h5")
intents = json.load(open("breastCancer.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

# NLP Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array([bag])

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(p)[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents['intents']:
            if intent['tags'] == tag:
                responses = intent['responses']
                return responses if isinstance(responses, str) else random.choice(responses)
    return "Sorry, I didn't understand that."

# Streamlit Page Configuration
st.set_page_config(page_title="Breast Cancer Chatbot", layout="centered")

# Custom CSS for Dark UI
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    a {
        color: #cccccc !important;
    }
    .st-emotion-cache-1v0mbdj {
        background-color: #1e1e1e;
    }
    .stTextInput>div>div>input {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h2 style='text-align: center; color: #E91E63;'>🩺 Breast Cancer Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Hi, Feel free to ask me questions about breast cancer.</p>", unsafe_allow_html=True)

# Initialize Chat History
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat Input Field
user_input = st.chat_input("Type your question here...")

if user_input:
    response = get_response(predict_class(user_input))
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Display Chat History
for i, (sender, message_text) in enumerate(st.session_state.chat_history):
    if sender == "user":
        message(message_text, is_user=True, key=f"user_{i}")
    else:
        message(message_text, key=f"bot_{i}")
