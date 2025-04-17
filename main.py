import streamlit as st
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.data import find

try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Load files
model = tf.keras.models.load_model("chatbotmodel.h5")
intents = json.load(open("breastCancer.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

# Helper functions
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
                # If responses is a string, return it; if it's a list, choose randomly
                responses = intent['responses']
                return responses if isinstance(responses, str) else random.choice(responses)
    return "Sorry, I didn't understand that."

# Streamlit interface
st.title("Breast Cancer Chatbot")

user_input = st.text_input("You:")

if user_input:
    intents_list = predict_class(user_input)
    response = get_response(intents_list)
    st.text_area("Bot:", value=response, height=100)
