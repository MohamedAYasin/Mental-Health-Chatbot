import nltk
import numpy as np
import json
import pickle
import random
import tensorflow as tf
import streamlit as st
from streamlit_chat import message
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizerFast, TFBertModel  # Use Fast tokenizer
from tensorflow.keras.models import load_model

# Download NLTK data (only if not already downloaded)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load chatbot data (consider caching for better performance)
@st.cache_resource  # Cache the loaded data
def load_chatbot_data():
    with open('streamlit/health.json') as json_file:
        intents = json.load(json_file)
    with open('streamlit/words.pkl', 'rb') as file:
        words = pickle.load(file)
    with open('streamlit/classes.pkl', 'rb') as file:
        classes = pickle.load(file)
    model = load_model('streamlit/chatbotmodel.h5')
    return intents, words, classes, model

intents, words, classes, model = load_chatbot_data()


# Load BERT tokenizer & model (only once and cache)
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Use Fast tokenizer
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert_model

tokenizer, bert_model = load_bert_model()

# Function to get sentence embeddings (optimized)
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(inputs).last_hidden_state  # Directly access last_hidden_state
    return tf.reduce_mean(outputs, axis=1).numpy()


# ... (rest of the code - predict_class, get_response, Streamlit UI)

# Streamlit UI
st.set_page_config(page_title="Akira - Mental Health Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Akira - Mental Health Bot")
st.write("Feeling overwhelmed? I'm here to help. Type your message below.")

# ... (chat history and user input handling)

# Display chat history (optimized with key)
for i, (sender, msg) in enumerate(reversed(st.session_state['messages'])):
    message(msg, is_user=(sender == "You"), key=f"{sender}_{i}")  # Ensure unique keys

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:gray;">© 2025 Mohamed Ahmed Yasin - Stay Strong 💙</p>', unsafe_allow_html=True)
