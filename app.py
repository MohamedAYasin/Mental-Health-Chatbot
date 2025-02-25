# import Libraries
import nltk
import numpy as np
import json
import pickle
import random
import tensorflow as tf
import streamlit as st
from streamlit_chat import message
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.models import load_model

# Download required NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load chatbot data
with open('streamlit/health.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('streamlit/words.pkl', 'rb'))
classes = pickle.load(open('streamlit/classes.pkl', 'rb'))
model = load_model('streamlit/chatbotmodel.h5')

# Load BERT tokenizer & model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to get sentence embeddings
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(inputs)
    return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()

# Predict intent
def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]
    
    # Set confidence threshold
    ERROR_THRESHOLD = 0.20
    results = [(classes[i], r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    return sorted(results, key=lambda x: x[1], reverse=True)

# Get response based on intent
def get_response(predictions):
    if not predictions:  # If no confident prediction, return fallback message
        return "I can't answer this question yet, please look for other resources."
    
    tag, confidence = predictions[0]  # Get best prediction
    
    # If confidence is low, fallback response
    if confidence < 0.5:
        return "I can't answer this question yet, please look for other resources."

    # Get response from the intents dataset
    for intent in intents['intents']:
        if tag in intent['tags']:
            return random.choice(intent['responses'])
    
    return "I'm here to listen. Tell me more about how you're feeling."

# Streamlit UI
st.set_page_config(page_title="Akira - Mental Health Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Akira - Mental Health Bot")
st.write("Feeling overwhelmed? I'm here to help. Type your message below.")

# Chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# User input
user_input = st.text_input("Your message...", key="input")
if st.button("Send") and user_input:
    st.session_state['messages'].append(("You", user_input))
    predictions = predict_class(user_input)
    bot_response = get_response(predictions)
    st.session_state['messages'].append(("Akira", bot_response))

# Display chat history with unique keys
for i, (sender, msg) in enumerate(reversed(st.session_state['messages'])):
    message(msg, is_user=(sender == "You"), key=f"{sender}_{i}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:gray;">© 2025 Mohamed Ahmed Yasin - Stay Strong 💙</p>', unsafe_allow_html=True)
