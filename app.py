import streamlit as st
import json
import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from streamlit_chat import message

# **Set page configuration FIRST**
st.set_page_config(page_title="Akira - Mental Health Chatbot", page_icon="🧠", layout="centered")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load chatbot data once
@st.cache_resource
def load_chatbot_data():
    with open('streamlit/health.json') as json_file:
        intents = json.load(json_file)
    
    words = pickle.load(open('streamlit/words.pkl', 'rb'))
    classes = pickle.load(open('streamlit/classes.pkl', 'rb'))
    model = load_model('streamlit/chatbotmodel.h5')
    
    return intents, words, classes, model

intents, words, classes, model = load_chatbot_data()

# Preprocess user input
def clean_up_sentence(sentence):
    sentence_words = sentence.lower().split()
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert input into model format
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag).reshape(1, -1)

# Predict intent
def predict_class(sentence):
    input_bow = bow(sentence, words)
    res = model.predict(input_bow)[0]
    
    ERROR_THRESHOLD = 0.25
    results = [(classes[i], r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    return sorted(results, key=lambda x: x[1], reverse=True)

# Get response
def get_response(predictions):
    if not predictions:
        return "I can't answer this question yet, please look for other resources."
    
    tag, confidence = predictions[0]
    
    if confidence < 0.5:
        return "I can't answer this question yet, please look for other resources."

    for intent in intents['intents']:
        if tag in intent['tags']:
            return random.choice(intent['responses'])
    
    return "I'm here to listen. Tell me more about how you're feeling."

# Streamlit UI
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
