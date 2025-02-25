# Import Libraries
import streamlit as st
import torch
import random
import pickle
import json
from transformers import DistilBertTokenizer, DistilBertModel
from tensorflow.keras.models import load_model
from streamlit_chat import message

# Load tokenizer and BERT model using Torch
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load chatbot model & assets
with open('streamlit/health.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('streamlit/words.pkl', 'rb'))
classes = pickle.load(open('streamlit/classes.pkl', 'rb'))
model = load_model('streamlit/chatbotmodel.h5')

# Function to get sentence embeddings (optimized)
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Predict intent (optimized threshold)
def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]
    
    ERROR_THRESHOLD = 0.30  # Increased for better filtering
    results = [(classes[i], r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    return sorted(results, key=lambda x: x[1], reverse=True)

# Get response from intent
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
st.set_page_config(page_title="Akira - Mental Health Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Akira - Mental Health Bot")
st.write("Feeling overwhelmed? I'm here to help. Type your message below.")

# Limit stored chat history (Fixes Memory Issue)
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# User input
user_input = st.text_input("Your message...", key="input")
if st.button("Send") and user_input:
    st.session_state['messages'].append(("You", user_input))
    
    predictions = predict_class(user_input)
    bot_response = get_response(predictions)
    
    st.session_state['messages'].append(("Akira", bot_response))

    # Keep only the last 20 messages to prevent memory issues
    st.session_state['messages'] = st.session_state['messages'][-20:]

# Display chat history
for i, (sender, msg) in enumerate(reversed(st.session_state['messages'])):
    message(msg, is_user=(sender == "You"), key=f"{sender}_{i}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:gray;">© 2025 Mohamed Ahmed Yasin - Stay Strong 💙</p>', unsafe_allow_html=True)
