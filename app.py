import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import json
import random
import pickle
from tensorflow.keras.models import load_model

# ------------------- Loading Pre-trained Models and Assets -------------------

# Load the pre-trained BERT tokenizer and model (this is for embeddings)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load the saved Keras chatbot model
model = load_model('streamlit/chatbotmodel.h5')

# Load the intents dictionary 
with open("streamlit/health.json", "r") as f:
    intents_dict = json.load(f)

# Load the list of classes (tags)
with open('streamlit/words.pkl', 'rb') as f:
    classes = pickle.load(f)

# ------------------- Defining Helper Functions -------------------

# Function to generate BERT embedding (cached to save memory)
@st.cache_resource
def get_bert_embedding(sentence):
    """
    Generate a BERT embedding for a given sentence.
    """
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to predict the intent of a sentence (cached to save memory)
@st.cache_resource
def predict_class(sentence):
    """
    Predict the intent of an input sentence using the trained chatbot model.
    """
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]

    ERROR_THRESHOLD = 0.33
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    else:
        return []

# Function to get a response based on predicted intents
def get_response(intents_list, intents_json):
    """
    Retrieve a response based on the predicted intent.
    """
    if not intents_list:
        return "Sorry, I don't understand. Can you rephrase your question?"

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if tag in intent['tags']:
            return random.choice(intent['responses'])
    return "Sorry, I couldn't understand. Please ask me something else."

# ------------------- Building the Streamlit Interface -------------------

# Title and description
st.title("Mental Health Support Chatbot")
st.markdown("Ask me anything about mental health. I'm here to help and listen.")

# User input field
user_input = st.text_input("You:")

if user_input:
    user_text = user_input.lower().strip()

    # Get predictions from the model
    predicted_intents = predict_class(user_text)
    # Get the chatbot's response based on predicted intent
    response = get_response(predicted_intents, intents_dict)

    # Display the bot's response
    st.write(f"Akira: {response}")
