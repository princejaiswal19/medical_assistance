import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import random

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model-builder.keras")

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer_file.pickle", "rb") as handle:
        return pickle.load(handle)

# Load label encoder
@st.cache_resource
def load_label_encoder():
    with open("dl_label_encoder.pickle", "rb") as handle:
        return pickle.load(handle)

# Load intents JSON
@st.cache_data
def load_intents():
    with open("symptoms_intents.json", "r") as file:
        return json.load(file)

# Initialize everything
model = load_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()
intents_data = load_intents()

# UI
st.title("🧠 Symptom Intent Classifier")
st.write("Enter your symptoms or a query to identify the intent.")

user_input = st.text_input("Your Input:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=128)
        prediction = model.predict(padded)
        class_index = np.argmax(prediction)
        intent = label_encoder.inverse_transform([class_index])[0]

        st.success(f"**Predicted Intent:** {intent}")

        # Show a response from JSON
        for intent_obj in intents_data["intents"]:
            if intent_obj["tag"] == intent:
                response = random.choice(intent_obj["responses"])
                st.info(f"💬 Response: {response}")
                break
