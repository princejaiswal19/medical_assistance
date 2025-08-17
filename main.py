from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import json
import random

# =============================
# Load model and resources
# =============================
model = tf.keras.models.load_model("model-builder.keras")

with open("tokenizer_file.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("dl_label_encoder.pickle", "rb") as handle:
    label_encoder = pickle.load(handle)

with open("symptoms_intents.json", "r") as file:
    intents_data = json.load(file)

# =============================
# FastAPI App
# =============================
app = FastAPI()

class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to Symptom Intent Classifier API"}

@app.post("/predict")
def predict(query: Query):
    user_input = query.text.strip()
    if user_input == "":
        return {"error": "Empty input"}

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=128)

    # Model prediction
    prediction = model.predict(padded)
    class_index = np.argmax(prediction)
    intent = label_encoder.inverse_transform([class_index])[0]

    # Pick random response
    response = None
    for intent_obj in intents_data["intents"]:
        if intent_obj["tag"] == intent:
            response = random.choice(intent_obj["responses"])
            break

    return {
        "input": user_input,
        "predicted_intent": intent,
        "response": response
    }
