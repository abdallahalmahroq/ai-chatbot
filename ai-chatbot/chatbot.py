import json
import pickle
import random
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
INTENTS_PATH = BASE_DIR / "data" / "intents.json"
THRESHOLD = 0.40


def load_chatbot_assets():
    """Load trained model, vectorizer, and intent responses."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train.py` first to generate model.pkl and vectorizer.pkl."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    with open(INTENTS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}
    return model, vectorizer, responses


def get_reply(user_input, model, vectorizer, responses):
    """Predict the best intent and return a response."""
    user_input = user_input.strip()
    if not user_input:
        return "Please type something so I can help."

    features = vectorizer.transform([user_input])
    probabilities = model.predict_proba(features)[0]
    max_probability = np.max(probabilities)
    predicted_tag = model.classes_[np.argmax(probabilities)]

    if max_probability < THRESHOLD:
        return "Sorry, I didn’t understand that 🤔"

    return random.choice(responses[predicted_tag])


def main():
    model, vectorizer, responses = load_chatbot_assets()
    print("🤖 Chatbot is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() in {"quit", "exit"}:
            print("Bot: Goodbye 👋")
            break

        print("Bot:", get_reply(user_input, model, vectorizer, responses))


if __name__ == "__main__":
    main()
