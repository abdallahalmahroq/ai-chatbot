import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


def load_training_data():
    """Load patterns and labels from the intents file."""
    with open(INTENTS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    sentences = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(pattern)
            labels.append(intent["tag"])

    return sentences, labels


def train_model():
    """Train TF-IDF + Logistic Regression intent classifier."""
    sentences, labels = load_training_data()

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(sentences)

    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(VECTORIZER_PATH, "wb") as file:
        pickle.dump(vectorizer, file)

    print("✅ Model trained and saved successfully!")


if __name__ == "__main__":
    train_model()
