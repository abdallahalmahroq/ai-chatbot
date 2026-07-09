# AI Chatbot

A simple machine-learning chatbot that classifies user messages into predefined intents and returns a suitable response. The project uses **TF-IDF** for text vectorization and **Logistic Regression** for intent classification.

## Features

- Intent-based chatbot responses
- TF-IDF text vectorization
- Logistic Regression classifier
- Customizable `intents.json` dataset
- Command-line chatbot interface
- Retrainable model using `train.py`

## Project Structure

```text
ai-chatbot/
├── data/
│   └── intents.json
├── chatbot.py
├── train.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- scikit-learn
- NumPy
- JSON
- Pickle

## How It Works

1. Training sentences and tags are loaded from `data/intents.json`.
2. Text is converted into numerical features using `TfidfVectorizer`.
3. A `LogisticRegression` model learns the relationship between patterns and intent tags.
4. The chatbot predicts the intent of the user input.
5. If the confidence score is high enough, the chatbot returns a random response from the matched intent.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-chatbot.git
cd ai-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run this command if you update `data/intents.json` or want to regenerate the model files:

```bash
python train.py
```

This creates:

- `model.pkl`
- `vectorizer.pkl`

## Run the Chatbot

```bash
python chatbot.py
```

Example:

```text
🤖 Chatbot is ready! Type 'quit' to exit.

You: hello
Bot: Hi there!
```

## Dataset Example

The chatbot dataset is stored in `data/intents.json`:

```json
{
  "tag": "greeting",
  "patterns": ["hi", "hello", "hey"],
  "responses": ["Hello!", "Hi there!"]
}
```

## Future Improvements

- Add more intents and training examples
- Add a web interface using Flask
- Store chat history
- Improve confidence handling
- Support Arabic intents and responses

## Author

Abdullah Almahroq
