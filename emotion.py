data = {
    "text": [
        "I am so happy today!",
        "This is the worst thing ever.",
        "I'm feeling sad and lonely.",
        "That was an amazing experience!",
        "Why would you say that?",
        "I'm furious right now.",
        "I can't stop smiling!",
        "This is frustrating."
    ],
    "emotion": [
        "joy",
        "anger",
        "sadness",
        "joy",
        "anger",
        "anger",
        "joy",
        "anger"
    ]
}

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.3)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

text = input("text:")
print("Predicted Emotion:", model.predict([text])[0])
print("Model accuracy:", model.score(X_test, y_test))