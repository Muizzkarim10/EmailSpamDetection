from pathlib import Path

import joblib
from scipy.sparse import hstack


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# Load trained components
word_vectorizer = joblib.load(MODEL_DIR / "word_tfidf.pkl")
char_vectorizer = joblib.load(MODEL_DIR / "char_tfidf.pkl")
model = joblib.load(MODEL_DIR / "spam_classifier.pkl")


def predict_email(subject, body):
    """
    Predict whether an email is spam or ham.
    """

    # Combine email fields
    text = f"{subject} {body}"

    # Transform using trained vectorizers
    word_features = word_vectorizer.transform([text])
    char_features = char_vectorizer.transform([text])

    # Combine word + character features
    features = hstack([
        word_features,
        char_features
    ])

    # Get prediction probability
    spam_probability = model.predict_proba(features)[0][1]

    # Threshold
    if spam_probability >= 0.50:
        prediction = "SPAM"
    else:
        prediction = "HAM"

    # Confidence
    confidence = (
        spam_probability
        if prediction == "SPAM"
        else 1 - spam_probability
    )

    return prediction, spam_probability, confidence


if __name__ == "__main__":

    subject = "Congratulations! You won a prize!"

    sender = "winner@free-prizes.com"

    body = """
    Congratulations! You have been selected to receive
    $10,000. Click here immediately to claim your reward.
    """

    prediction, spam_probability, confidence = predict_email(
        subject,
        body
    )

    print("\n================================")
    print("       EMAIL SPAM DETECTOR")
    print("================================")

    print(f"\nPrediction:       {prediction}")
    print(f"Spam Probability: {spam_probability:.2%}")
    print(f"Confidence:       {confidence:.2%}")

    print("================================\n")