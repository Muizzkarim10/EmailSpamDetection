from pathlib import Path
import base64
import re
import email

import pandas as pd
import joblib
from scipy.sparse import hstack

from gmail_reader import BASE_DIR, get_latest_emails

#Thresholds
SPAM_THRESHOLD = 0.50
UNCERTAINTY_LOW = 0.40
UNCERTAINTY_HIGH = 0.60


# MODEL
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

word_vectorizer = joblib.load(
    MODEL_DIR / "word_tfidf.pkl"
)

char_vectorizer = joblib.load(
    MODEL_DIR / "char_tfidf.pkl"
)

model = joblib.load(
    MODEL_DIR / "spam_classifier.pkl"
)

# EMAIL PARSING
def decode_body(data):
    """Decode Gmail's base64url encoded body."""

    if not data:
        return ""

    try:
        decoded = base64.urlsafe_b64decode(
            data + "=" * (-len(data) % 4)
        )

        return decoded.decode(
            "utf-8",
            errors="replace"
        )

    except Exception:
        return ""


def extract_body(payload):
    """
    Extract plain-text body from a Gmail message.
    Handles multipart messages.
    """

    mime_type = payload.get("mimeType", "")

    # Simple plain-text email
    if mime_type == "text/plain":

        data = payload.get("body", {}).get("data")

        return decode_body(data)

    # Multipart email
    for part in payload.get("parts", []):

        part_mime = part.get("mimeType", "")

        if part_mime == "text/plain":

            data = part.get("body", {}).get("data")

            body = decode_body(data)

            if body:
                return body

        # Recursively check nested multipart sections
        if part.get("parts"):

            body = extract_body(part)

            if body:
                return body

    return ""


# HEADER EXTRACTION
def get_header(payload, name):

    headers = payload.get("headers", [])

    for header in headers:

        if header["name"].lower() == name.lower():

            return header["value"]

    return ""


# PREDICTION
def predict_email(subject, body):

    # IMPORTANT:
    # This must match training exactly.
    text = f"{subject} {body}"

    word_features = word_vectorizer.transform(
        [text]
    )

    char_features = char_vectorizer.transform(
        [text]
    )

    features = hstack([
        word_features,
        char_features
    ])

    spam_probability = model.predict_proba(
        features
    )[0][1]

    if spam_probability < UNCERTAINTY_LOW:
        prediction = "HAM"

    elif spam_probability > UNCERTAINTY_HIGH:
        prediction = "SPAM"

    else:
        prediction = "UNCERTAIN"

    return prediction, spam_probability


# MAIN

if __name__ == "__main__":

    emails = get_latest_emails(50)

    results = []

    for i, email_data in enumerate(emails, start=1):

        payload = email_data["payload"]

        sender = get_header(payload, "From")
        subject = get_header(payload, "Subject")
        body = extract_body(payload)

        prediction, probability = predict_email(
            subject,
            body
        )

        results.append({
            "#": i,
            "Sender": sender,
            "Subject": subject,
            "Probability": probability,
            "Prediction": prediction
        })

    results_df = pd.DataFrame(results)

    # Keep the table readable
    results_df["Sender"] = results_df["Sender"].apply(
        lambda x: x[:35] + "..." if len(x) > 35 else x
    )

    results_df["Subject"] = results_df["Subject"].apply(
        lambda x: x[:60] + "..." if len(x) > 60 else x
    )

    results_df["Probability"] = results_df["Probability"].apply(
        lambda x: f"{x:.2%}"
    )

    print("\n")
    print("=" * 120)
    print("EMAIL SPAM DETECTOR")
    print("=" * 120)

    print(results_df.to_string(index=False))

    print("=" * 120)

    spam_count = sum(
        r["Prediction"] == "SPAM"
        for r in results
    )

    ham_count = sum(
        r["Prediction"] == "HAM"
        for r in results
    )

    uncertain_count = sum(
        r["Prediction"] == "UNCERTAIN"
        for r in results
    )

    print("\nSUMMARY")
    print("-" * 40)
    print(f"Total emails: {len(results)}")
    print(f"HAM:          {ham_count}")
    print(f"SPAM:         {spam_count}")
    print(f"UNCERTAIN:    {uncertain_count}")

    print("\nNo emails were modified.")