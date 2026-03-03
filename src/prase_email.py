import os
import pandas as pd

from email import policy
from email.parser import BytesParser

# Dataset paths
DATA_DIR = "data"

HAM_FOLDERS = [
    "easy_ham/easy_ham",
    "hard_ham/hard_ham"
]

SPAM_FOLDERS = [
    "spam_2/spam_2"
]

# Parse one email
def parse_email(file_path):

    try:
        with open(file_path, "rb") as f:
            email = BytesParser(
                policy=policy.default
            ).parse(f)

        subject = email["Subject"]
        sender = email["From"]

        body = ""

        try:
            body_part = email.get_body(
                preferencelist=("plain", "html")
            )

            if body_part:
                body = body_part.get_content()

        except (LookupError, UnicodeDecodeError):

            # Fallback for emails with unknown/broken encodings
            payload = email.get_payload(decode=True)

            if payload:
                body = payload.decode(
                    "utf-8",
                    errors="replace"
                )

            else:
                raw_payload = email.get_payload()

                if isinstance(raw_payload, str):
                    body = raw_payload

        return {
            "subject": subject if subject else "",
            "sender": sender if sender else "",
            "body": body if body else ""
        }

    except Exception as e:

        print(f"Error parsing {file_path}: {e}")

        return None


# Process folders
emails = []


def process_folder(folder, label):

    folder_path = os.path.join(DATA_DIR, folder)

    for filename in os.listdir(folder_path):

        file_path = os.path.join(
            folder_path,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        result = parse_email(file_path)

        if result is not None:

            result["label"] = label

            emails.append(result)

# Parse ham
print("Parsing ham emails...")

for folder in HAM_FOLDERS:

    process_folder(
        folder,
        "ham"
    )

# Parse spam
print("Parsing spam emails...")

for folder in SPAM_FOLDERS:

    process_folder(
        folder,
        "spam"
    )

# Create DataFrame
df = pd.DataFrame(emails)

print("\nDataset created.")

print("Shape:", df.shape)

print("\nClass distribution:")

print(
    df["label"].value_counts()
)

# Save dataset
output_path = "data/emails.csv"

df.to_csv(
    output_path,
    index=False
)

print(
    f"\nDataset saved to: {output_path}"
)