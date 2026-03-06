from pathlib import Path
import base64
import email
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# Read-only Gmail permission
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]


BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


def get_gmail_service():

    creds = None

    # Load existing OAuth token
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    # Refresh or request authorization
    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        # Save token for future runs
        TOKEN_FILE.write_text(creds.to_json())

    return build(
        "gmail",
        "v1",
        credentials=creds
    )


def get_latest_emails(max_results=10):

    service = get_gmail_service()

    # Get latest messages from inbox
    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=max_results
    ).execute()

    messages = results.get("messages", [])

    emails = []

    for message in messages:

        msg = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()

        emails.append(msg)

    return emails


if __name__ == "__main__":

    emails = get_latest_emails(10)

    print(f"\nFound {len(emails)} emails.\n")

    for i, email_data in enumerate(emails, start=1):

        print("=" * 60)
        print(f"EMAIL {i}")
        print("=" * 60)

        headers = email_data["payload"].get(
            "headers",
            []
        )

        for header in headers:

            if header["name"].lower() in [
                "subject",
                "from"
            ]:
                print(
                    f"{header['name']}: "
                    f"{header['value']}"
                )

        print()