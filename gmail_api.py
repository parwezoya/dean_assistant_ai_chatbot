import os.path
import base64
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import pandas as pd


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_emails():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)

    results = service.users().messages().list(userId='me', maxResults=50).execute()
    messages = results.get('messages', [])

    email_data = []

    for msg in messages:
        msg_id = msg['id']
        msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        headers = msg['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
        snippet = msg.get('snippet', '')

        
        parts = msg['payload'].get('parts', [])
        body = ''
        for part in parts:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data:
                    body += base64.urlsafe_b64decode(data).decode()

        email_data.append({
            'sender': sender,
            'subject': subject,
            'snippet': snippet,
            'body': body
        })

    return pd.DataFrame(email_data)

df = get_emails()
df.to_csv("gmail_emails.csv", index=False)
print(" Emails saved to gmail_emails.csv")
