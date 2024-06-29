import os
import streamlit as st
import pandas as pd
import joblib
import os
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account


# Load credentials from 'credentials.json' or from environment variables
if os.path.exists('credentials.json'):
    credentials = service_account.Credentials.from_service_account_file('credentials.json')
else:
    credentials = service_account.Credentials.from_service_account_info({
        "type": "service_account",
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL")
    })


# Load the saved model and vectorizer
model_path = 'spam_classifier.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("Model or vectorizer not found. Please train and save the model first.")

# Streamlit UI
st.title('Spam SMS Detection App')

# Header and Navigation
st.markdown("""
<nav style="background-color: #f8f9fa; padding: 10px;">
    <ul style="list-style-type: none; margin: 0; padding: 0; overflow: hidden;">
        <li style="float: left;"><a href="#single-message-prediction" style="text-decoration: none; color: #000; padding: 14px 16px; display: block;">Single Message Prediction</a></li>
        <li style="float: left;"><a href="#batch-prediction" style="text-decoration: none; color: #000; padding: 14px 16px; display: block;">Batch Prediction</a></li>
        <li style="float: left;"><a href="#email-fetch" style="text-decoration: none; color: #000; padding: 14px 16px; display: block;">Email Fetch</a></li>
    </ul>
</nav>
""", unsafe_allow_html=True)

# Email Authentication
st.header('Email Authentication')
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
creds = None

if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

def get_emails(max_results=5):
    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=max_results).execute()
        messages = results.get('messages', [])
        email_data = []
        if not messages:
            st.write('No messages found.')
        else:
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                email_data.append({
                    'snippet': msg['snippet']
                })
        return email_data
    except HttpError as error:
        st.error(f'An error occurred: {error}')
        return []

st.header('Fetch Emails')
st.write('You can fetch the latest 5 emails for free. For more emails, a charge of $5 per 10 additional emails will apply.')
max_emails = st.number_input('Enter the number of emails to fetch', min_value=1, max_value=100, value=5, step=1)
if max_emails > 5:
    st.write('A charge of $5 will apply for every 10 additional emails.')

if st.button('Fetch Emails'):
    if max_emails > 5:
        st.error('Please proceed with the payment for additional emails.')
    else:
        email_data = get_emails(max_results=max_emails)
        email_df = pd.DataFrame(email_data)
        st.write(email_df)
        if st.button('Predict Fetched Emails'):
            email_predictions = model.predict(vectorizer.transform(email_df['snippet']))
            email_df['Prediction'] = email_predictions
            st.write(email_df)
            st.download_button(
                label="Download Predictions",
                data=email_df.to_csv(index=False).encode('utf-8'),
                file_name='email_predictions.csv',
                mime='text/csv',
            )

st.header('Single Message Prediction')
single_message = st.text_area('Enter your message here:')
if st.button('Predict Single Message'):
    if single_message:
        single_prediction = model.predict(vectorizer.transform([single_message]))
        if single_prediction[0] == 1:
            st.write('The message is Spam.')
        else:
            st.write('The message is Not Spam.')
    else:
        st.error('Please enter a message.')

st.header('Batch Prediction')
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'message' in data.columns:
        predictions = model.predict(vectorizer.transform(data['message']))
        data['Prediction'] = predictions
        st.write(data)
        st.download_button(
            label="Download Predictions",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv',
        )
    else:
        st.error("The uploaded file does not contain a 'message' column.")

st.markdown("This app was created by Chukwuemeka Agomoh")

# Footer
st.markdown("""
<footer style="background-color: #f8f9fa; padding: 10px; text-align: center;">
    <p>&copy; 2024 Chukwuemeka Agomoh. All rights reserved.</p>
</footer>
""", unsafe_allow_html=True)

