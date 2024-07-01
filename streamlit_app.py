import os
import streamlit as st
import pandas as pd
import joblib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# OAuth 2.0 client credentials
CLIENT_SECRETS_FILE = "credentials.json"

# Load the saved model and vectorizer
model_path = 'best_spam_model_Support Vector Machine.pkl'
vectorizer_path = 'vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("Model or vectorizer not found. Please train and save the model first.")

# Custom CSS to adjust the width of the main container
st.markdown(
    """
    <style>
    @media (min-width: 768px) {
        .main .block-container {
            max-width: 95%;
            padding-left: 5%;
            padding-right: 5%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for styling
st.markdown("""
<style>
.navbar {
    background-color: #f8f9fa;
    padding: 10px;
    display: flex;
    justify-content: space-between; /* Distribute space between items */
    align-items: center;
    margin-bottom: 20px;
}

.nav-link {
    font-size: 18px;
    text-decoration: none;
    color: #000;
    padding: 14px 16px;
    display: block;
    flex-grow: 1; /* Allow links to grow and take available space */
    text-align: center; /* Center-align text within each link */
}
            

.nav-link:hover {
    background-color: #495057; /* Darker background on hover */
    color: #ffffff; 
}

footer {
    background-color: #f8f9fa;
    padding: 10px;
    text-align: center;
}

footer p {
    color: black;
}

section-divider {
    border-top: 1px solid #bbb;
    margin: 20px 0;
}

.instructions {
    font-size: small;
    color: grey;
    font-style: italic;
    margin-bottom: 10px;
}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
""", unsafe_allow_html=True)

# Add Font Awesome icons to the header
st.markdown("""
<nav class="navbar">
    <a class="nav-link" href="#single-message-prediction"><i class="fas fa-envelope-open-text"></i> Single Message Prediction</a>
    <a class="nav-link" href="#batch-prediction"><i class="fas fa-file-upload"></i> Batch Prediction</a>
    <a class="nav-link" href="#email-fetch"><i class="fas fa-mail-bulk"></i> Email Fetch</a>
</nav>
""", unsafe_allow_html=True)

# Custom CSS to style the title with a dark gradient background, texture, and tooltip
st.markdown(
    """
    <style>
    .title-container {
        display: flex;
        justify-content: center;
        width: 100%; /* Make container stretch to full width */
        margin: 0; /* Remove default margins */
        padding: 0; /* Remove default padding */
        position: relative;
    }

    .title {
        font-family: 'Roboto', sans-serif;
        font-size: 3em;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        border-radius: 0px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        position: relative;
        width: 100%; /* Make title stretch */
        margin: 0 auto; /* Center the title within the container */
    }

    .title::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('https://www.transparenttextures.com/patterns/asfalt-light.png');
        opacity: 0.6;  /* Adjust opacity to make the pattern more or less visible */
    }

    .tooltip {
        position: relative;
        display: inline-block;
        width: 100%; /* Make title stretch */
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(0, 0, 0, 0.8);
        color: #ffffff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position the tooltip above the text */
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: rgba(0, 0, 0, 0.8) transparent transparent transparent;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    .stApp {
        background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
        color: white;
    }
    </style>
    <div class="tooltip title-container">
        <div class="title">Spam SMS Detection App</div>
        <div class="tooltiptext">Please do not forget to leave the GitHub repo a star ⭐</div>
    </div>
    """,
    unsafe_allow_html=True
)


# Streamlit UI
#st.title('Spam SMS Detection App')

# Initialize session state for email data
if 'email_data' not in st.session_state:
    st.session_state.email_data = None

# Single Message Prediction Section
st.header('Single Message Prediction')
st.markdown('<p class="instructions">Enter a single message in the text area below and click "Predict Single Message" to determine if it is spam or not.</p>', unsafe_allow_html=True)
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

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Batch Prediction Section
st.header('Batch Prediction')
st.markdown('<p class="instructions">Upload a CSV file with a column named "message" to predict spam messages in bulk.</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'message' in data.columns:
        predictions = model.predict(vectorizer.transform(data['message']))
        data['Prediction'] = ['Spam' if pred == 1 else 'Not Spam' for pred in predictions]
        st.write(data)
        st.download_button(
            label="Download Predictions",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv',
        )
    else:
        st.error("The uploaded file does not contain a 'message' column.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Function to authenticate and get Gmail service
def authenticate():
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None

    # Check if token.json exists and load it, else authenticate with the client secrets file
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)  # Correctly using the redirect URI
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
    return creds

# Function to fetch emails from Gmail
def get_emails(folder='inbox', max_results=5):
    creds = authenticate()
    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', labelIds=[folder.upper()], maxResults=max_results).execute()
        messages = results.get('messages', [])
        email_data = []
        if not messages:
            st.write(f'No messages found in {folder.capitalize()}.')
        else:
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                email_data.append({
                    'message': msg['snippet']  # Extracting snippet as 'message' column
                })
        return pd.DataFrame(email_data)
    except HttpError as error:
        st.error(f'An error occurred: {error}')
        return pd.DataFrame()

# Email Fetch Section
st.header('Email Fetch')
st.markdown('<p class="instructions">Select a Gmail folder and fetch the latest emails for spam detection. You can download the fetched emails as a CSV file and use it for batch prediction.</p>', unsafe_allow_html=True)

# Select Gmail folder and fetch emails
selected_folder = st.selectbox('Select Gmail Folder', ['INBOX', 'SPAM', 'SENT'])
max_emails = st.number_input('Enter the number of emails to fetch', min_value=1, max_value=100, value=5, step=1)

if st.button('Fetch Emails'):
    st.session_state.email_data = get_emails(folder=selected_folder.lower(), max_results=max_emails)
    if not st.session_state.email_data.empty:
        st.write(st.session_state.email_data)
       # st.markdown("---")
        st.download_button(
            label="Download Emails as CSV",
            data=st.session_state.email_data.to_csv(index=False).encode('utf-8'),
            file_name='email_data.csv',
            mime='text/csv',
            help='You can use the downloaded CSV to test or run the batch prediction.',
        )
        st.markdown('<p class="instructions">You can use the downloaded CSV file to test or run the batch prediction.</p>', unsafe_allow_html=True)

if st.session_state.email_data is not None:
    st.markdown("---")
    if st.button('Predict Fetched Emails'):
        try:
            email_vectors = vectorizer.transform(st.session_state.email_data['message'])
            email_predictions = model.predict(email_vectors)
            st.session_state.email_data['Prediction'] = ['Spam' if pred == 1 else 'Not Spam' for pred in email_predictions]
            st.write(st.session_state.email_data)
            st.download_button(
                label="Download Predictions",
                data=st.session_state.email_data.to_csv(index=False).encode('utf-8'),
                file_name='email_predictions.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f'Error during prediction: {e}')
            st.write("Debug info:")
            st.write(st.session_state.email_data.head())  # Show a few rows of the dataframe for debugging
else:
    st.write("No emails to display.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Footer
st.markdown("""
<footer>
    <p>&copy; 2024 Chukwuemeka Agomoh. All rights reserved.</p>
</footer>
""", unsafe_allow_html=True)
# This app was created by Chukwuemeka Agomoh s