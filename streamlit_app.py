import os
import streamlit as st
import pandas as pd
import joblib
import uuid
import json
import google.auth.exceptions
from google.cloud import storage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
import streamlit.components.v1 as components
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Google Cloud Storage client using credentials from Streamlit secrets
#service_account_info = json.loads(json.dumps(dict(st.secrets["gcp_service_account"])))
service_account_info = st.secrets["gcp_service_account"]
st.write("service_account_info:", st.secrets["gcp_service_account"])
# Debug: Print the service_account_info to ensure it is correctly parsed
#st.write("Service Account Info:", service_account_info)

try:
    credentials = ServiceAccountCredentials.from_service_account_info(service_account_info)
    storage_client = storage.Client(credentials=credentials)
except google.auth.exceptions.GoogleAuthError as e:
    st.error(f"Error initializing Google Cloud Storage client: {e}")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# Debug: Print the bucket name to ensure it is correctly loaded
st.write("GCS_BUCKET_NAME:", GCS_BUCKET_NAME)

# Paystack API keys from .env file
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY")
st.write("PAYSTACK_PUBLIC_KEY:", PAYSTACK_PUBLIC_KEY)
#print(f"PAYSTACK_SECRET_KEY: {PAYSTACK_SECRET_KEY}")
# Use PAYSTACK_SECRET_KEY in your application logic
# Example: Make an API call using the secret key

# Access the environment variables
client_id = os.getenv("CLIENT_ID")
project_id = os.getenv("PROJECT_ID")
#st.write("PROJECT_ID:", st.secrets["client_secrets"]["json"]["project_id"])
auth_uri = os.getenv("AUTH_URI")
token_uri = os.getenv("TOKEN_URI")
auth_provider_x509_cert_url = os.getenv("AUTH_PROVIDER_X509_CERT_URL")
client_secret = os.getenv("CLIENT_SECRET")
redirect_uris = [os.getenv(f"REDIRECT_URI{i}") for i in range(1, 9) if os.getenv(f"REDIRECT_URI{i}")]


# OAuth 2.0 client credentials
# Load client secrets JSON from secrets.toml
client_secrets_json = st.secrets["client_secrets"]["json"]
st.write("CLIENT_SECRETS_FILE:", client_secrets_json)
# Parse JSON string into a Python dictionary
CLIENT_SECRETS_FILE = json.loads(client_secrets_json)

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
/* Navbar styling */
.navbar {
    background-color: #f8f9fa;
    padding: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.nav-link {
    font-size: 18px;
    text-decoration: none;
    color: #000;
    padding: 14px 16px;
    display: block;
    flex-grow: 1;
    text-align: center;
}

.nav-link:hover {
    background-color: #495057;
    color: #ffffff; 
}

/* Footer styling */
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

/* Instructions styling */
.instructions {
    font-size: small;
    color: grey;
    font-style: italic;
    margin-bottom: 10px;
}

/* Title styling with fade-in effect */
.title-container {
    display: flex;
    justify-content: center;
    width: 100%;
    margin: 0;
    padding: 0;
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
    width: 100%;
    margin: 0 auto;
    animation: fadeIn 2s ease-in-out; /* Fade-in effect */
}
            
/* Background UI for header */
.header-container {
background: linear-gradient(to right, rgba(106, 17, 203, 0.7), rgba(37, 117, 252, 0.7)); /* Gradient background */
    padding: 20px; /* Padding around the text */
    border-radius: 10px; /* Rounded corners */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
    text-align: center; /* Center align the text */
    color: white; /* White text color for contrast */
    font-family: 'Roboto', sans-serif; /* Clean and modern font */
    font-size: 2em; /* Larger font size for emphasis */
    font-weight: bold; /* Bold text */
    margin-bottom: 20px; /* Space below the header */
    animation: fadeIn 2s ease-in-out; /* Fade-in effect */
    text-align: center;
}

.header-container2 {
background: linear-gradient(to right, rgba(106, 17, 203, 0.4), rgba(37, 117, 252, 0.4)); /* Gradient background */
    padding: 20px; /* Padding around the text */
    border-radius: 10px; /* Rounded corners */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
    text-align: center; /* Center align the text */
    color: white; /* White text color for contrast */
    font-family: 'Roboto', sans-serif; /* Clean and modern font */
    font-size: 2em; /* Larger font size for emphasis */
    font-weight: bold; /* Bold text */
    margin-bottom: 20px; /* Space below the header */
    animation: fadeIn 2s ease-in-out; /* Fade-in effect */
    text-align: center;
}

.title::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('https://www.transparenttextures.com/patterns/asfalt-light.png');
    opacity: 0.6;
}

.tooltip {
    position: relative;
    display: inline-block;
    width: 100%;
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
    bottom: 125%;
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

/* Loading animation styling */
.loading-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
}

.loading-dots {
    display: flex;
    justify-content: center;
    align-items: center;
}

.loading-dots div {
    width: 12px;
    height: 12px;
    margin: 3px;
    border-radius: 50%;
    background-color: #fff;
    animation: loading 0.8s infinite alternate;
}

.loading-dots div:nth-child(2) {
    animation-delay: 0.2s;
}

.loading-dots div:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes loading {
    from { opacity: 0; }
    to { opacity: 1; }
}

.stApp {
    background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
    color: white;
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
    <div class="tooltip title-container">
        <div class="title">Spam SMS Detection App</div>
        <div class="tooltiptext">Please do not forget to leave the GitHub repo a star ⭐</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
#st.title('Spam SMS Detection App')


def page1():

    # Initialize session state for email data
    if 'email_data' not in st.session_state:
        st.session_state.email_data = None

    # Single Message Prediction Section
    st.header('Single Message Prediction')
    st.markdown('<p class="instructions">Enter a single message in the text area below and click "Predict Single Message" to determine if it is spam or not.</p>', unsafe_allow_html=True)
    single_message = st.text_area('Enter your message here:')
    if st.button('Predict Single Message'):
        with st.spinner('Processing...'):
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
        with st.spinner('Processing...'):
            data = pd.read_csv(uploaded_file)
            # Handle missing values (NaN) in 'message' column
            data['message'].fillna("", inplace=True)
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

    # Function to save token for a specific user
    def save_token(user_id, token_info):
        try:
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(f'tokens/{user_id}.json')
            blob.upload_from_string(json.dumps(token_info))
            st.success(f"Token for user {user_id} saved successfully.")
        except Exception as e:
            st.error(f"Error saving token for user {user_id}: {e}")
    
    # Function to load token for a specific user
    def load_token(user_id):
        try:
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(f'tokens/{user_id}.json')
            if blob.exists():
                return json.loads(blob.download_as_string())
            else:
                st.warning(f"Token for user {user_id} not found.")
                return None
        except Exception as e:
            st.error(f"Error loading token for user {user_id}: {e}")
            return None
    
    # Function to authenticate and get Gmail service
    def authenticate():
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        creds = None
    
        # Get user ID from cookies or create a new one
        if 'user_id' in st.session_state:
            user_id = st.session_state['user_id']
        else:
            user_id = str(uuid.uuid4())
            st.session_state['user_id'] = user_id
    
        # Check if the token file for the current user exists and load it
        token_info = load_token(user_id)
        if token_info:
            creds = ServiceAccountCredentials.from_authorized_user_info(token_info, SCOPES)
    
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_config(CLIENT_SECRETS_FILE, SCOPES)
                auth_url, _ = flow.authorization_url(prompt='consent')
    
                # Inject JavaScript to open the authorization URL automatically
                components.html(f"""
                    <script>
                        window.location.href = "{auth_url}";
                    </script>
                """, height=0)
    
                st.write("Please authorize the application in the newly opened tab.")
    
                # Capture the authorization response URL automatically
                auth_response_url = st.experimental_get_query_params().get('code')
    
                if auth_response_url:
                    try:
                        flow.fetch_token(code=auth_response_url)
                        creds = flow.credentials
                        save_token(json.loads(creds.to_json()), user_id)
                    except google.auth.exceptions.GoogleAuthError as e:
                        st.error(f"Error during authentication: {e}")
    
        return creds

    # Paystack Payment Integration
    def initialize_paystack_payment(email, amount, currency):
        url = "https://api.paystack.co/transaction/initialize"
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json",
        }

        # Convert amount to smallest currency unit
        if currency == "NGN":
            amount_in_smallest_unit = amount * 100  # Kobo
        elif currency == "USD":
            amount_in_smallest_unit = amount * 100  # Cents
        else:
            raise ValueError("Unsupported currency")

        payload = {
            "email": email,
            "amount": amount_in_smallest_unit,
            "currency": currency
        }
        response = requests.post(url, headers=headers, json=payload)

        # Print response for debugging
        st.write(response.json()) 

        return response.json()

    # Function to create a new customer
    def create_customer(email, first_name, last_name, phone=None):
        url = "https://api.paystack.co/customer"
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['data']['customer_code']
            else:
                st.error(f"Failed to create customer. Error: {response.json().get('message')}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to create customer. Error: {str(e)}")
            return None

    # Function to verify customer using BVN
    def verify_bvn(customer_code_or_email, country, account_number, bvn, bank_code, first_name, last_name):
        url = f"https://api.paystack.co/customer/{customer_code_or_email}/identification"
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "country": country,
            "type": "bank_account",
            "account_number": account_number,
            "bvn": bvn,
            "bank_code": bank_code,
            "first_name": first_name,
            "last_name": last_name
        }


        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f'Failed to verify BVN. Error: {str(e)}')
            return None
        
    # Function to set session state for customer details
    def set_customer_session(email, first_name, last_name, phone, customer_code):
        st.session_state['customer_email'] = email
        st.session_state['customer_first_name'] = first_name
        st.session_state['customer_last_name'] = last_name
        st.session_state['customer_phone'] = phone
        st.session_state['customer_code'] = customer_code

    # Initialize session state
    if 'customer_code' not in st.session_state:
        st.session_state['customer_code'] = ''
        st.session_state['customer_email'] = ''
        st.session_state['customer_first_name'] = ''
        st.session_state['customer_last_name'] = ''
        st.session_state['customer_phone'] = ''

    # Function to fetch emails from Gmail with pagination support
    def get_emails(folder='inbox', max_results=5):
        creds = authenticate()
        with st.spinner('Authenticating...'):
            try:
                service = build('gmail', 'v1', credentials=creds)
                email_data = []
                page_token = None
                
                # Fetch messages in a loop until max_results is reached or no more messages are available
                while True:
                    # Fetch messages with optional page token
                    results = service.users().messages().list(userId='me', labelIds=[folder.upper()], maxResults=max_results, pageToken=page_token).execute()
                    messages = results.get('messages', [])
                    
                    if not messages:
                        break  # No more messages to fetch
                    
                    for message in messages:
                        msg = service.users().messages().get(userId='me', id=message['id']).execute()
                        email_data.append({
                            'message': msg['snippet']
                        })
                        
                        if len(email_data) >= max_results:
                            break  # Stop fetching if max_results is reached
                    
                    if len(email_data) >= max_results:
                        break  # Stop fetching if max_results is reached
                    
                    if 'nextPageToken' in results:
                        page_token = results['nextPageToken']
                    else:
                        break  # No more pages to fetch
                
                # Process fetched emails into a DataFrame
                if not email_data:
                    st.write(f'No messages found in {folder.capitalize()}.')
                else:
                    return pd.DataFrame(email_data)
            
            except HttpError as error:
                st.error(f'An error occurred: {error}')
                return pd.DataFrame()

    # Email Fetch Section
    st.header('Email Fetch')
    st.markdown('<p class="instructions">Select a Gmail folder and fetch the latest emails for spam detection. You can download the fetched emails as a CSV file and use it for batch prediction.You can only fetch 10 emails for free, Payment is required for more emails</p>', unsafe_allow_html=True)

    # Select Gmail folder and fetch emails
    selected_folder = st.selectbox('Select Gmail Folder', ['INBOX', 'SPAM', 'SENT', 'CATEGORY_SOCIAL', 'CATEGORY_PROMOTIONS'])
    max_emails = st.number_input('Enter the number of emails to fetch', min_value=1, max_value=10, value=10, step=1)
    if max_emails > 10:
        st.write('A charge of $10 will apply for you to run spam detection on more emails.')

    if st.button('Fetch Emails'):
        if max_emails > 10:
            st.error('Please proceed with payment below for additional emails.')
        else:
            with st.spinner('Fetching emails...'):
                st.session_state.email_data = get_emails(folder=selected_folder.lower(), max_results=max_emails)
                if not st.session_state.email_data.empty:
                    st.write(st.session_state.email_data)
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
            with st.spinner('Processing...'):
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
                    st.write(st.session_state.email_data.head())
    else:
        st.write("No emails to display.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("<h2 class='header-container'>PAYMENT AND BVN VERIFICATION</h2>", unsafe_allow_html=True)
    st.markdown('<p class="instructions">Proceed with payment and BVN verification to fetch more mails to run Spam prediction on.</p>', unsafe_allow_html=True)
    # Payment Button for Additional Emails
    email = st.text_input("Enter your email for payment:")
    currency = st.selectbox("Choose your currency:", ["NGN", "USD"])

    if st.button('Pay to Fetch More Emails'):
        with st.spinner('Initializing payment...'):
            try:
                if currency == "NGN":
                    amount_to_charge = 15000  # 15000 Naira
                elif currency == "USD":
                    amount_to_charge = 10  # 10 Dollars
                else:
                    raise ValueError("Unsupported currency selected")
                
                payment_response = initialize_paystack_payment(email, amount_to_charge, currency)
                
                if payment_response.get('status'):
                    payment_url = payment_response['data']['authorization_url']
                    st.success('Payment initialized. Redirecting to payment page...')
                    # Inject JavaScript to open the payment URL
                    components.html(f"""
                        <script>
                            window.open("{payment_url}", "_blank");
                        </script>
                    """, height=0)
                else:
                    st.error('Failed to initialize payment. Please try again.')
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Section for creating a new customer
    st.header('Create New Customer:')
    st.markdown('<p class="instructions">Register as a Customer to make payment.</p>', unsafe_allow_html=True)
    email = st.text_input('Email', key='create_email')
    first_name = st.text_input('First Name', key='create_first_name')
    last_name = st.text_input('Last Name', key='create_last_name')
    phone = st.text_input('Phone (optional)', key='create_phone')

    if st.button('Create Customer'):
        if email and first_name and last_name:
                with st.spinner('Creating New Customer...'):
                    customer_code = create_customer(email, first_name, last_name, phone)
                    if customer_code:
                        set_customer_session(email, first_name, last_name, phone, customer_code)
                        st.success(f'Customer created successfully. Customer Code: {customer_code}')
        else:
            st.error('Please fill in all required fields to create a customer.')

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Section for verifying customer using BVN
    st.header('Verify Customer using BVN')

    email_verify = st.text_input('Email', st.session_state.get('customer_email', ''), key='verify_email')
    country = st.selectbox('Country', ['NG'], key='verify_country')  # Add more country codes as required
    account_number = st.text_input('Account Number', key='verify_account_number')
    bvn = st.text_input('BVN', key='verify_bvn')
    bank_code = st.text_input('Bank Code', key='verify_bank_code')
    first_name_verify = st.text_input('First Name', st.session_state.get('customer_first_name', ''), key='verify_first_name')
    last_name_verify = st.text_input('Last Name', st.session_state.get('customer_last_name', ''), key='verify_last_name')

    if st.button('Verify BVN', key='verify_bvn_button'):
        if email_verify and account_number and bvn and bank_code and first_name_verify and last_name_verify:
            with st.spinner('Verifying BVN...'):
                verification_response = verify_bvn(
                # st.session_state['customer_code'],
                    email_verify,
                    country,
                    account_number,
                    bvn,
                    bank_code,
                    first_name_verify,
                    last_name_verify
                )
                if verification_response:
                    if verification_response.get('status'):
                        st.success('BVN Verification In Progress.')
                        st.json(verification_response)  # Display the full JSON response
                    else:
                        st.error(f"BVN verification failed: {verification_response.get('message', 'Unknown error')}")
                else:
                    st.error('An error occurred while verifying the BVN.')
        else:
            st.error('Please complete all fields.')

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

def page2():
    # CSS for custom styling
    st.markdown(
        """
        <style>
        .spm-title-container {
            text-align: center;
            margin-bottom: 30px;
        }
        .bold-underline {
            font-weight: bold;
            text-decoration: underline;
        }
        .spm-title {
            font-size: 40px;
            font-weight: bold;
        }
        .spm-section-title {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .spm-content {
            font-size: 18px;
            margin: 20px;
        }
        .spm-icon {
            width: 40px;
            height: 40px;
            margin-right: 10px;
        }
        .spm-contact-link {
            font-size: 18px;
            color: blue;
        }
        .spm-resource-link {
            font-size: 18px;
            color: blue;
            display: block;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="spm-tooltip spm-title-container header-container2">
            <div class="spm-title bold-underline">Resources & Links</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="spm-section-title bold-underline">Resources Used in the Project</div>', unsafe_allow_html=True)
    html_content = """
    <div class="spm-content">
        This project integrates various resources to deliver a comprehensive spam SMS detection application. 
        Below are the resources and methodologies used:
        <ul>
            <li><strong><u>Model & Vectorizer:</u></strong><br> The spam detection model and vectorizer are stored in my GitHub repository. They were trained by me using a comprehensive dataset and are designed to efficiently classify spam messages.</li>
            <li><strong><u>Sample CSV Files:</u></strong><br> Sample CSV files used for testing and validation are also available in the repository. Users can download these to understand the structure and type of data used.</li>
            <li><strong><u>Paystack Payment Resources & API:</u></strong><br> Paystack's API is utilized for payment processing and BVN verification. Best practices are implemented to handle errors, secure sensitive data, and manage API responses effectively.</li>
            <li><strong><u>Google OAuth & Gmail API:</u></strong><br> Google OAuth is integrated to authenticate users securely, and the Gmail API is used to fetch user emails for analysis. Note that the app does not store any personal data.</li>
        </ul>
    </div>
    """
    # Render the HTML content
    st.markdown(html_content, unsafe_allow_html=True)

    st.markdown('<div class="spm-section-title bold-underline">GitHub Repositories</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spm-content">
            You can find all the code, models, and resources in my GitHub repositories:
            <a class="spm-resource-link" href="https://github.com/Agomzyemeka/Spam-SMS-Detection-App" target="_blank">Spam SMS Detection App Repository</a>
            This repository contains the notebook used to train the spam detection model, sample datasets, and the Streamlit app code.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="spm-section-title bold-underline">Payment Integration</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spm-content">
            The app uses Paystack for payment processing. The API integration ensures secure transactions and includes features for BVN verification to enhance user trust and security. Error handling mechanisms and best practices are followed to ensure smooth API interactions.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="spm-section-title bold-underline">Google OAuth & Gmail API</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spm-content">
            Integration with Google OAuth and Gmail API allows users to authenticate securely and fetch their emails for analysis. The app respects user privacy and does not store any personal data, adhering to best privacy practices.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="spm-section-title bold-underline">Contact & Links</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spm-content">
            If you are interested in creating a machine learning model for your personal use or to improve your business, feel free to reach out to me. Below are my contact details:
            <ul>
                <li>Email: <a class="spm-contact-link" href="mailto:emyagomoh54321@gmail.com">emyagomoh54321@gmail.com</a></li>
                <li>LinkedIn: <a class="spm-contact-link" href="https://www.linkedin.com/in/chukwuemeka-agomoh-68726524b/" target="_blank">Chukwuemeka Agomoh</a></li>
                <li>Github: <a class="spm-contact-link" href="https://github.com/Agomzyemeka" target="_blank">My Github Repo</a></li>
                <li>Calendly: <a class="spm-contact-link" href="https://calendly.com/agomzyemeka/30min" target="_blank">Schedule a meeting</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="spm-section-title bold-underline">Skills Demonstrated</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spm-content">
            Through this project, I have demonstrated skills in:
            <ul>
                <li>API Integration & Management</li>
                <li>Machine Learning & Model Training</li>
                <li>Web Development with Streamlit</li>
                <li>Payment Processing & Security</li>
                <li>OAuth Authentication & Data Privacy</li>
                <li>Robust Error Handling & Logging</li>
            </ul>
            These skills are crucial for any modern industry or firm looking to leverage AI and machine learning for business improvement. 
        </div>
        """,
        unsafe_allow_html=True
    )



def page3():
    st.title("Thank you for Testing My App")
    st.markdown("<h2 class='header-container'>Please Remeber to leave the GitHub repo a star ⭐</h2>", unsafe_allow_html=True)
    st.write("Please Remeber to leave the GitHub repo a star ⭐")

def main():
    # Create a sidebar with a selectbox for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Pages", ["Home", "Resources & Links"])

    # Show the selected page
    if page == "Home":
        page1()
    elif page == "Resources & Links":
        page2()

if __name__ == "__main__":
    main()


# Footer with centered text
st.markdown(
    """
    <footer>
        <p>&copy; 2024 Chukwuemeka Agomoh. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
