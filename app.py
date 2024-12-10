import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import pandas as pd
from groq import Groq
import joblib
import numpy as np
import requests

# Load environment variables
load_dotenv()

# Set the API key environment variable globally
os.environ["GROQ_API_KEY"] = "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"

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
# Load the churn model
model = joblib.load('churn_model.joblib')
# Define the list of questions
questions = [
    "What is the customer's gender? (Male or Female)",
    "Is the customer a senior citizen? (1 for Yes, 0 for No)",
    "Is the customer married? (Yes or No)",
    "Does the customer have dependents? (Yes or No)",
    "How many months has the customer been with the company?",
    "Does the customer have phone service? (Yes or No)",
    "Is the customer a dual customer? (No phone service, No, Yes)",
    "What is the customer's internet service provider? (DSL, Fiber optic, or No)",
    "Does the customer have online security? (Yes, No, No internet service)",
    "Does the customer have online backup? (Yes, No, No internet service)",
    "Does the customer have device protection? (Yes, No, No internet service)",
    "Does the customer receive technical support? (Yes, No, No internet service)",
    "Does the customer have streaming TV? (Yes, No, No internet service)",
    "Does the customer have streaming movies? (Yes, No, No internet service)",
    "What type of contract does the customer have? (Month-to-month, One year, Two year)",
    "Does the customer have paperless billing? (Yes or No)",
    "What is the customer's payment method? (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))",
    "What is the customer's monthly charge amount?",
    "What is the customer's total charges amount?"
]
column_info = {
    "gender": "Gender of the customer. Possible values: 'Female', 'Male'.",
    "Senior_Citizen ": "Whether the customer is a senior citizen or not. Possible values: 0 (No), 1 (Yes).",
    "Is_Married": "Marital status of the customer. Possible values: 'Yes', 'No'.",
    "Dependents": "Whether the customer has dependents. Possible values: 'Yes', 'No'.",
    "tenure": "The number of months the customer has been with the company.",
    "Phone_Service": "Whether the customer has phone service. Possible values: 'Yes', 'No'.",
    "Dual": "Whether the customer has dual services (phone + internet). Possible values: 'No phone service', 'Yes', 'No'.",
    "Internet_Service": "The type of internet service the customer has. Possible values: 'DSL', 'Fiber optic', 'No'.",
    "Online_Security": "Whether the customer has online security. Possible values: 'Yes', 'No', 'No internet service'.",
    "Online_Backup": "Whether the customer has online backup. Possible values: 'Yes', 'No', 'No internet service'.",
    "Device_Protection": "Whether the customer has device protection. Possible values: 'Yes', 'No', 'No internet service'.",
    "Tech_Support": "Whether the customer receives technical support. Possible values: 'Yes', 'No', 'No internet service'.",
    "Streaming_TV": "Whether the customer has streaming TV service. Possible values: 'Yes', 'No', 'No internet service'.",
    "Streaming_Movies": "Whether the customer has streaming movies service. Possible values: 'Yes', 'No', 'No internet service'.",
    "Contract": "The type of contract the customer has. Possible values: 'Month-to-month', 'One year', 'Two year'.",
    "Paperless_Billing": "Whether the customer has paperless billing. Possible values: 'Yes', 'No'.",
    "Payment_Method": "The customer's payment method. Possible values: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'.",
    "Monthly_Charges": "The monthly charges for the customer's service.",
    "Total_Charges": "The total charges accumulated by the customer."
}
# Function to prepare the customer data for the model
def prepare_data_for_model(customer_data):
    """
    This function takes a dictionary of customer data and applies the required preprocessing
    steps (log transformation, label encoding, one-hot encoding) to prepare it for the model.
    
    :param customer_data: Dictionary containing raw customer data
    :return: Processed DataFrame ready for model prediction
    """
    # Ensure 'Total_Charges' is numeric, in case it's a string
    customer_data['Total_Charges'] = float(customer_data['Total_Charges'])

    # Step 1: Log transformation of 'Total_Charges'
    customer_data['Total_Charges'] = np.log(customer_data['Total_Charges'] + 1)

    # Step 2: Convert 'tenure' to int64 and 'Monthly_Charges' to float64
    customer_data['tenure'] = int(customer_data['tenure'])
    customer_data['Monthly_Charges'] = float(customer_data['Monthly_Charges'])

    # Step 3: Ensure 'Senior_Citizen ' is encoded as an integer (0 or 1)
    customer_data['Senior_Citizen '] = int(customer_data['Senior_Citizen '])

    # Step 4: Label Encoding (Mapping categorical columns to numeric)
    label_encoding_map = {
        'gender': {'Female': 0, 'Male': 1},
        'Is_Married': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'Phone_Service': {'Yes': 0, 'No': 1},  # Reversed mapping
        'Paperless_Billing': {'No': 0, 'Yes': 1}
    }

    # Apply label encoding based on the map and store the encoded values in the original data
    for column, mapping in label_encoding_map.items():
        if customer_data[column] in mapping:
            customer_data[column] = mapping[customer_data[column]]

    # Step 5: One-Hot Encoding
    one_hot_categories = {
        'Dual': ['Dual_No', 'Dual_No phone service', 'Dual_Yes'],
        'Internet_Service': ['Internet_Service_DSL', 'Internet_Service_Fiber optic', 'Internet_Service_No'],
        'Online_Security': ['Online_Security_No', 'Online_Security_No internet service', 'Online_Security_Yes'],
        'Online_Backup': ['Online_Backup_No', 'Online_Backup_Yes'],
        'Device_Protection': ['Device_Protection_No', 'Device_Protection_Yes'],
        'Tech_Support': ['Tech_Support_No', 'Tech_Support_Yes'],
        'Streaming_TV': ['Streaming_TV_No', 'Streaming_TV_Yes'],
        'Contract': ['Contract_Month-to-month', 'Contract_One year', 'Contract_Two year'],
        'Payment_Method': ['Payment_Method_Bank transfer (automatic)', 'Payment_Method_Credit card (automatic)',
                           'Payment_Method_Electronic check', 'Payment_Method_Mailed check'],
        'Streaming_Movies': ['Streaming_Movies_No', 'Streaming_Movies_Yes']
    }

    # Initialize new columns to 0
    for column, categories in one_hot_categories.items():
        for category in categories:
            customer_data[category] = 0  # Default value is 0 for all categories

        # Set the corresponding column to 1 based on the user's input
        value = customer_data[column]
        if column == 'Dual':
            if value == 'No phone service':
                customer_data['Dual_No phone service'] = 1
            elif value == 'Yes':
                customer_data['Dual_Yes'] = 1
            else:
                customer_data['Dual_No'] = 1
        elif column == 'Internet_Service':
            customer_data[f'Internet_Service_{value}'] = 1
        elif column == 'Online_Security':
            customer_data[f'Online_Security_{value}'] = 1
        elif column == 'Online_Backup':
            customer_data[f'Online_Backup_{value}'] = 1
        elif column == 'Device_Protection':
            customer_data[f'Device_Protection_{value}'] = 1
        elif column == 'Tech_Support':
            customer_data[f'Tech_Support_{value}'] = 1
        elif column == 'Streaming_TV':
            customer_data[f'Streaming_TV_{value}'] = 1
        elif column == 'Contract':
            customer_data[f'Contract_{value}'] = 1
        elif column == 'Payment_Method':
            customer_data[f'Payment_Method_{value}'] = 1
        elif column == 'Streaming_Movies':
            customer_data[f'Streaming_Movies_{value}'] = 1

    # Step 6: Drop the original columns before one-hot encoding
    columns_to_drop = [
        'Dual', 'Internet_Service', 'Online_Security', 'Online_Backup', 'Device_Protection',
        'Tech_Support', 'Streaming_TV', 'Contract', 'Payment_Method', 'Streaming_Movies', 'Online_Security_No internet service'
    ]
    
    # Drop original columns if they exist in the data
    for col in columns_to_drop:
        if col in customer_data:
            del customer_data[col]

    # Step 7: Convert to DataFrame and return
    processed_data = pd.DataFrame([customer_data])

    # Convert data types to match the model's expected input types
    processed_data[['gender', 'Is_Married', 'Dependents', 'Phone_Service', 'Paperless_Billing']] = \
        processed_data[['gender', 'Is_Married', 'Dependents', 'Phone_Service', 'Paperless_Billing']].astype('int32')

    float_columns = ['Dual_No', 'Dual_No phone service', 'Dual_Yes', 'Internet_Service_DSL',
                     'Internet_Service_Fiber optic', 'Internet_Service_No', 'Online_Security_No',
                     'Online_Security_Yes', 'Online_Backup_No',
                     'Online_Backup_Yes', 'Device_Protection_No', 'Device_Protection_Yes', 'Tech_Support_No',
                     'Tech_Support_Yes', 'Streaming_TV_No', 'Streaming_TV_Yes', 'Contract_Month-to-month',
                     'Contract_One year', 'Contract_Two year', 'Payment_Method_Bank transfer (automatic)',
                     'Payment_Method_Credit card (automatic)', 'Payment_Method_Electronic check',
                     'Payment_Method_Mailed check', 'Streaming_Movies_No', 'Streaming_Movies_Yes']

    processed_data[float_columns] = processed_data[float_columns].astype('float64')

    return processed_data

def get_response(df, prediction, column_info):
    if df.empty:
        return "No data available"

    # Format the customer data as specified
    def format_customer_data(df):
        # Assuming the first row contains the necessary data
        customer_data = df.iloc[0]
        return "\n".join([f"{col.replace('_', ' ').title()}: {customer_data[col]}" for col in df.columns])

    # Create formatted string from the DataFrame
    customer_data_str = format_customer_data(df)

    # Create the prompt using the formatted data string
    prompt = f"""
    Please explain why the customer with the following data is predicted to churn or not (Prediction: {'Churn' if prediction == 1 else 'No Churn'}):
    
    Collected Customer Data:
    {customer_data_str}

    Column Information:
    {column_info}

    Your explanation should be based on the data provided and how it affects the churn prediction.
    """
    
    # Set API URL and headers for Groq
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    # Prepare the data for the request
    payload = {
        "model": "llama3-8b-8192",  # or whatever model you're using
        "messages": [{"role": "user", "content": prompt}]
    }

    # Make the POST request to the API
    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to create the conversational chain
def create_chain():
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0
    )

    # Define the prompt template for interacting with the marketing team
    interaction_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "Welcome to the marketing assistant! I will be helping you collect information "
            "to classify churn for our clients.\n\n"
            "Let's start by gathering some details about the client.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Please answer the following question:\n{question}\n\n"
            "I will verify the answer and make sure it is correct. After that, we will proceed to the next question.\n\n"
        )
    )

    # Create the conversational chain with the assistant
    doc_chain = LLMChain(
        llm=llm,
        prompt=interaction_prompt
    )

    return doc_chain

# Streamlit App Configuration
st.set_page_config(
    page_title="Churn Classification Assistant 🤖",
    page_icon="🤖",
    layout="centered"
)

# def reset_process():
#     # Reset the session state variables
#     st.session_state.question_index = 0
#     st.session_state.client_info = {}
#     st.session_state.chat_history = []
#     st.write("Let's start over. I'll ask the questions again.")

st.title("Marketing Assistant for Churn Classification")

# Display default introductory message
st.markdown(""" 
    Welcome to the Marketing Assistant!  
    This chatbot is designed to help you gather essential information about clients for churn classification.  
    I will ask you a few questions to collect details about the customer, such as their gender, contract type, and other factors that may influence churn prediction.  
    Please provide accurate answers to each question to help improve the churn classification model.

    Let's begin by gathering some basic information.
""")

# Create conversational chain
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_chain()

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize the question index to keep track of where we are in the list
if "question_index" not in st.session_state:
    st.session_state.question_index = 0

# Initialize client info in session state (only once)
if "client_info" not in st.session_state:
    st.session_state.client_info = {}

# Function to export client_info to a DataFrame
def export_client_info_to_dataframe():
    client_info = st.session_state.client_info
    df = pd.DataFrame([client_info])
    # print("DataFrame:", df)
    return df

# Function to handle the input and store the answer
def handle_user_input(user_input):
    if not user_input:  # Check if the input is empty or None
        return "I didn't receive any input. Could you please provide an answer?"
    current_question = questions[st.session_state.question_index]
    client_info = st.session_state.client_info  # Accessing the session state's client_info
    
    # Normalize and verify the user's input for each question
    if current_question == questions[0]:  # Gender question
        if "male" in user_input.lower() or "man" in user_input.lower():
            client_info["gender"] = "Male"
        elif "female" in user_input.lower() or "woman" in user_input.lower():
            client_info["gender"] = "Female"
        else:
            return "I didn't quite catch that. Could you please specify if the customer is Male or Female?"

    elif current_question == questions[1]:  # Senior Citizen
        if user_input.strip() == "1" or user_input.strip().lower() == "yes":
            client_info["Senior_Citizen "] = 1
        elif user_input.strip() == "0" or user_input.strip().lower() == "no":
            client_info["Senior_Citizen "] = 0
        else:
            return "Please answer with '1' for Yes or '0' for No."

    elif current_question == questions[2]:  # Married
        if user_input.strip().lower() == "yes":
            client_info["Is_Married"] = "Yes"
        elif user_input.strip().lower() == "no":
            client_info["Is_Married"] = "No"
        else:
            return "Please answer with 'Yes' or 'No'."

    elif current_question == questions[3]:  # Dependents
        if user_input.strip().lower() == "yes":
            client_info["Dependents"] = "Yes"
        elif user_input.strip().lower() == "no":
            client_info["Dependents"] = "No"
        else:
            return "Please answer with 'Yes' or 'No'."

    elif current_question == questions[4]:  # Months with company
        if user_input.isdigit():
            client_info["tenure"] = int(user_input)
        else:
            return "Please provide a valid number of months."

    elif current_question == questions[5]:  # Phone service
        if user_input.strip().lower() == "yes":
            client_info["Phone_Service"] = "Yes"
        elif user_input.strip().lower() == "no":
            client_info["Phone_Service"] = "No"
        else:
            return "Please answer with 'Yes' or 'No'."

    elif current_question == questions[6]:  # Dual customer
        if user_input.strip().lower() == "yes":
            client_info["Dual"] = "Yes"
        elif user_input.strip().lower() == "no":
            client_info["Dual"] = "No"
        elif "no phone service" in user_input.lower():
            client_info["Dual"] = "No phone service"
        else:
            return "Please answer with 'Yes', 'No', or 'No phone service'."

    elif current_question == questions[7]:  # Internet service
    # Mapping user input to the correct format
        internet_service_map = {
            "dsl": "DSL",
            "fiber optic": "Fiber optic",
            "no": "No"
        }
        # Normalize the input and store the formatted result
        user_input_normalized = user_input.strip().lower()
        if user_input_normalized in internet_service_map:
            client_info["Internet_Service"] = internet_service_map[user_input_normalized]
        else:
            return "Please provide 'DSL', 'Fiber optic', or 'No' for internet service."


    elif current_question == questions[8]:  # Online security
        if user_input.strip().lower() in ["yes", "no", "no internet service"]:
            client_info["Online_Security"] = user_input.strip().capitalize()
        else:
            return "Please answer with 'Yes', 'No', or 'No internet service'."

    elif current_question == questions[9]:  # Online backup
        if user_input.strip().lower() in ["yes", "no", "no internet service"]:
            client_info["Online_Backup"] = user_input.strip().capitalize()
        else:
            return "Please answer with 'Yes', 'No', or 'No internet service'."

    elif current_question == questions[10]:  # Device protection
        if user_input.strip().lower() in ["yes", "no", "no internet service"]:
            client_info["Device_Protection"] = user_input.strip().capitalize()
        else:
            return "Please answer with 'Yes', 'No', or 'No internet service'."

    elif current_question == questions[11]:  # Technical support
        if user_input.strip().lower() in ["yes", "no", "no internet service"]:
            client_info["Tech_Support"] = user_input.strip().capitalize()
        else:
            return "Please answer with 'Yes', 'No', or 'No internet service'."

    elif current_question == questions[12]:  # Streaming TV
        if user_input.strip().lower() in ["yes", "no", "no internet service"]:
            client_info["Streaming_TV"] = user_input.strip().capitalize()
        else:
            return "Please answer with 'Yes', 'No', or 'No internet service'."

    elif current_question == questions[13]:  # Streaming movies
        if user_input.strip().lower() in ["yes", "no", "no internet service"]:
            client_info["Streaming_Movies"] = user_input.strip().capitalize()
        else:
            return "Please answer with 'Yes', 'No', or 'No internet service'."

    elif current_question == questions[14]:  # Contract type
        if user_input.strip().lower() in ["month-to-month", "one year", "two year"]:
            client_info["Contract"] = user_input.strip().capitalize()
        else:
            return "Please provide 'Month-to-month', 'One year', or 'Two year'."

    elif current_question == questions[15]:  # Paperless billing
        if user_input.strip().lower() == "yes":
            client_info["Paperless_Billing"] = "Yes"
        elif user_input.strip().lower() == "no":
            client_info["Paperless_Billing"] = "No"
        else:
            return "Please answer with 'Yes' or 'No'."

    elif current_question == questions[16]:  # Payment method
        if user_input.strip().lower() in ["electronic check", "mailed check", "bank transfer (automatic)", "credit card (automatic)"]:
            client_info["Payment_Method"] = user_input.strip().capitalize()
        else:
            return "Please provide a valid payment method."

    elif current_question == questions[17]:  # Monthly charge
        if user_input.replace('.', '', 1).isdigit():
            client_info["Monthly_Charges"] = float(user_input)
        else:
            return "Please provide a valid amount."

    elif current_question == questions[18]:  # Total charges
        if user_input.replace('.', '', 1).isdigit():
            client_info["Total_Charges"] = float(user_input)
        else:
            return "Please provide a valid amount."
        
    # Proceed to the next question after handling the current answer
    if st.session_state.question_index < len(questions) - 1:
        st.session_state.question_index += 1
        next_question = questions[st.session_state.question_index]
        return next_question
    else:
        df = export_client_info_to_dataframe()
        df.to_csv('client_info.csv', index=False)
        client_data_dict = df.iloc[0].to_dict()

        # Call your prepare_data_for_model function
        processed_data = prepare_data_for_model(client_data_dict)
        
        # Display the processed data
        st.write("Thank you! We have collected all the necessary information.")
        st.write("Here is the information you provided:")
        st.dataframe(df)  # Display the original DataFrame in a table format
        
        # st.write("Here is the processed data ready for the model:")
        # st.dataframe(processed_data)  # Display the processed data
        # Step 2: Make the prediction
        prediction = model.predict(processed_data)

        # Step 3: Display the prediction
        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
        # step 4: LLM Generate Explanation
        explanation = get_response(df, prediction, column_info)
        st.write("Explanation of the prediction:")
        st.write(explanation)

        # # Ask the user if they want to start the process again
        # user_restart_input = st.text_input("Do you want to start again? (Yes or No)")
        
        # if user_restart_input.strip().lower() == "yes":
        #     # Reset the session state to start over
        #     st.session_state.question_index = 0
        #     st.session_state.client_info = {}

        #     # Explicitly set the chatbot's state variables for a fresh start
        #     st.session_state.chat_history = []

        #     # Refresh the page to start over
        #     st.experimental_rerun()

        #     return "Let's start from the beginning. Please answer the first question: What is the customer's gender?"

        # else:
        #     return "Thank you for using the chatbot. Have a great day!"
        st.write("If You want a help again please refresh the page.")
        return "Thank you for using the chatbot. Have a great day!"

# Automatically prompt the first question after intro
if len(st.session_state.chat_history) == 0:
    first_question = questions[st.session_state.question_index]
    st.session_state.chat_history.append({"role": "assistant", "content": first_question})

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response
user_input = st.chat_input("Please provide the information to start.")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            # Process user input and determine the next step
            response = handle_user_input(user_input)

            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {e}")
