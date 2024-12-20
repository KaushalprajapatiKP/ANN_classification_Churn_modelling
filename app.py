import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title and description
st.title('Customer Churn Prediction')
st.markdown("""
    Predict the likelihood of a customer churning based on various attributes.
    Fill in the details below and get the churn probability.
""")

# Sidebar for input
st.sidebar.header('Customer Information')

# User input
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92)
balance = st.sidebar.number_input('Balance', min_value=0, value=1000)
credit_score = st.sidebar.number_input('Credit Score', min_value=0, value=650)
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0, value=50000)
tenure = st.sidebar.slider('Tenure', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display results
st.subheader('Prediction Results')
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.markdown('<h3 style="color: red;">The customer is likely to churn.</h3>', unsafe_allow_html=True)
else:
    st.markdown('<h3 style="color: green;">The customer is not likely to churn.</h3>', unsafe_allow_html=True)

# Additional styling for better UX
st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 18px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)
