import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = os.getenv('API_URL')

@st.cache_data()
def classify_intent(user_text):
    try:
        data = {'text': user_text}
        
        response = requests.post(API_URL, json=data)
        
        if response.status_code == 200:
            classification = response.text
            return classification
        else:
            return f'API error: {response.status_code}'
    
    except Exception as e:
        return f'Error: {e}'
      
if __name__ == '__main__':
    st.header('Banking Intent Classifier', divider='rainbow')
    text_input = st.text_input("Enter some text 👇",)
    submit_btn = st.button('Classify Intent', type='secondary')
    
    if submit_btn:
        classification = classify_intent(text_input)
        st.success(classification)



