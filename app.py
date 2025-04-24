import streamlit as st

#from main import FUNCTION

# Title
st.title('My Projekt')

# Textfield for the User
user_input = st.text_input('GitHub-Link:')

# Button and give answer
if st.button('Send'):
    st.write('Answer: {user_input}')