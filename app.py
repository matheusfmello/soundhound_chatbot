import streamlit as st
import pandas as pd
from chains import call_chain
from uuid import uuid4

st.title("SoundHound Chatbot 🎧")

user_id = uuid4()
user_id = 'def'

history_db = pd.read_csv('chat_history.csv')
user_history = history_db[history_db['user_id']==user_id]    

for index, row in user_history.iterrows():
    with st.chat_message(row['author']):
        st.markdown(row['content'])
        
if user_history.empty:
    with st.chat_message("ai"):
    
        st.write("Hello, I'm a SoundHound 🎧 AI assistant. How can I help you?")

# React to user input
if prompt := st.chat_input("Type anything"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history

    response = call_chain(user_input=prompt, user_id=user_id)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    

