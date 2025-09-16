import streamlit as st
from langchain_openai import ChatOpenAI

# Get the API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=api_key,
)