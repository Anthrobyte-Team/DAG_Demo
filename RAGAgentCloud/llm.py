from langchain_openai import ChatOpenAI
import streamlit as st

OPENAI_API_KEY = st.secrets["openai"]["api_key"]
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.2

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_MODEL,
    temperature=OPENAI_TEMPERATURE
)
