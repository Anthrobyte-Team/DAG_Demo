from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
from .web_prompt import SUMMARY_PROMPT

# Load OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

def summarize_news(news_results: list) -> str:
    """
    Send news results to LLM and get a custom summary.
    """
    # Combine news results into a single string
    news_text = "\n\n".join(news_results)
    prompt = SUMMARY_PROMPT.format(news=news_text)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=500,
        openai_api_key=OPENAI_API_KEY
    )

    response = llm.invoke([
        SystemMessage(content="You summarize news."),
        HumanMessage(content=prompt)
    ])

    return response.content.strip()
