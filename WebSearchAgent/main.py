import requests
import os
import streamlit as st

SERPER_API_KEY = st.secrets["websearch"]["SERPER_API_KEY"]

def web_search(query: str, num_results: int = 1):
    """
    Search the web using Serper API and return top results.
    """
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("organic", [])[:num_results]:
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })
    return results