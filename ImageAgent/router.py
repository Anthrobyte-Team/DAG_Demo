import json
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

def _get_openai_api_key():
    try:
        import streamlit as st
        return st.secrets["openai"]["api_key"]
    except (ImportError, AttributeError, KeyError):
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            return secrets["openai"]["api_key"]
        except Exception:
            logger.exception("Could not load OpenAI API key from secrets.toml")
            return None

OPENAI_API_KEY = _get_openai_api_key()
client_llm = OpenAI(api_key=OPENAI_API_KEY)

def pick_best_folder_name(user_query: str, folder_names: list) -> str:
    logger.info("Picking best folder name for query='%s'", user_query)

    if not folder_names:
        logger.warning("No folder names provided for matching.")
        return None

    system_prompt = (
        "You are an expert at matching user queries to folder names. "
        "Given a user query and a list of folder names, return ONLY the best matching folder name as a JSON string: {\"best_folder\": \"folder_name\"}."
    )
    folder_list = "\n".join(folder_names)
    user_prompt = f"User query: {user_query}\nFolder names:\n{folder_list}"

    logger.debug("Sending request to LLM with %d folder names.", len(folder_names))

    try:
        resp = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        logger.debug("LLM raw response: %s", resp)

        best_folder = json.loads(resp.choices[0].message.content)["best_folder"]
        logger.info("Best folder chosen: %s", best_folder)
        return best_folder
    except Exception:
        logger.exception("Failed to parse best folder name from LLM response.")
        return None
