from pymilvus import MilvusClient
import streamlit as st

try:
    ZILLIZ_URI = st.secrets["milvus-image"]["ZILLIZ_URI"]
    ZILLIZ_TOKEN = st.secrets["milvus-image"]["ZILLIZ_TOKEN"]
    COLLECTION_NAME = st.secrets["milvus-image"]["COLLECTION_NAME"]
except Exception:
    # Fallback to toml if not running in Streamlit or secrets missing
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    ZILLIZ_URI = secrets["milvus-image"]["ZILLIZ_URI"]
    ZILLIZ_TOKEN = secrets["milvus-image"]["ZILLIZ_TOKEN"]
    COLLECTION_NAME = secrets["milvus-image"]["COLLECTION_NAME"]

def get_milvus_client():
    return MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

def get_collection_name():
    return COLLECTION_NAME
