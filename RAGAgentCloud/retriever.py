import hashlib, json, os
from typing import List
from langchain_core.documents import Document
from pymilvus import connections
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
import logging
import streamlit as st
logger = logging.getLogger(__name__)
from .sql_store import exists_by_doc_hash

# --- Load secrets from Streamlit ---
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
MILVUS_URI   = st.secrets["milvus-rag"]["ZILLIZ_URI"]
MILVUS_TOKEN = st.secrets["milvus-rag"]["ZILLIZ_TOKEN"]
MILVUS_DB    = st.secrets["milvus-rag"].get("db", "default")
COLLECTION_NAME = st.secrets["milvus-rag"]["COLLECTION_NAME"]

USE_SQL_DEDUP  = st.secrets["general"].get("USE_SQL_DEDUP", 1) == 1
USE_JSON_DEDUP = st.secrets["general"].get("USE_JSON_DEDUP", 1) == 1

connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=MILVUS_DB)
_CONNECTION_ARGS = {"uri": MILVUS_URI, "token": MILVUS_TOKEN, "db_name": MILVUS_DB}

_CLUSTER_TAG  = hashlib.md5((MILVUS_URI + "|" + MILVUS_DB).encode()).hexdigest()[:8]
REGISTRY_FILE = f".milvus_hash_registry.{_CLUSTER_TAG}.json"

_INDEX_PARAMS  = {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
_SEARCH_PARAMS = {"metric_type": "IP", "params": {"ef": 64}}

def _load_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_registry(reg):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)

def get_hash_for_docs(docs: List[Document]) -> str:
    combined = "".join(d.page_content for d in docs)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()

def _vectorstore():
    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    return Milvus(
        embedding_function=emb,
        collection_name=COLLECTION_NAME,
        connection_args=_CONNECTION_ARGS,
        index_params=_INDEX_PARAMS,
        search_params=_SEARCH_PARAMS,
    )

def get_collection_retriever():
    logger.info("retriever_ready", extra={"search_type": "mmr", "k": 6, "fetch_k": 40, "ef": 64, "metric": _SEARCH_PARAMS["metric_type"]})
    return _vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 40, "lambda_mult": 0.7}  # ↑ relevance, ↓ noise
    )

def _attach_hash(d: Document, doc_hash: str) -> Document:
    meta = dict(d.metadata or {})
    meta["doc_hash"] = doc_hash
    d.metadata = meta
    return d

def build_retriever_from_docs(docs: List[Document]):
    if not docs:
        return get_collection_retriever()

    reg = _load_registry()
    doc_hash = get_hash_for_docs(docs)

    # Deduplication for both PDFs and URLs
    try:
        if USE_SQL_DEDUP and exists_by_doc_hash(doc_hash):
            logger.info("index_result", extra={"doc_hash": doc_hash, "chunks": len(docs), "dedup": "sql"})
            print("✅ Document already exists (SQL). Skipping upload.")
            return get_collection_retriever()
    except Exception as e:
        logger.warning("sql_dedup_error", extra={"err": str(e)})

    # Upload to Milvus
    docs = [_attach_hash(d, doc_hash) for d in docs]
    vector_store = Milvus.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small"),
        collection_name=COLLECTION_NAME,
        connection_args=_CONNECTION_ARGS,
        index_params=_INDEX_PARAMS,
    )

    # Insert row into SQL for both PDFs and URLs
    from sql_store import upsert_asset
    import datetime, getpass

    m = dict(docs[0].metadata or {})
    file_type = "pdf" if m.get("source_type") == "pdf" else "url"
    upsert_asset({
        "file_id": m.get("file_id", ""),
        "doc_hash": doc_hash,
        "file_name": m.get("file_name", m.get("source_url", "")),  # For URLs, use source_url
        "file_type": file_type,
        "blob_url": m.get("file_url", m.get("source_url", "")),    # For URLs, use source_url
        "size_bytes": None,
        "display_name": m.get("display_name", m.get("title","")),
        "pages_total": m.get("pages_total", -1),
        "image_count": 0 if file_type == "pdf" else None,
        "ingested_by": os.getenv("RAG_INGESTED_BY") or getpass.getuser(),
        "ingested_at": datetime.datetime.utcnow(),
        "milvus_collection": COLLECTION_NAME,
        "agent_type": "rag agent",
    })

    # Update local JSON registry if enabled
    if USE_JSON_DEDUP:
        reg[doc_hash] = True
        _save_registry(reg)

    logger.info("index_result", extra={"doc_hash": doc_hash, "chunks": len(docs), "dedup": "new"})
    print("✅ New document indexed in Milvus Cloud and registered in SQL.")
    return get_collection_retriever()