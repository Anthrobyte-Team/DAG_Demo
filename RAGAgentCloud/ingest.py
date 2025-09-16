import os, re, hashlib, datetime, getpass
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import streamlit as st

logger = logging.getLogger(__name__)

META_TEMPLATE = {
    "source_type": "",    
    "source": "",          
    "source_url": "",
    "canonical_url": "",
    "title": "",
    "file_id": "",
    "file_name": "",
    "display_name": "",
    "pdf_title": "",
    "source_path": "",      # no local uploads now
    "pages_total": -1,
    "page": -1,
    "start_index": -1,
    "file_url": "",         # <-- Blob URL used by Sources UI
}

# --- Azure config from secrets ---
try:
    AZURE_CONN_STR  = st.secrets["AZURE-RAG"]["AZURE_STORAGE_CONNECTION_STRING"]
    AZURE_CONTAINER = st.secrets["AZURE-RAG"]["AZURE_BLOB_CONTAINER"]
    AZURE_BASE      = (st.secrets["AZURE-RAG"]["AZURE_BLOB_BASE"] or "").rstrip("/")
    AZURE_SAS       = st.secrets["AZURE-RAG"]["AZURE_BLOB_SAS"]
except Exception:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    AZURE_CONN_STR  = secrets["AZURE-RAG"]["AZURE_STORAGE_CONNECTION_STRING"]
    AZURE_CONTAINER = secrets["AZURE-RAG"]["AZURE_BLOB_CONTAINER"]
    AZURE_BASE      = (secrets["AZURE-RAG"]["AZURE_BLOB_BASE"] or "").rstrip("/")
    AZURE_SAS       = secrets["AZURE-RAG"]["AZURE_BLOB_SAS"]

if AZURE_SAS and not AZURE_SAS.startswith("?"):
    AZURE_SAS = "?" + AZURE_SAS

def _apply_meta(doc: Document, overrides: dict) -> Document:
    meta = dict(META_TEMPLATE)
    if isinstance(doc.metadata, dict):
        for k in ("page", "start_index"):
            if k in doc.metadata:
                meta[k] = doc.metadata[k]
    meta.update(overrides or {})
    doc.metadata = meta
    return doc

def _safe_filename(name: str) -> str:
    nm = (name or "uploaded.pdf").strip()
    nm = re.sub(r"[^A-Za-z0-9._-]+", "_", nm)
    return nm[:120] or "uploaded.pdf"

# ---- Azure upload helper
def _upload_to_azure(file_id: str, safe_name: str, data: bytes) -> str:
    """
    Upload PDF bytes to Azure Blob as pdfs/<file_id><safe_name> and
    return a URL usable by the app.
    """
    try:
        from azure.storage.blob import BlobServiceClient, ContentSettings

        svc = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob_name = f"rag_uploads/{file_id}{safe_name}"
        bc = svc.get_blob_client(container=AZURE_CONTAINER, blob=blob_name)
        bc.upload_blob(data, overwrite=True,
                       content_settings=ContentSettings(content_type="application/pdf"))

        if AZURE_BASE:
            # Correct URL: <base>/<blob_name>?<sas>
            return f"{AZURE_BASE}/{blob_name}{AZURE_SAS}"
        else:
            # Public container fallback (no SAS): requires public read
            # Try to extract account from connection string
            account = ""
            for part in AZURE_CONN_STR.split(";"):
                if part.startswith("AccountName="):
                    account = part.split("=", 1)[1]
            return f"https://{account}.blob.core.windows.net/{AZURE_CONTAINER}/{blob_name}{AZURE_SAS}"
    except Exception:
        return ""

# Web ingest
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl
from langchain_community.document_loaders import WebBaseLoader

def _canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                  if not k.lower().startswith("utm_") and k.lower() not in {"ref","ref_src"}]
        params.sort()
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ""))
    except Exception:
        return url

def load_and_split_docs(url: str) -> List[Document]:
    """Loads a web page, splits it, and attaches metadata for deduplication and tracking."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    canonical = _canonical_url(url)
    title = docs[0].metadata.get("title", "") if docs else ""
    for i, doc in enumerate(chunks):
        doc.metadata.update({
            "source_type": "url",
            "source": url,
            "source_url": url,
            "canonical_url": canonical,
            "title": title,
            "file_id": "",           # Not needed for URLs
            "file_name": url,        # Use URL as file_name
            "display_name": title or url,
            "pdf_title": "",
            "source_path": "",
            "pages_total": -1,
            "page": -1,
            "start_index": -1,
            "file_url": url,         # For URLs, just use the URL
        })
    return chunks

# ---- PDF ingest: bytes -> Azure -> split in memory -> stamp file_url ----
from .sql_store import upsert_asset

def load_and_split_pdf(uploaded_file) -> List[Document]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF ingestion")

    data = uploaded_file.read()
    file_id   = hashlib.md5(data).hexdigest()                  # binary identity
    file_name = getattr(uploaded_file, "name", "uploaded.pdf") or "uploaded.pdf"
    safe_name = _safe_filename(file_name)

    # Upload to Azure (no local uploads/)
    file_url = _upload_to_azure(file_id, safe_name, data) or ""

    # Open from memory to read metadata/pages
    pdf_title, pages_total = "", -1
    page_docs: List[Document] = []
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        meta = doc.metadata or {}
        pdf_title = (meta.get("title") or "").strip()
        pages_total = len(doc)
        for p in range(pages_total):
            text = doc.load_page(p).get_text("text")
            page_docs.append(Document(page_content=text, metadata={"page": p}))
    finally:
        try: doc.close()
        except Exception: pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)
    chunks = splitter.split_documents(page_docs)

    display_name = pdf_title or file_name
    enriched: List[Document] = []
    for c in chunks:
        enriched.append(_apply_meta(c, {
            "source_type": "pdf",
            "source": "",
            "file_id": file_id,
            "file_name": file_name,
            "display_name": display_name,
            "pdf_title": pdf_title,
            "source_path": "",
            "pages_total": pages_total,
            "file_url": file_url,
        }))

    # Content hash (use enriched text so it matches retriever JSON hash)
    doc_hash = hashlib.md5("".join(d.page_content for d in enriched).encode("utf-8")).hexdigest()

    logger.info("ingest_split", extra={"source_type": "pdf", "source": file_name, "chunks": len(enriched)})
    return enriched
