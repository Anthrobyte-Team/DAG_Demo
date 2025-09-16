import streamlit as st
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
import logging
import certifi

logger = logging.getLogger(__name__)

# Load DB URL from Streamlit secrets
DB_URL = st.secrets["mysql"]["uri"]
_engine = create_engine(DB_URL, pool_pre_ping=True, future=True, connect_args={"ssl": {"ca": certifi.where()}})

# Upsert asset metadata
def upsert_asset(row: Dict[str, Any]) -> None:
    """
    Expects keys:
        file_id, doc_hash, file_name, file_type, blob_url, size_bytes,
        display_name, pages_total, image_count, ingested_by, ingested_at, milvus_collection, agent_type
    """
    with _engine.begin() as conn:
        try:
            conn.execute(text("""
                INSERT INTO rag_entries
                  (file_id, doc_hash, file_name, file_type, blob_url, size_bytes,
                   display_name, pages_total, image_count, ingested_by, ingested_at, milvus_collection, agent_type)
                VALUES
                  (:file_id, :doc_hash, :file_name, :file_type, :blob_url, :size_bytes,
                   :display_name, :pages_total, :image_count, :ingested_by, :ingested_at, :milvus_collection, :agent_type)
            """), row)
        except IntegrityError:
            conn.execute(text("""
                UPDATE rag_entries
                   SET doc_hash=:doc_hash, file_type=:file_type, blob_url=:blob_url,
                       size_bytes=:size_bytes, display_name=:display_name,
                       pages_total=:pages_total, image_count=:image_count,
                       ingested_by=:ingested_by, ingested_at=:ingested_at,
                       milvus_collection=:milvus_collection, agent_type=:agent_type
                 WHERE file_id=:file_id
            """), row)

# Check if a document exists by its hash
def exists_by_doc_hash(doc_hash: str) -> bool:
    with _engine.begin() as conn:
        r = conn.execute(text("SELECT 1 FROM rag_entries WHERE doc_hash=:h LIMIT 1"), {"h": doc_hash}).first()
        return bool(r)

# Get blob URL by file_id or file_name
def url_by_file(file_id: Optional[str], file_name: Optional[str]) -> Optional[str]:
    with _engine.begin() as conn:
        if file_id:
            r = conn.execute(text("SELECT blob_url FROM rag_entries WHERE file_id=:fid"), {"fid": file_id}).first()
            return r[0] if r else None
        if file_name:
            r = conn.execute(text("""
                SELECT blob_url FROM rag_entries
                 WHERE file_name=:fn
                 ORDER BY ingested_at DESC
                 LIMIT 1
            """), {"fn": file_name}).first()
            return r[0] if r else None
        return None

# Get all RAG agent sources
def get_ragagent_sources() -> list[dict]:
    with _engine.begin() as conn:
        result = conn.execute(
            text("SELECT * FROM rag_entries WHERE agent_type = 'rag agent'")
        )
        return [dict(row) for row in result]