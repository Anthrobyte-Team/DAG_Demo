import os, datetime, getpass
from typing import Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import logging
import certifi

logger = logging.getLogger(__name__)

def _get_db_url():
    try:
        import streamlit as st
        return st.secrets["mysql"]["uri"]
    except (ImportError, AttributeError, KeyError):
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            return secrets["mysql"]["uri"]
        except Exception:
            logger.exception("Could not load DB URL from secrets.toml")
            return None

DB_URL = _get_db_url()
_engine = create_engine(DB_URL, pool_pre_ping=True, future=True, connect_args={"ssl": {
        "ca": certifi.where()
    }})

def upsert_asset(row: Dict[str, Any]) -> None:
    """Insert or update row in rag_entries."""
    logger.info(f"Upserting asset into rag_entries (file_id={row.get('file_id')})")
    try:
        with _engine.begin() as conn:
            try:
                conn.execute(text("""
                    INSERT INTO rag_entries
                      (file_id, doc_hash, file_name, file_type, blob_url, size_bytes,
                       display_name, pages_total, image_count,
                       ingested_by, ingested_at, milvus_collection, agent_type)
                    VALUES
                      (:file_id, :doc_hash, :file_name, :file_type, :blob_url, :size_bytes,
                       :display_name, :pages_total, :image_count,
                       :ingested_by, :ingested_at, :milvus_collection, :agent_type)
                """), row)
                logger.debug("Insert succeeded for row=%s", row)
            except IntegrityError:
                logger.warning(f"Duplicate file_id={row.get('file_id')}, updating instead.")
                conn.execute(text("""
                    UPDATE rag_entries
                       SET doc_hash=:doc_hash, file_type=:file_type, blob_url=:blob_url,
                           size_bytes=:size_bytes, display_name=:display_name,
                           pages_total=:pages_total, image_count=:image_count,
                           ingested_by=:ingested_by, ingested_at=:ingested_at,
                           milvus_collection=:milvus_collection,
                           agent_type=:agent_type
                     WHERE file_id=:file_id
                """), row)
                logger.debug("Update succeeded for row=%s", row)
    except SQLAlchemyError:
        logger.exception("Database error during upsert_asset.")


def exists_by_doc_hash(doc_hash: str) -> bool:
    logger.info(f"Checking existence of doc_hash={doc_hash}")
    try:
        with _engine.begin() as conn:
            r = conn.execute(
                text("SELECT 1 FROM rag_entries WHERE doc_hash=:h LIMIT 1"), {"h": doc_hash}
            ).first()
            exists = bool(r)
            logger.debug(f"Exists={exists} for doc_hash={doc_hash}")
            return exists
    except SQLAlchemyError:
        logger.exception("Error checking existence by doc_hash.")
        return False


def get_pdf_url_by_doc_hash(doc_hash: str) -> str:
    logger.info(f"Fetching PDF URL for doc_hash={doc_hash}")
    try:
        with _engine.begin() as conn:
            r = conn.execute(
                text("SELECT blob_url FROM rag_entries WHERE doc_hash=:h LIMIT 1"), {"h": doc_hash}
            ).first()
            if r:
                logger.debug(f"PDF URL for doc_hash={doc_hash}: {r[0]}")
            else:
                logger.warning(f"No PDF URL found for doc_hash={doc_hash}")
            return r[0] if r else None
    except SQLAlchemyError:
        logger.exception("Error fetching PDF URL by doc_hash.")
        return None


def get_pdf_url_by_folder_name(folder_name: str) -> str:
    logger.info(f"Fetching PDF URL for folder_name={folder_name}")
    try:
        with _engine.begin() as conn:
            r = conn.execute(
                text("SELECT blob_url FROM rag_entries WHERE file_name=:n LIMIT 1"), {"n": folder_name}
            ).first()
            if r:
                logger.debug(f"PDF URL for folder={folder_name}: {r[0]}")
            else:
                logger.warning(f"No PDF URL found for folder={folder_name}")
            return r[0] if r else None
    except SQLAlchemyError:
        logger.exception("Error fetching PDF URL by folder name.")
        return None


def get_all_folder_names() -> list:
    logger.info("Fetching all folder names from rag_entries")
    try:
        with _engine.begin() as conn:
            r = conn.execute(text("SELECT file_name FROM rag_entries")).fetchall()
            names = [row[0] for row in r]
            logger.debug(f"Found {len(names)} folder names.")
            return names
    except SQLAlchemyError:
        logger.exception("Error fetching all folder names.")
        return []


def get_imageagent_sources() -> list[dict]:
    logger.info("Fetching all imageagent sources from rag_entries")
    try:
        with _engine.begin() as conn:
            result = conn.execute(
                text("SELECT * FROM rag_entries WHERE agent_type = 'imageagent'")
            )
            rows = [dict(row._mapping) for row in result]  # ✅ safe conversion
            logger.debug(f"Found {len(rows)} imageagent sources.")
            return rows
    except SQLAlchemyError:
        logger.exception("Error fetching imageagent sources.")
        return []

