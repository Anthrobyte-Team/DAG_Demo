import re
import logging
from datetime import datetime
from .utils import images_to_pdf
from .sql_store import get_pdf_url_by_folder_name, get_imageagent_sources

logger = logging.getLogger(__name__)

MAX_RESULTS = 1000


def _tokens(s: str):
    toks = re.findall(r"[a-z0-9]+", s.lower())
    logger.debug(f"Tokenized string='{s}' → tokens={toks}")
    return toks


def fetch_by_folder_exact(collection, folder_name, max_results=MAX_RESULTS):
    logger.info(f"Fetching exact matches for folder_name='{folder_name}' (collection={collection})")

    blob_urls = []
    try:
        url_str = get_pdf_url_by_folder_name(folder_name)
        if url_str:
            # If multiple URLs are stored, split by comma
            blob_urls = [u for u in url_str.split(",") if u.strip()]
        logger.info(f"Found {len(blob_urls)} PDF blob URLs for folder='{folder_name}'")
        logger.debug(f"Blob URLs: {blob_urls}")
    except Exception:
        logger.exception(f"Error while fetching blob URLs for folder='{folder_name}'")

    return [{"file_path": u} for u in blob_urls]


def fetch_by_folder_flexible(client, collection, phrase, max_results=MAX_RESULTS):
    logger.info(f"Flexible search in collection={collection} with phrase='{phrase}' (limit={max_results})")

    toks = _tokens(phrase)
    if not toks:
        logger.warning(f"No valid tokens parsed from phrase='{phrase}' → returning []")
        return []

    like_expr = " && ".join([f'(folder_name like \"%{t}%\")' for t in toks])
    logger.debug(f"Generated filter expression: {like_expr}")

    try:
        results = client.query(
            collection_name=collection,
            filter=like_expr,
            output_fields=["image_name", "file_path", "folder_name"],
            limit=max_results,
        )
        logger.info(f"Flexible search returned {len(results)} results for phrase='{phrase}'")
        logger.debug(f"Results: {results}")
        return results
    except Exception:
        logger.exception(f"Error while running flexible search for phrase='{phrase}'")
        return []


def fetch_pdf_url_by_folder_name(folder_name):
    logger.info(f"Fetching PDF URL for folder_name='{folder_name}'")
    try:
        url = get_pdf_url_by_folder_name(folder_name)
        if url:
            logger.debug(f"PDF URL for folder='{folder_name}': {url}")
        else:
            logger.warning(f"No PDF URL found for folder='{folder_name}'")
        return url
    except Exception:
        logger.exception(f"Error fetching PDF URL for folder='{folder_name}'")
        return None


def fetch_imageagent_sources():
    logger.info("Fetching all image agent sources from SQL.")
    try:
        sources = get_imageagent_sources()
        logger.info(f"Retrieved {len(sources)} sources.")
        logger.debug(f"Sources: {sources}")
        return sources
    except Exception:
        logger.exception("Error fetching image agent sources.")
        return []
