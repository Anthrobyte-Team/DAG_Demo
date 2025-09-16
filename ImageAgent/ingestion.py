import os, datetime, getpass
import logging
from .utils import compute_folder_hash, images_to_pdf, upload_images_to_azure
from .sql_store import exists_by_doc_hash, upsert_asset, get_pdf_url_by_doc_hash
from .feature_extractor import get_extractor

logger = logging.getLogger(__name__)


def process_and_insert(client, folder_path, collection_name):
    logger.info(f"Starting ingestion for folder: {folder_path}, collection: {collection_name}")
    extractor = get_extractor()

    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        logger.warning("No images found for ingestion.")
        return False

    logger.info(f"Found {len(image_paths)} images for ingestion: {image_paths}")

    for img_path in image_paths:
        try:
            logger.info(f"Extracting features for {img_path}")
            vec = extractor(img_path)

            logger.info(f"Inserting {img_path} into Milvus collection={collection_name}")
            client.insert(
                collection_name=collection_name,
                data=[{
                    "image_name": os.path.basename(img_path),
                    "file_path": img_path,
                    "folder_name": os.path.basename(folder_path),
                    "embedding": vec.tolist()
                }]
            )
            logger.debug(f"Successfully inserted {img_path} into Milvus.")
        except Exception as e:
            logger.exception(f"Failed to insert {img_path} into Milvus.")
    return True


def handle_folder_upload(client, folder_path, collection_name):
    logger.info(f"Handling folder upload: {folder_path}")

    folder_hash, image_count, total_size = compute_folder_hash(folder_path)
    logger.debug(f"Folder hash={folder_hash}, image_count={image_count}, total_size={total_size}")

    if exists_by_doc_hash(folder_hash):
        logger.warning(f"Folder {folder_path} already ingested (hash={folder_hash}). Skipping Milvus + blob upload.")
        return

    # Ingest images to Milvus
    success = process_and_insert(client, folder_path, collection_name)
    if not success:
        logger.error("Ingestion failed or no images found. Aborting upload.")
        return

    # Gather image paths again for Azure upload
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        logger.error("No images found for Azure upload. Aborting.")
        return

    # --- Generate PDF from images and upload to Azure ---
    logger.info("Generating PDF from images for Azure upload.")
    pdf_blob_url = images_to_pdf(image_paths, output_name=f"{os.path.basename(folder_path)}.pdf")
    logger.info(f"PDF successfully generated and uploaded to Azure: {pdf_blob_url}")
    # ---------------------------------------------------

    # Store metadata in SQL (store only PDF blob URL)
    row = {
        "file_id": folder_hash,
        "doc_hash": folder_hash,
        "file_name": os.path.basename(folder_path),
        "file_type": "images",
        "blob_url": pdf_blob_url,
        "size_bytes": total_size,
        "display_name": os.path.basename(folder_path),
        "pages_total": 1,
        "image_count": image_count,
        "ingested_by": getpass.getuser(),
        "ingested_at": datetime.datetime.now(),
        "milvus_collection": collection_name,
        "agent_type": "imageagent",  # <-- NEW FIELD
    }
    try:
        upsert_asset(row)
        logger.info("SQL entry created successfully for uploaded PDF.")
    except Exception as e:
        logger.exception("Failed to create SQL entry for uploaded PDF.")
