import os, re, hashlib, tempfile, logging
from fpdf import FPDF
from PIL import Image
import streamlit as st
from azure.storage.blob import BlobServiceClient, ContentSettings

# --- Setup logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ALLOWED_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# --- Azure config ---
try:
    AZURE_ACCOUNT   = st.secrets["AZURE-Image"]["AZURE_STORAGE_ACCOUNT"]
    AZURE_CONTAINER = st.secrets["AZURE-Image"]["AZURE_BLOB_CONTAINER"]
    AZURE_BASE      = (st.secrets["AZURE-Image"]["AZURE_BLOB_BASE"] or "").rstrip("/")
    AZURE_SAS       = st.secrets["AZURE-Image"]["AZURE_BLOB_SAS"]
except Exception:
    # Fallback to toml if not running in Streamlit or secrets missing
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    AZURE_ACCOUNT   = secrets["AZURE-Image"]["AZURE_STORAGE_ACCOUNT"]
    AZURE_CONTAINER = secrets["AZURE-Image"]["AZURE_BLOB_CONTAINER"]
    AZURE_BASE      = (secrets["AZURE-Image"]["AZURE_BLOB_BASE"] or "").rstrip("/")
    AZURE_SAS       = secrets["AZURE-Image"]["AZURE_BLOB_SAS"]

if AZURE_SAS and not AZURE_SAS.startswith("?"):
    AZURE_SAS = "?" + AZURE_SAS


def upload_images_to_azure(image_paths, folder_name):
    """Upload multiple images to Azure Blob Storage and return their URLs."""
    logger.info("Starting image upload to Azure | folder=%s | total_images=%d", folder_name, len(image_paths))

    # Use credentials loaded at the top of the file
    account   = AZURE_ACCOUNT
    container = AZURE_CONTAINER
    base_no_q = AZURE_BASE
    sas_q     = AZURE_SAS

    svc = BlobServiceClient(account_url=f"https://{account}.blob.core.windows.net", credential=sas_q or None)

    urls = []
    for img_path in image_paths:
        safe_name = _sanitize_name(os.path.basename(img_path))
        blob_name = f"imageagent_uploads/images/{folder_name}/{safe_name}"
        logger.info("Uploading image: %s -> %s", img_path, blob_name)

        with open(img_path, "rb") as f:
            data = f.read()
        bc = svc.get_blob_client(container=container, blob=blob_name)
        try:
            bc.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="image/jpeg"))
            logger.info("✅ Uploaded %s successfully", img_path)
        except Exception as e:
            logger.error("❌ Failed to upload %s: %s", img_path, e)
            raise RuntimeError(f"Failed to upload {img_path} to Azure Blob Storage: {e}")

        if base_no_q:
            url = f"{base_no_q}/{blob_name}{sas_q}"
        else:
            url = f"https://{account}.blob.core.windows.net/{container}/{blob_name}{sas_q}"
        urls.append(url)
        logger.debug("Generated blob URL: %s", url)

    logger.info("Finished uploading %d images", len(urls))
    return urls


def compute_folder_hash(folder_path):
    """Compute MD5 hash of all images in a folder."""
    logger.info("Computing folder hash for: %s", folder_path)

    hash_obj = hashlib.md5()
    image_count = 0
    total_size = 0
    
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(ALLOWED_IMG_EXTS)
    ])
    logger.info("Found %d image files", len(image_files))

    for fname in image_files:
        fpath = os.path.join(folder_path, fname)
        logger.debug("Processing file: %s", fpath)
        with open(fpath, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)
        image_count += 1
        total_size += os.path.getsize(fpath)

    folder_hash = hash_obj.hexdigest()
    logger.info("Computed hash=%s | images=%d | total_size=%d bytes", folder_hash, image_count, total_size)
    return folder_hash, image_count, total_size


def _azure_client() -> BlobServiceClient:
    if not (AZURE_ACCOUNT and AZURE_SAS):
        logger.error("Azure credentials missing. Please set AZURE_STORAGE_ACCOUNT and AZURE_BLOB_SAS in .env")
        raise RuntimeError("Azure credentials missing: set AZURE_STORAGE_ACCOUNT and AZURE_BLOB_SAS in .env")
    logger.debug("Creating Azure BlobServiceClient")
    return BlobServiceClient(
        account_url=f"https://{AZURE_ACCOUNT}.blob.core.windows.net",
        credential=AZURE_SAS
    )


def _sanitize_name(name: str) -> str:
    logger.debug("Sanitizing name: %s", name)
    name = name.strip().replace(" ", "_")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "", name)
    logger.debug("Sanitized name: %s", sanitized)
    return sanitized


def _upload_pdf_to_azure(pdf_path: str, output_name: str) -> str:
    """Upload a local PDF to Azure and return public/SAS URL."""
    logger.info("Uploading PDF: %s as %s", pdf_path, output_name)

    safe_name = _sanitize_name(output_name)
    with open(pdf_path, "rb") as f:
        data = f.read()
    file_id = hashlib.md5(data).hexdigest()
    blob_name = f"imageagent_uploads/{file_id}_{safe_name}"

    # Use secrets loaded at the top of the file
    account   = AZURE_ACCOUNT
    container = AZURE_CONTAINER
    base_no_q = AZURE_BASE
    sas_q     = AZURE_SAS

    svc = BlobServiceClient(account_url=f"https://{account}.blob.core.windows.net", credential=sas_q or None)
    bc = svc.get_blob_client(container=container, blob=blob_name)
    bc.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="application/pdf"))
    logger.info("✅ PDF uploaded successfully to blob: %s", blob_name)

    if base_no_q:
        url = f"{base_no_q}/{blob_name}{sas_q}"
    else:
        url = f"https://{account}.blob.core.windows.net/{container}/{blob_name}{sas_q}"
    
    logger.debug("Generated PDF URL: %s", url)
    return url


def images_to_pdf(image_paths, output_name="output.pdf"):
    """Build a PDF from given images, upload directly to Azure Blob, and return the blob URL."""
    logger.info("Starting PDF build | images=%d | output_name=%s", len(image_paths), output_name)

    page_w, page_h = 595, 842
    tmp_pdf = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fp:
            tmp_pdf = fp.name
        logger.debug("Temporary PDF file created: %s", tmp_pdf)

        pdf = FPDF(unit="pt", format="A4")
        for img_path in image_paths:
            logger.info("Processing image for PDF: %s", img_path)
            if img_path.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                try:
                    resp = requests.get(img_path)
                    cover = Image.open(BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    logger.warning("⚠️ Skipping URL %s due to error: %s", img_path, e)
                    continue
            else:
                try:
                    cover = Image.open(img_path).convert("RGB")
                except Exception as e:
                    logger.warning("⚠️ Skipping file %s due to error: %s", img_path, e)
                    continue

            w, h = cover.size
            ratio = min(page_w / w, page_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            logger.debug("Resizing image %s -> %dx%d", img_path, new_w, new_h)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_fp:
                tmp_img = temp_img_fp.name
            try:
                cover.resize((new_w, new_h)).save(tmp_img, "JPEG")
                pdf.add_page()
                pdf.image(tmp_img, x=(page_w - new_w)//2, y=(page_h - new_h)//2, w=new_w, h=new_h)
            finally:
                try: os.remove(tmp_img)
                except Exception: pass

        pdf.output(tmp_pdf, "F")
        logger.info("PDF built successfully: %s", tmp_pdf)

        url = _upload_pdf_to_azure(tmp_pdf, output_name)
        logger.info("📄 PDF uploaded to Azure: %s", url)
        return url
    finally:
        if tmp_pdf:
            try:
                os.remove(tmp_pdf)
                logger.debug("Temporary PDF deleted: %s", tmp_pdf)
            except Exception:
                logger.warning("Failed to delete temporary PDF: %s", tmp_pdf)
