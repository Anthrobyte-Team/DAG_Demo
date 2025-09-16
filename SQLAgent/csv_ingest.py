import io, re, time, json, logging
import pandas as pd
from sqlalchemy import text
from sqlalchemy.types import Integer, Float, DateTime, String, Text
from sqlalchemy.exc import SQLAlchemyError

from SQLAgent.db_connection import engine 
from SQLAgent.llm_setup import llm

logger = logging.getLogger(__name__)

def sanitize_table_name(name: str) -> str:
    """
    Produce a safe, short table_name. Keep it lowercase.
    Replace spaces with underscores; allow hyphens if you want,
    but avoid special chars that break SQL identifiers.
    """
    name = name.lower()
    name = re.sub(r'[^\w\-]+', '_', name)  # allow letters, digits, underscore, hyphen
    name = re.sub(r'_{2,}', '_', name).strip('_')
    # add timestamp to avoid collision
    suffix = int(time.time() % 100000)
    return f"{name}_{suffix}"

def map_pd_dtype_to_sqlalchemy(dtype) -> object:
    """Simple dtype -> SQLAlchemy type mapper (extend as needed)."""
    if pd.api.types.is_integer_dtype(dtype):
        return Integer()
    if pd.api.types.is_float_dtype(dtype):
        return Float()
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return DateTime()
    # fallback
    return Text()

def create_table_from_df(df: pd.DataFrame, table_name: str, if_exists='fail'):
    """Write DataFrame to SQL using SQLAlchemy engine."""
    # build dtype map for to_sql
    dtype_map = {col: map_pd_dtype_to_sqlalchemy(dt) for col, dt in zip(df.columns, df.dtypes)}
    df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False, dtype=dtype_map)
    logger.info(f"Created table {table_name} with {len(df.columns)} columns and {len(df)} rows.")

def run_query_and_fetch(query: str):
    """Execute the query and return a list of dict rows (safe for JSON)."""
    try:
        with engine.connect() as conn:
            res = conn.execute(text(query))
            rows = [dict(r) for r in res.mappings().all()]
        return rows
    except SQLAlchemyError as e:
        logger.error(f"Query execution error: {e}")
        return {"error": str(e)}

def generate_description_with_llm(column_names, sample_rows):
    """
    Use your llm (llm.invoke(...).content) to create a short metadata description.
    Keep this prompt short so token cost is low: provide columns + 3 sample rows only.
    """
    prompt = (
        "You are a data catalog assistant. Given a list of columns and 3 example rows, "
        "write a concise (2-3 sentences) plain-English description of what this dataset contains "
        "and the primary entities it describes.\n\n"
        f"Columns: {column_names}\n\n"
        f"Example rows:\n{sample_rows}\n\n"
        "Output only the short description (no bullet points)."
    )
    try:
        response = llm.invoke(prompt)
        # chain uses response.content in other places
        return getattr(response, "content", str(response)).strip()
    except Exception as e:
        logger.error(f"LLM description generation failed: {e}")
        return ""

def insert_metadata_row(metadata: dict):
    """
    Insert a row into csv_master_metadata. Use parameterized SQL to avoid injection.
    """
    insert_sql = text("""
    INSERT INTO csv_master_metadata
    (file_name, user_name, description, table_name, column_names)
    VALUES (:file_name, :user_name, :description, :table_name, :column_names)
    """)
    params = {
        "file_name": metadata.get("file_name"),
        "user_name": metadata.get("user_name"),
        "description": metadata.get("description"),
        "table_name": metadata.get("table_name"),
        "column_names": json.dumps(metadata.get("column_names", [])),
    }
    try:
        with engine.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Inserted metadata row for table {metadata.get('table_name')}")
    except Exception as e:
        logger.exception("Failed to insert metadata row")
        raise 

def ingest_csv_file(file_stream, filename, user_name, description=None, want_chart=False):
    """
    Main orchestration:
      - read CSV (pandas)
      - create table
      - generate description if missing
      - run a default query
      - optionally generate chart code (uses your chain.generate_chart_code)
      - write metadata into master table
    """
    # 1) read CSV into pandas
    try:
        if isinstance(file_stream, (bytes, bytearray)):
            file_buffer = io.BytesIO(file_stream)
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_csv(file_stream)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise

    # 2) generate table name and create table
    base_name = re.sub(r'\.csv$', '', filename, flags=re.IGNORECASE)
    table_name = sanitize_table_name(base_name)

    # create table (replace if exists; you can change policy)
    try:
        create_table_from_df(df, table_name, if_exists='replace')
    except Exception as e:
        logger.exception("Failed to create table from CSV")
        raise

    # 3) column names + sample rows
    column_names = df.columns.tolist()
    sample_rows = df.head(3).to_dict(orient='records')  # small sample

    # 4) description
    if not description or description.strip() == "":
        description = generate_description_with_llm(column_names, sample_rows)

    # 5) write to metadata table
    metadata = {
        "file_name": filename,
        "user_name": user_name,
        "description": description,
        "table_name": table_name,
        "column_names": column_names,
    }
    insert_metadata_row(metadata)
    return metadata
