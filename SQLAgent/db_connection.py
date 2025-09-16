import os, logging
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import streamlit as st
import certifi

MYSQL_URI = st.secrets["mysql"]["uri"]

# Create SQLAlchemy engine
try:
    engine = create_engine(MYSQL_URI, pool_pre_ping=True, pool_size=10, max_overflow=20, connect_args={"ssl": {
        "ca": certifi.where()
    }})
    logging.info("SQLAlchemy engine created.")
except Exception as e:
    logging.error(f"Failed to create SQLAlchemy engine: {e}")
    engine = None

# create SQLDatabase for your langchain usage
try:
    db = SQLDatabase(engine)
    logging.info("Successfully connected to the MySQL database (SQLDatabase).")
except Exception as e:
    logging.error(f"Failed to connect to the MySQL database (SQLDatabase): {e}")
    db = None

# MYSQL_URI = os.getenv("MYSQL_URI")
# try:
#     db = SQLDatabase.from_uri("sqlite:///Chinook.db")
#     logging.info("Successfully connected to the database.")
# except Exception as e:
#     logging.error(f"Failed to connect to the database: {e}")
#     db = None
