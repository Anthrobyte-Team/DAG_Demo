import os, sys, time, uuid, hashlib, base64
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
from decimal import Decimal
import tempfile
import zipfile

# Logging
import logging
from logging_setup import setup_logging

# Plotly
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SKLEARN
from sklearn.inspection import permutation_importance

#Langchain
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

setup_logging()
logger = logging.getLogger("app")

# SQLAgent
from SQLAgent.llm_setup import llm
from SQLAgent.csv_ingest import ingest_csv_file
from SQLAgent.chain import write_query, execute_query, generate_answer, generate_chart_code

# RAG core
from RAGAgentCloud.prompt import prompt, condense_prompt
from RAGAgentCloud.ingest import load_and_split_docs, load_and_split_pdf
from RAGAgentCloud.retriever import build_retriever_from_docs, get_collection_retriever
from RAGAgentCloud.smalltalk import detect_smalltalk_kind, choose_smalltalk
from RAGAgentCloud.sources_utils import render_sources, filter_sources_by_answer, build_relevant_pdf, append_imageagent_source

# WebSearchAgent
from WebSearchAgent.main import web_search
from WebSearchAgent.llm import summarize_news

# Image Agent
from ImageAgent.sql_store import exists_by_doc_hash
from ImageAgent.utils import compute_folder_hash
from ImageAgent.ingestion import handle_folder_upload
from ImageAgent.milvus_client import get_milvus_client, get_collection_name

# ML Agent
from MLAgent.config import REQUIRED_COLUMNS
from MLAgent.data_utils import (
    load_csv, check_columns, apply_filters, apply_filters_v2,
    canonicalize_filters,
    extract_target_qty_from_question, infer_qty_defaults
)
from MLAgent.analytics import (
    sku_features, vendor_features, cluster_table_train,
    fig_vendor_clusters, fig_sku_clusters, fig_discount_multi,
    discount_curve_training, discount_curve_training_enriched,
    discount_curve_grid, predict_for_vendor, predict_for_vendor_enriched, fig_residuals_scatter
)
from MLAgent.price_anomaly import detect_anomalies, score_price_model,fit_price_model, get_price_model


# Sessions
from sessions import (
    list_sessions, new_session, load_session, save_session, delete_session,
    list_orc_sessions, new_orc_session, load_orc_session, save_orc_session, delete_orc_session,
    jsonable_messages, list_decision_sessions, new_decision_session, load_decision_session, save_decision_session, delete_decision_session
)

# Orchestrator
from auth import login_gate, logout_sidebar
from orchestrator_setup import decision_orchestrator, orchestrator, build_tables_prompt, clear_llm_memory, migrate_llm_memory

# Page Styling
st.markdown("""
    <style>
    /* Make the Streamlit header transparent/glass-like */
    header {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(8px) !important;
        box-shadow: 0 2px 16px 0 rgba(0,0,0,0.04) !important;
        border-bottom: 1px solid #ffefe5 !important;
    }
    /* Inject logo into the header using ::before */
    header::before {
        content: "";
        display: inline-block;
        vertical-align: middle;
        background: url('https://vragstorageaccount.blob.core.windows.net/logos/Anthrobyte_Logo.png') no-repeat center center;
        background-size: contain;
        width: 220px;
        height: 220px;
        margin-right: 18px;
        margin-left: 12px;
        position: relative;
        top: 4px;
    }
    /* Optional: Hide Streamlit's default logo if present */
    header .st-emotion-cache-1avcm0n {
        display: none !important;
    }
    body, .stApp {
        background: radial-gradient(ellipse at 53% 4%, #ffefe5 0%, #fff 100%);
        animation: gradientBG 12s ease-in-out infinite alternate;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    html, body, [class^="st-"], [class*=" st-"] {
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif !important;
    }

    .login-split-container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: stretch;
        min-height: 90vh;
        width: 100vw;
        margin: 0 -3rem;
    }
    .login-left {
        flex: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding-left: 6vw;
        padding-right: 2vw;
    }
    .login-logo {
        margin-bottom: 2.5rem;
        margin-top: 2rem;
        width: 160px;
    }
    .login-heading {
        font-size: 2.3rem;
        font-weight: 800;
        color: #232323;
        margin-bottom: 0.7em;
        line-height: 1.1;
    }
    .login-tagline {
        font-size: 1.15rem;
        color: #232323cc;
        margin-bottom: 2.5em;
        font-weight: 500;
    }
    .login-right {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding-right: 8vw;
        min-width: 400px;
    }
    .login-box {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(255, 102, 0, 0.09);
        padding: 2.5em 2em 2em 2em;
        min-width: 340px;
        max-width: 400px;
        border: 1.5px solid #ff6600;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .login-box h2 {
        color: #ff6600;
        font-weight: 800;
        margin-bottom: 1.2em;
        text-align: center;
    }

    .stTextInput>div>div>input, .stTextArea textarea {
        border-radius: 10px !important;
        border: 1.5px solid #ff6600 !important;
        font-size: 1.05rem !important;
        padding: 0.7em !important;
        background: #fff7f2 !important;
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff6600 0%, #ff9a3c 100%);
        color: #fff !important;
        font-weight: 700 !important;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.5em;
        font-size: 1.1rem;
        transition: box-shadow 0.2s, background 0.2s;
        box-shadow: 0 2px 8px 0 rgba(255, 102, 0, 0.07);
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff9a3c 0%, #ff6600 100%);
        box-shadow: 0 4px 16px 0 rgba(255, 102, 0, 0.15);
    }

    .stAlert {
        border-radius: 12px !important;
        font-size: 1.05rem !important;
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif !important;
    }
    .stAlert[data-testid="stAlert-error"] {
        background: #fff0f0 !important;
        color: #d32f2f !important;
        border-left: 5px solid #d32f2f !important;
    }
    .stAlert[data-testid="stAlert-success"] {
        background: #eafff2 !important;
        color: #1a7f37 !important;
        border-left: 5px solid #1a7f37 !important;
    }
    .stAlert[data-testid="stAlert-warning"] {
        background: #fffbe6 !important;
        color: #ff9800 !important;
        border-left: 5px solid #ff9800 !important;
    }
    .stAlert[data-testid="stAlert-info"] {
        background: #e3f2fd !important;
        color: #1976d2 !important;
        border-left: 5px solid #1976d2 !important;
    }

    .stExpander {
        border-radius: 14px !important;
        background: #fff7f2 !important;
        border: 1.5px solid #ff6600 !important;
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif !important;
    }

    /* ===================== EXPANDER CHEVRON FIX (NO FONTS) ===================== */
    /* 1) Hide the ligature text inside the icon span that Streamlit renders */
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"] * {
        color: transparent !important;
        font-size: 0 !important;
    }

    /* 2) Reserve space for the icon so layout doesn’t jump */
    [data-testid="stExpander"] summary > :first-child {
        width: 28px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
    }

    /* 3) Draw our own arrow using plain Unicode (works with any font) */
    /* default: collapsed */
    [data-testid="stExpander"] details:not([open]) summary > :first-child::after {
        content: '▸';
        color: #333 !important;
        font-size: 20px !important;
        line-height: 1 !important;
    }

    /* expanded (when <details open> is present) */
    [data-testid="stExpander"] details[open] summary > :first-child::after {
        content: '▾';
        color: #333 !important;
        font-size: 20px !important;
        line-height: 1 !important;
    }

    /* fallback for builds that use aria-expanded on <summary> */
    [data-testid="stExpander"] summary[aria-expanded="false"] > :first-child::after {
        content: '▸';
    }
    [data-testid="stExpander"] summary[aria-expanded="true"] > :first-child::after {
        content: '▾';
    }


    /* (Optional safety) If a dedicated toggle node exists, blank it too */
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpanderToggleIcon"] * {
        color: transparent !important;
        font-size: 0 !important;
    }
    /* =================== END EXPANDER CHEVRON FIX (NO FONTS) =================== */
            
    /* ==== SIDEBAR TOGGLE ICON FIX (NO FONT DEPENDENCY) ==== */

    /* Header toggle: hide the ligature text Streamlit renders */
    header [data-testid="stIconMaterial"],
    header [data-testid="stIconMaterial"] * {
    color: transparent !important;
    font-size: 0 !important;
    }

    /* Draw a clean double-arrow in the header toggle */
    header [data-testid="stIconMaterial"]::after {
    content: '»';                 /* double arrow right */
    color: #333 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    }

    /* If Streamlit also renders an icon inside the sidebar itself, clean that too */
    [data-testid="stSidebar"] [data-testid="stIconMaterial"],
    [data-testid="stSidebar"] [data-testid="stIconMaterial"] * {
    color: transparent !important;
    font-size: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stIconMaterial"]::after {
    content: '«';                 /* double arrow left */
    color: #333 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    }
    /* ==== END SIDEBAR TOGGLE ICON FIX ==== */


    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header .st-emotion-cache-1avcm0n { display: none !important; }

    @media (max-width: 900px) { }

    /* Style the first container in the right column as the login box */
    section.main > div > div > div:nth-child(2) > div > div:first-child {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(255, 102, 0, 0.09);
        padding: 2.5em 2em 2em 2em;
        min-width: 320px;
        max-width: 370px;
        border: 1.5px solid #ff6600;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

# bootstrap logging 
setup_logging()
logger = logging.getLogger("app")
logger.info("App started.")
 
def show_price_importance(bundle, df_slice: pd.DataFrame):
    """Permutation importance; best-effort."""
    try:
        feats = (bundle.get("num_cols", []) or []) + (bundle.get("cat_cols", []) or [])
        if not feats:
            st.info("Feature importance unavailable (bundle has no feature list)."); return
        pre = bundle.get("pre", None); model = bundle["automl"]
        X = pre.transform(df_slice[feats]) if pre is not None else df_slice[feats]
        y = df_slice["Unit_Price_USD"].astype(float).values
        r = permutation_importance(model, X, y, n_repeats=5, random_state=7)
        imp = pd.DataFrame({"feature": feats, "importance": r.importances_mean}).sort_values("importance", ascending=False)
        st.subheader("Why were rows flagged? (Permutation importance)")
        st.dataframe(imp.head(10).reset_index(drop=True))
    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")

# --- paths / sys.path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLAGENT_DIR = os.path.join(BASE_DIR, "SQLAgent")
RAGAGENT_DIR = os.path.join(BASE_DIR, "RAGAgentCloud")
sys.path.append(SQLAGENT_DIR)
sys.path.append(RAGAGENT_DIR)
logger.info("Paths set and modules imported.")

# --- Streamlit Title---
st.set_page_config(page_title="Anthrobyte AI Advisor", layout="wide")

# Add logo and title in header
st.markdown(
    """
    <div style='display: flex; align-items: center; gap: 1.2rem; margin-bottom: 2.5rem; margin-top: 2.5rem;'>
        <img src="https://vragstorageaccount.blob.core.windows.net/logos/AnthroByte_small_logo.png" alt="Anthrobyte Logo" style="height: 52px;">
        <span style='font-size:2.3rem; font-weight:800; color:#232323; font-family:Montserrat,Segoe UI,sans-serif;'>Anthrobyte AI Advisor</span>
    </div>
    """,
    unsafe_allow_html=True
)

def _qhash(q: str) -> str:
    return hashlib.md5((q or "").encode()).hexdigest()

# --- RAG CRC caching (build once, reuse) ---
@st.cache_resource(show_spinner=False)
def _build_crc_cached(index_version: int):
    """
    Build the ConversationalRetrievalChain once and reuse it across reruns.
    'index_version' forces a rebuild after you ingest new docs.
    """
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_collection_retriever(),
        condense_question_prompt=condense_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

def _get_crc():
    if "rag_index_version" not in st.session_state:
        st.session_state.rag_index_version = 0
    return _build_crc_cached(st.session_state.rag_index_version)

def _refresh_rag_chain():
    # bump version so cache invalidates, then rebuild immediately
    st.session_state.rag_index_version = st.session_state.get("rag_index_version", 0) + 1
    st.session_state.rag_chain = _get_crc()

# --- ECharts option cleaning helpers ---
def snake_to_camel(s):
    parts = s.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

def convert_keys_to_camel(obj):
    if isinstance(obj, dict):
        return {snake_to_camel(k): convert_keys_to_camel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_camel(i) for i in obj]
    else:
        return obj

def ensure_jsonable_echarts_option(obj):
    # Remove JsCode handling entirely
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif 'numpy' in str(type(obj)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    elif isinstance(obj, dict):
        return {str(k): ensure_jsonable_echarts_option(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_jsonable_echarts_option(v) for v in obj]
    else:
        return str(obj)


def clean_for_json(obj):
    """Recursively convert JsCode objects to strings for JSON serialization."""
    try:
        from streamlit_echarts import JsCode
    except ImportError:
        class JsCode: pass
    if isinstance(obj, JsCode):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    else:
        return obj

# Authentication
login_gate() 
logout_sidebar() 

# --- ensure in-memory structures exist even if user lands directly on tabs later ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = _get_crc()
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "tab3_messages" not in st.session_state:
    st.session_state.tab3_messages = []
if "orc_turns" not in st.session_state:
    st.session_state["orc_turns"] = []
# (optional) top-level followups cache
if "orc_followups" not in st.session_state:
    st.session_state["orc_followups"] = []

# --- current user (from auth) ---
user = st.session_state.get("user", {"id": "anonymous"})
user_id = user.get("id", "anonymous")

# Sidebars: Session pickers
# ---- Orchestrator sessions ----
with st.sidebar:
    st.markdown("### Orchestrator Sessions")
    orc_sessions = list_orc_sessions(user_id)
    ids = [s["id"] for s in orc_sessions]

    # Keep picker aligned to active session
    cur = st.session_state.get("orc_session_id")
    if cur and (st.session_state.get("orc_session_picker") not in ids or
                st.session_state.get("orc_session_picker") != cur):
        st.session_state["orc_session_picker"] = cur

    # SINGLE selectbox (remove the second one entirely)
    picked_orc = st.selectbox(
        "Open orchestrator session",
        options=ids,
        format_func=lambda sid: next((s["title"] for s in orc_sessions if s["id"] == sid), sid),
        key="orc_session_picker",
    )
    if picked_orc and st.session_state.get("orc_session_id") != picked_orc:
        rec = load_orc_session(user_id, picked_orc)
        st.session_state["orc_session_id"] = picked_orc
        st.session_state["tab3_messages"] = rec.get("messages", [])
        st.session_state["orc_turns"] = rec.get("turns", [])          # keep turns in sync
        st.session_state["orc_followups"] = rec.get("followups", [])  # optional top-level cache
        st.rerun()

    # Create new orchestrator chat
    if st.button("➕ New Orchestrator Chat"):
        sid = new_orc_session(user_id, title="New orchestrator chat")
        st.session_state["orc_session_id"] = sid
        st.session_state["tab3_messages"] = []
        st.session_state["orc_turns"] = []
        st.rerun()

    # Delete controls (keep these, just drop the extra selectbox)
    if orc_sessions:
        st.caption("Manage Orchestrator chats")
        for s in orc_sessions:
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.write(f"• {s['title']}")
            with c2:
                if st.button("🗑️", key=f"del_orc_{s['id']}", help=f"Delete '{s['title']}'"):
                    if delete_orc_session(user_id, s["id"]):
                        if st.session_state.get("orc_session_id") == s["id"]:
                            st.session_state["orc_session_id"] = None
                            st.session_state["tab3_messages"] = []
                            st.session_state["orc_turns"] = []
                        st.session_state.pop("orc_last_result", None)
                        st.session_state.pop("ml_preset", None)
                        for k in list(st.session_state.keys()):
                            if k.startswith("ml_"):
                                st.session_state.pop(k)
                        st.rerun()
                    else:
                        st.error("Could not delete this chat.")

# ---- RAG sessions ----
with st.sidebar:
    st.markdown("### RAG Chat Sessions")
    rag_sessions = list_sessions(user_id)

    # Create new RAG chat
    if st.button("➕ New RAG Chat"):
        sid = new_session(user_id, title="New chat")
        st.session_state["rag_session_id"] = sid
        st.session_state["rag_messages"] = []
        st.rerun()

    # Quick picker
    if rag_sessions:
        picked = st.selectbox(
            "Open session",
            options=[s["id"] for s in rag_sessions],
            format_func=lambda sid: next((s["title"] for s in rag_sessions if s["id"] == sid), sid),
            key="rag_session_picker",
        )
        if st.session_state.get("rag_session_id") != picked:
            rec = load_session(user_id, picked)
            st.session_state["rag_session_id"] = picked
            st.session_state["rag_messages"] = rec.get("messages", [])
            st.rerun()

        # Delete controls
        st.caption("Manage RAG chats")
        for s in rag_sessions:
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.write(f"• {s['title']}")
            with c2:
                if st.button("🗑️", key=f"del_rag_{s['id']}", help=f"Delete '{s['title']}'"):
                    if delete_session(user_id, s["id"]):
                        if st.session_state.get("rag_session_id") == s["id"]:
                            st.session_state["rag_session_id"] = None
                            st.session_state["rag_messages"] = []
                        st.rerun()
                    else:
                        st.error("Could not delete this chat.")

# App tabs
tabs = st.tabs(["SQL Agent", "RAG Agent", "Image Agent", "Web Agent", "Orchestrator","Decision Support", "RAG Chat", "ML Agent"])
logger.info("Tabs initialized.")

# TAB 1: SQLAgent 
with tabs[0]:
    logger.info("Entered SQL Agent tab.")
    csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_upload")
    user_name = st.text_input("Enter your name (required):")
    description = st.text_area("Short description of the table (optional):")

    if csv_file is not None:
        logger.info("CSV file uploaded.")
        st.info(f"Table will be created from file: '{csv_file.name}'")
        if st.button("Upload CSV"):
            if not user_name.strip():
                st.warning("Please enter your name before uploading.")
            else:
                try:
                    metadata = ingest_csv_file(
                        csv_file,
                        csv_file.name,
                        user_name,
                        description,
                        want_chart=False
                    )
                    logger.info(f"CSV '{csv_file.name}' ingested. Table '{metadata['table_name']}' created. Metadata row added.")
                    st.success(f"Table '{metadata['table_name']}' created and metadata added.")
                except Exception as e:
                    logger.error(f"Error during CSV ingestion: {e}")
                    st.error(f"Error uploading CSV: {e}")

    user_q_sql = st.text_input("Ask a question about your uploaded CSV", key="sql_query")

    if user_q_sql:
        logger.info(f"User SQL question: {user_q_sql}")
        tables_prompt = build_tables_prompt()
        state = {
            "question": user_q_sql,
            "tables_prompt": tables_prompt,
        }
        try:
            # Step 1: Generate SQL query
            query_out = write_query(state)
            logger.info(f"SQL query generated: {query_out.get('query')}")
            state.update(query_out)
            # Step 2: Execute SQL query
            result_out = execute_query(state)
            logger.info("SQL query executed.")
            state.update(result_out)
            # Step 3: Generate answer
            answer_out = generate_answer(state)
            logger.info("SQL answer generated.")
            st.write("**LLM SQL Response:**")
            st.info(answer_out["answer"])
            # Optional: Chart code
            chart_out = generate_chart_code(state)
            logger.info("Chart code generated.")
            if chart_out["chart_code"]:
                chart_code = chart_out["chart_code"].strip()
                # Remove code fences if present
                if chart_code.startswith("```"):
                    parts = chart_code.split("```")
                    if len(parts) >= 2:
                        chart_code = parts[1].strip()

                local_vars = {}
                try:
                    if "datetime." in chart_code:
                        chart_code = "import datetime\n" + chart_code
                    exec(chart_code, {"go": go, "pd": pd}, local_vars)
                    chart = None
                    for val in local_vars.values():
                        if isinstance(val, go.Figure):
                            chart = val
                            break
                    if chart:
                        st.subheader("Chart")
                        st.plotly_chart(chart)
                        logger.info("Chart rendered successfully.")
                    else:
                        st.info("No chart generated. The chart code did not produce a Plotly figure.")
                        logger.warning("No chart generated from chart code.")
                except Exception as e:
                    logger.error(f"Error rendering chart: {e}")
                    st.error(f"Error rendering chart: {e}")
        except Exception as e:
            logger.error(f"Error processing SQL question: {e}")
            st.error(f"Error: {e}")

# TAB 2: RAGAgentCloud
with tabs[1]:
    logger.info("Entered RAG Agent tab.")
    # build chain once
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = _get_crc()
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
        logger.info("Initialized rag_messages in session state.")

    # indexing controls
    mode = st.radio("Choose input mode:", ["URL(s)", "PDF(s)"], key="rag_mode")
    logger.info(f"RAG input mode selected: {mode}")

    if mode == "URL(s)":
        urls_text = st.text_area("Enter one or more URLs (one per line):", key="rag_urls_text")
        if st.button("Index Website(s)", key="rag_index_urls") and urls_text.strip():
            urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
            total_chunks = 0
            for u in urls:
                with st.spinner(f"Indexing: {u}"):
                    logger.info(f"Indexing URL: {u}")
                    docs = load_and_split_docs(u)
                    total_chunks += len(docs)
                    build_retriever_from_docs(docs)
            st.success(f"✅ Finished. Total chunks processed: {total_chunks}")
            logger.info(f"Finished indexing URLs. Total chunks: {total_chunks}")
            _refresh_rag_chain()
            st.toast("Search index refreshed.", icon="✅")
            logger.info("Search index refreshed after URL indexing.")

    else:
        files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True, key="rag_pdf_upl")
        if files and st.button("Index PDF(s)", key="rag_index_pdfs"):
            total_chunks = 0
            for f in files:
                with st.spinner(f"Indexing: {f.name}"):
                    logger.info(f"Indexing PDF: {f.name}")
                    docs = load_and_split_pdf(f)
                    total_chunks += len(docs)
                    build_retriever_from_docs(docs)
            st.success(f"✅ Finished. Total chunks processed: {total_chunks}")
            logger.info(f"Finished indexing PDFs. Total chunks: {total_chunks}")
            _refresh_rag_chain()
            st.toast("Search index refreshed.", icon="✅")
            logger.info("Search index refreshed after PDF indexing.")

    # # render chat history (keeps source expanders)
    # for idx, m in enumerate(st.session_state.rag_messages):
    #     with st.chat_message(m["role"]):
    #         st.markdown(m["content"])
    #         if m.get("sources"):
    #             render_sources(m.get("question", ""), m["sources"], key_seed=f"rag_hist_{idx}")
    # logger.info("Rendered RAG chat history.")

# TAB3: Image Upload only
with tabs[2]:
    logger.info("Entered Image Upload tab.")
    st.header("Step 1: Upload Image Folder (as ZIP)")
    uploaded_zip = st.file_uploader(
        "Upload a ZIP file containing your image folder", type=["zip"]
    )

    if uploaded_zip:
        zip_name = os.path.splitext(uploaded_zip.name)[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            folder_path = os.path.join(temp_dir, zip_name)
            os.makedirs(folder_path, exist_ok=True)
            zip_path = os.path.join(temp_dir, uploaded_zip.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(folder_path)
            st.success(f"Folder '{zip_name}' uploaded and extracted.")

            # Compute hash for deduplication
            folder_hash, image_count, total_size = compute_folder_hash(folder_path)
            st.caption(f"Computed folder hash: `{folder_hash}`")

            if exists_by_doc_hash(folder_hash):
                st.warning("This folder has already been ingested. Skipping ingestion and upload.")
            else:
                if st.button("Ingest Folder"):
                    client = get_milvus_client()
                    handle_folder_upload(client, folder_path, get_collection_name())
                    st.success("Ingestion complete!")

# TAB 4: Web Search
with tabs[3]:
    logger.info("Entered Web Agent tab.")
    st.header("Web Search Agent")
    user_web_query = st.text_input("Enter your web search query", key="web_search_query")
    num_results = st.slider("Number of results", min_value=1, max_value=10, value=3)

    if user_web_query:
        logger.info(f"User web search query: {user_web_query}")
        with st.spinner("Searching the web..."):
            try:
                results = web_search(user_web_query, num_results=num_results)
                logger.info(f"Web search returned {len(results) if results else 0} results.")
                if results:
                    # show raw results
                    for idx, r in enumerate(results, 1):
                        st.markdown(f"**{idx}. [{r['title']}]({r['link']})**")
                        st.write(r['snippet'])
                        st.write(f"[Source Link]({r['link']})")

                    # summarize
                    formatted_results = [f"{r['title']}\n{r['link']}\n{r['snippet']}" for r in results]
                    summary = summarize_news(formatted_results)
                    st.subheader("Summary")
                    st.success(summary)
                    logger.info("Web search summary generated and displayed.")
                else:
                    st.info("No results found.")
                    logger.warning("No web search results found.")
            except Exception as e:
                logger.error(f"Error during web search: {e}")
                st.error(f"Error: {e}")

# TAB 5: Orchestrator
with tabs[4]:
    logger.info("Entered Orchestrator tab.")
    st.header("Ask a Question")
    cur_orc = st.session_state.get("orc_session_id")
    if cur_orc and st.button("Delete this Orchestrator chat"):
        if delete_orc_session(user_id, cur_orc):
            st.session_state["orc_session_id"] = None
            st.session_state["tab3_messages"] = []
            # also clear ML state
            st.session_state.pop("orc_last_result", None)
            st.session_state.pop("ml_preset", None)
            # clear any ML widget keys
            for k in list(st.session_state.keys()):
                if k.startswith("ml_"):
                    st.session_state.pop(k)
            try:
                clear_llm_memory(cur_orc)
            except Exception:
                pass
            st.rerun()
        else:
            st.error("Could not delete current chat.")
 
    if "tab3_messages" not in st.session_state:
        st.session_state.tab3_messages = []
        logger.info("Initialized tab3_messages in session state.")
 
    # Render past messages
    turns = st.session_state.get("orc_turns", [])
    turn_idx = 0
    pending_user = None

    for m in st.session_state.tab3_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "user":
                pending_user = m["content"]
            elif m["role"] == "assistant" and pending_user is not None:
                # attach followups for this user->assistant turn, if any
                if turn_idx < len(turns):
                    fu = (turns[turn_idx] or {}).get("followups", [])
                    if fu:
                        st.markdown("**Follow-up questions:**")
                        st.markdown("\n".join(f"- {q}" for q in fu))
                turn_idx += 1
                pending_user = None

            
    def render_followup_chips(result):
        sugs = ((result or {}).get("suggested_followups") or {}).get("followups", [])
        if not sugs:
            return
        st.markdown("**Follow-up questions:**")
        st.markdown("\n".join(f"- {q}" for q in sugs))

        # Auto-submit a follow-up if a chip was clicked on previous rerun
    auto_q = None

    user_q_direct = st.chat_input("Your question", key="direct_query")
    if user_q_direct:
        user_q_direct = user_q_direct.strip()

    if not user_q_direct:
        pass
    else:
        logger.info(f"User orchestrator question: {user_q_direct}")

        st.session_state.tab3_messages.append({"role": "user", "content": user_q_direct})
        with st.chat_message("user"):
            st.markdown(user_q_direct)
 
        # Build chat pairs from the UI history so backend can compact & condense
        ui_pairs = []
        pending_user = None
        for m in st.session_state.tab3_messages:
            if m["role"] == "user":
                pending_user = m["content"]
            elif m["role"] == "assistant" and pending_user is not None:
                ui_pairs.append((pending_user, m["content"]))
                pending_user = None

        session_id = st.session_state.get("orc_session_id") or st.session_state.setdefault("_orc_ephemeral_session_id", str(uuid.uuid4()))
        force = bool(st.session_state.pop("_orc_force_llm", False))
        logger.info(f"Processing auto-submitted question: '{auto_q}'. Force LLM agent: {force}") # <<< ADDED LOG
        with st.spinner("Thinking..."):
            try:
                result = orchestrator(user_q_direct, chat_history=ui_pairs, session_id=session_id)
                st.session_state["orc_last_result"] = result
                logger.info("Orchestrator returned result.")

                final = result.get("final_answer", "")
                sugs_obj = (result or {}).get("suggested_followups") or {}
                sugs = sugs_obj.get("followups", [])
                mode = sugs_obj.get("followup_mode", "multi")

                # Build + stash structured turn (user, assistant, followups, timestamp)
                turn = {
                    "user": user_q_direct,
                    "assistant": final,
                    "followups": sugs,
                    "ts": time.time(),
                }
                # Mirror in session_state for rendering on reopen
                prev_turns = st.session_state.get("orc_turns", [])
                st.session_state["orc_turns"] = prev_turns + [turn]


                with st.chat_message("assistant"):
                    st.success(final)
                    # Inline follow-up rendering with guard for empty separator
                    sugs = ((result or {}).get("suggested_followups") or {}).get("followups", [])
                    if sugs:
                        st.markdown("**Follow-up questions:**")
                        st.markdown("\n".join(f"- {q}" for q in sugs))
 
                st.session_state.tab3_messages.append({"role": "assistant", "content": final})
 
                with st.expander("📋 Orchestration Plan & Sub-Results"):
                    st.json(clean_for_json(result))

                # Render sources for RAG sub-questions if present
                for idx, sub_result in enumerate(result.get("results", [])):
                    if sub_result.get("agent") == "rag" and sub_result.get("sources"):
                        filtered_sources = filter_sources_by_answer(sub_result["sources"], sub_result.get("answer", ""), sub_result["question"])
                        final_sources = append_imageagent_source(filtered_sources, sub_result["question"])
                        render_sources(sub_result["question"], final_sources, key_seed=f"tab3_{idx}")
                        logger.info(f"Rendered sources for RAG sub-question: {sub_result.get('question')}")
 
                # Render sources for Web sub-questions if present
                for idx, sub_result in enumerate(result.get("results", [])):
                    if sub_result.get("agent") == "web" and sub_result.get("sources"):
                        st.subheader(f"🌐 Sources for: {sub_result['question']}")
                        for r in sub_result["sources"]:
                            st.markdown(f"- [{r['title']}]({r['link']}) — {r['snippet']}")
                        logger.info(f"Rendered sources for Web sub-question: {sub_result.get('question')}")        

                # Render SQL charts if present; also capture ML preset
                for sub_result in result.get("results", []):
                    if sub_result.get("agent") == "ml":
                        st.session_state["ml_preset"] = sub_result
                        st.info("ML task detected — switch to the *ML Agent* tab to run with suggested defaults.")

                    if sub_result.get("agent") == "sql" and sub_result.get("chart_code"):
                        code = sub_result["chart_code"]
                        if isinstance(code, str):
                            code = code.strip()
                            # Remove code fences if present
                            if code.startswith("```"):
                                parts = code.split("```")
                                if len(parts) >= 2:
                                    code = parts[1].strip()
                            # Remove a leading 'python' line if present
                            if code.lower().startswith("python"):
                                code = code[len("python"):].lstrip()

                        # --- guardrail: block dangerous ops ---
                        forbidden = ["import os", "import sys", "subprocess", "shutil", "open(", "_import_", "eval(", "exec(", "socket", "requests","import streamlit_echarts", "JsCode("]
                        if any(bad in code for bad in forbidden):
                            st.error("Blocked unsafe code in generated chart.")
                            logger.error("Unsafe code detected in chart_code.")
                            continue

                        local_vars = {}
                        try:
                            safe_globals = {
                                "pd": pd, "np": np,
                                "go": go, "make_subplots": make_subplots,
                                "pdk": pdk,
                                "Decimal": Decimal,  # <-- Add this line
                            }
                            if "datetime." in code:
                                code = "import datetime\n" + code

                            exec(code, safe_globals, local_vars)

                            deck = local_vars.get("deck", None)       # pydeck.Deck
                            option = local_vars.get("option", None)   # ECharts option dict
                            fig = None                                 # Plotly go.Figure

                            for v in local_vars.values():
                                if hasattr(v, "to_plotly_json"):  # Plotly figure-like
                                    fig = v
                                    break

                            st.subheader(f"Chart for: {sub_result.get('question')}")

                            if deck is not None:
                                st.pydeck_chart(deck)
                                logger.info("Rendered pydeck chart.")

                            elif option is not None:
                                option = ensure_jsonable_echarts_option(option)
                                option = convert_keys_to_camel(option)
                                st_echarts(options=option, height="520px")
                                logger.info("Rendered ECharts chart.")

                            elif fig is not None:
                                if not getattr(fig.layout, "template", None):
                                    fig.update_layout(template="presentation")
                                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                                logger.info("Rendered Plotly chart.")

                            else:
                                st.info("No chart generated. The chart code did not produce a renderable figure.")
                                logger.warning("No renderable object found (deck/option/fig).")

                        except Exception as e:
                            logger.error(f"Error rendering chart for sub-question: {e}")
                            st.error(f"Error rendering chart: {e}")
 
                # Auto-save orchestrator session
                user = st.session_state.get("user", {"id": "anonymous"})
                user_id = user.get("id", "anonymous")
                sid = st.session_state.get("orc_session_id")
                if not sid:
                    title = (user_q_direct[:60] + "…") if len(user_q_direct) > 60 else user_q_direct
                    sid = new_orc_session(user_id, title=title or "Orchestrator chat")
                    st.session_state["orc_session_id"] = sid
 
                rec = load_orc_session(user_id, sid)
                title = rec.get("title") or ((user_q_direct[:60] + "…") if len(user_q_direct) > 60 else user_q_direct) or "Orchestrator chat"

                # store latest followups at top-level, and append this turn
                save_orc_session(
                    user_id,
                    sid,
                    st.session_state.tab3_messages,
                    title=title,
                    followups=sugs,
                    turns=[turn],
                )

                # NEW: migrate LLM memory from the ephemeral id to the persisted session id
                old_ephem = st.session_state.get("_orc_ephemeral_session_id")
                if old_ephem and old_ephem != sid:
                    try:
                        migrate_llm_memory(old_ephem, sid)
                        st.session_state["_orc_ephemeral_session_id"] = sid
                    except Exception as e:
                        logger.warning(f"Could not migrate LLM memory: {e}")
 
            except Exception as e:
                logger.error(f"Error running orchestrator: {e}")
                st.error(f"Error running orchestrator: {e}")
 
    # === Always render ML parameter UI from the last result (persists across reruns) ===
    active_result = st.session_state.get("orc_last_result")
    if active_result:
        for idx, sub_result in enumerate(active_result.get("results", [])):
            if sub_result.get("agent") != "ml" or sub_result.get("error"):
                continue
 
            model_key = sub_result.get("model")
            ui = (sub_result.get("ui") or {})
            label = ui.get("label", model_key)
            defaults = ui.get("params", {}) or {}
 
            st.subheader(f"🧪 {label} — Parameters & Training")
 
            # Tiny hydrator: if user uploaded a CSV already (via key="ml_csv"), load it now
            if "ml_df" not in st.session_state:
                upl = st.session_state.get("ml_csv")
                if upl is not None:
                    df_tmp = load_csv(upl)
                    missing = check_columns(df_tmp, REQUIRED_COLUMNS)
                    if not missing:
                        st.session_state["ml_df"] = df_tmp
 
            if "ml_df" not in st.session_state:
                st.warning("No CSV loaded. Go to the **ML Data** tab and upload your transactions CSV first.")
                continue
            df_ml = st.session_state["ml_df"]
 
            # ---------- filters parsed from the orchestrator ----------
            filters_raw = sub_result.get("filters", {}) or {}
            filters = canonicalize_filters(df_ml, filters_raw)
            missing_ctx = sub_result.get("missing_context", []) or []
            if missing_ctx and sub_result.get("model") in ("volume_discount_basic", "volume_discount_enriched"):
                st.warning("Missing in question: " + ", ".join(missing_ctx) + ". Use the pickers below.")
 
            # ---- Price Anomaly Detection ----
            if model_key == "price_anomaly":
                c1, c2 = st.columns(2)
                with c1:
                    time_budget_s = st.number_input(
                        "AutoML time budget (seconds)", min_value=10, max_value=600,
                        value=int(defaults.get("time_budget_s", 60)), step=5, key=f"ml_t_persist_{idx}"
                    )
                with c2:
                    percentile = st.slider(
                        "Anomaly threshold (percentile of |residual|)", 80, 99,
                        int(defaults.get("percentile", 95)), key=f"ml_p_persist_{idx}"
                    )
 
                # Model source selector + save toggle
                src_col, save_col = st.columns([0.6, 0.4])
                with src_col:
                    model_source = st.radio(
                        "Model source",
                        ["Use existing trained model", "Train on uploaded CSV"],
                        index=0, key=f"pa_src_{idx}", horizontal=True
                    )
                with save_col:
                    save_chk = st.checkbox("Save/overwrite trained model", value=False, key=f"pa_save_{idx}")
 
                # PREVIEW & TOGGLE FILTERS
                will_filter = any([filters.get("vendors"), filters.get("skus"),
                                filters.get("regions"), filters.get("seasons"),
                                filters.get("carriers"), filters.get("currencies")])
                if will_filter:
                    st.caption(
                        f"Using filters from question → "
                        f"Vendors: {filters['vendors'] or 'ALL'}, "
                        f"SKUs: {filters['skus'] or 'ALL'}, "
                        f"Region: {filters['regions'] or 'ALL'}, "
                        f"Season: {filters['seasons'] or 'ALL'}, "
                        f"Carrier: {filters['carriers'] or 'ALL'}, "
                        f"Currency: {filters['currencies'] or 'ALL'}"
                    )
                use_question_filters = st.checkbox(
                    "Apply filters extracted from the question",
                    value=will_filter, key=f"pa_useflt_{idx}"
                )
 
                if st.button("Run Price Anomaly Detection", key=f"ml_run_persist_{idx}"):
 
                    if get_price_model is not None:
                        use_pretrained = (model_source == "Use existing trained model")
                        bundle = get_price_model(
                            use_pretrained=use_pretrained, df=df_ml,
                            time_budget_s=int(time_budget_s), save_new=save_chk
                        )
                    else:
                        bundle = fit_price_model(df_ml, time_budget_s=int(time_budget_s))
 
                    if bundle is None:
                        st.error("No saved model found and no data to train. Upload a CSV or switch to 'Train on uploaded CSV'.")
                    else:
                        # (2) TRUE MULTI-SKU FILTERING
                        if use_question_filters and filters["skus"]:
                            skus_to_run = filters["skus"]
                        else:
                            skus_to_run = [None]  # single pass (no SKU filter)
 
                        outs = []
                        df_used_for_scoring = []
                        for sku_single in skus_to_run:
                            df_run = apply_filters(
                                df_ml.copy(),
                                sku=sku_single,
                                vendors=filters["vendors"] if use_question_filters else None,
                                regions=filters["regions"] if use_question_filters else None,
                                seasons=filters["seasons"] if use_question_filters else None,
                                carriers=filters["carriers"] if use_question_filters else None,
                                currencies=filters["currencies"] if use_question_filters else None,
                            )
                            if df_run.empty:
                                continue
                            out, thr = detect_anomalies(df_run, bundle, percentile=int(percentile))
                            outs.append(out)
                            df_used_for_scoring.append(df_run)
 
                        if not outs:
                            st.error("No rows after applying filters."); st.stop()
 
                        out = pd.concat(outs, ignore_index=True)
                        df_used = pd.concat(df_used_for_scoring, ignore_index=True)
 
                        mae, rmse, r2 = score_price_model(bundle, df_used)
                        st.write(f"**MAE:** {mae:.4f} | **RMSE:** {rmse:.4f} | **R²:** {r2:.4f}")
 
                        st.subheader("Flagged Anomalies")
                        thr = out["Residual_Abs"].quantile(int(percentile)/100.0)  # display-only
                        st.write(f"Anomaly threshold |residual| (display): **{thr:.4f}**")
                        st.dataframe(out[out["Anomaly_Flag"] == 1].reset_index(drop=True))
 
                        fig_residuals_scatter(out, threshold=thr)
 
                        # (5) WHY? Feature importance
                        show_price_importance(bundle, df_used)
 
            # ---- SKU Segmentation ----
            elif model_key == "sku_segmentation":
                use_filters_for_cluster = st.checkbox(
                    "Use question filters for clustering (context-specific)",
                    value=any([filters["regions"], filters["seasons"], filters["carriers"], filters["currencies"], filters["skus"], filters["vendors"]]),
                    key=f"vendor_cluster_ctx_{idx}"
                )
 
                # optionally slice before building features
                df_for_feats = apply_filters_v2(
                    df_ml,
                    skus=filters["skus"] if use_filters_for_cluster else None,
                    vendors=filters["vendors"] if use_filters_for_cluster else None,
                    regions=filters["regions"] if use_filters_for_cluster else None,
                    seasons=filters["seasons"] if use_filters_for_cluster else None,
                    carriers=filters["carriers"] if use_filters_for_cluster else None,
                    currencies=filters["currencies"] if use_filters_for_cluster else None,
                )
                if use_filters_for_cluster and df_for_feats.empty:
                    st.warning("No rows for the chosen context; falling back to global clustering.")
                    df_for_feats = df_ml
 
                k = st.number_input("Number of clusters (k)", min_value=2, max_value=12,
                                    value=int(defaults.get("n_clusters", 3)), key=f"ml_k_persist_{idx}")
                feats_df = sku_features(df_for_feats)
                default_cols = [c for c in feats_df.columns if c != "SKU_ID"]
                feature_cols = st.multiselect("Features to use", default_cols, default=default_cols, key=f"ml_feats_persist_{idx}")
                if st.button("Run SKU Segmentation", key=f"ml_run_persist_{idx}"):
                    clustered = cluster_table_train(feats_df[["SKU_ID"] + feature_cols], n_clusters=int(k), feature_cols=feature_cols)
                    st.subheader("Clustered SKUs")
                    st.dataframe(clustered)
                    fig = fig_sku_clusters(clustered.rename(columns={"Total_Quantity": "Total_Quantity"}))
                    if fig is not None:
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    # show asked SKUs
                    if filters.get("skus"):
                        asked = clustered[clustered["SKU_ID"].astype(str).isin(filters["skus"])]
                        if not asked.empty:
                            st.subheader("Requested SKUs and their clusters")
                            st.dataframe(asked[["SKU_ID", "Cluster"]])
 
            # ---- Vendor Segmentation ----
            elif model_key == "vendor_segmentation":
                use_filters_for_cluster = st.checkbox(
                    "Use question filters for clustering (context-specific)",
                    value=any([filters["regions"], filters["seasons"], filters["carriers"], filters["currencies"], filters["skus"]]),
                    key=f"vendor_cluster_ctx_{idx}"
                )
                df_for_feats = apply_filters_v2(
                    df_ml,
                    skus=filters["skus"] if use_filters_for_cluster else None,
                    vendors=filters["vendors"] if use_filters_for_cluster else None,
                    regions=filters["regions"] if use_filters_for_cluster else None,
                    seasons=filters["seasons"] if use_filters_for_cluster else None,
                    carriers=filters["carriers"] if use_filters_for_cluster else None,
                    currencies=filters["currencies"] if use_filters_for_cluster else None,
                )
                if use_filters_for_cluster and df_for_feats.empty:
                    st.warning("No rows for the chosen context; falling back to global clustering.")
                    df_for_feats = df_ml
 
                k = st.number_input("Number of clusters (k)", min_value=2, max_value=12,
                                    value=int(defaults.get("n_clusters", 3)), key=f"ml_k_persist_{idx}")
                feats_df = vendor_features(df_for_feats)
                default_cols = [c for c in feats_df.columns if c != "Vendor_ID"]
                feature_cols = st.multiselect("Features to use", default_cols, default=default_cols, key=f"ml_feats_persist_{idx}")
                if st.button("Run Vendor Segmentation", key=f"ml_run_persist_{idx}"):
                    clustered = cluster_table_train(feats_df[["Vendor_ID"] + feature_cols], n_clusters=int(k), feature_cols=feature_cols)
                    st.subheader("Clustered Vendors")
                    st.dataframe(clustered)
                    fig = fig_vendor_clusters(clustered)
                    if fig is not None:
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    if filters.get("vendors"):
                        asked = clustered[clustered["Vendor_ID"].astype(str).isin(filters["vendors"])]
                        if not asked.empty:
                            st.subheader("Requested Vendors and their clusters")
                            st.dataframe(asked[["Vendor_ID", "Cluster"]])
 
            # ---- Volume-based Discount (basic/enriched) ----
            elif model_key in ("volume_discount_basic", "volume_discount_enriched"):
                # REQUIRED picks (pre-filled)
                sku_values = sorted(df_ml["SKU_ID"].astype(str).unique().tolist())
                sku_default = filters["skus"][0] if filters["skus"] else None
                sku_index = sku_values.index(sku_default) if sku_default in sku_values else 0
                sku_pick = st.selectbox("SKU (required)", sku_values, index=sku_index, key=f"ml_sku_persist_{idx}")
 
                vendors_avail = sorted(df_ml[df_ml["SKU_ID"].astype(str) == sku_pick]["Vendor_ID"].astype(str).unique().tolist())
                vendor_defaults = [v for v in (filters.get("vendors") or []) if v in vendors_avail] or vendors_avail
                vendors_pick = st.multiselect("Vendors (required – one or more)", vendors_avail, default=vendor_defaults, key=f"ml_vendors_persist_{idx}")
 
                currencies_avail = sorted(df_ml["Currency"].astype(str).unique().tolist())
                curr_defaults = [c for c in (filters.get("currencies") or []) if c in currencies_avail]
                currencies_pick = st.multiselect("Currency (required – choose exactly one)", currencies_avail, default=curr_defaults[:1] if curr_defaults else [], key=f"ml_currs_persist_{idx}")
 
                # Always render context pickers (you enforce exactly-one)
                regions_all = sorted(df_ml["Region"].astype(str).unique().tolist())
                seasons_all = sorted(df_ml["Season"].astype(str).unique().tolist())
                carriers_all = sorted(df_ml["Carrier_Type"].astype(str).unique().tolist())
 
                regions_pick = st.multiselect("Regions", regions_all, default=[r for r in (filters.get("regions") or []) if r in regions_all], key=f"ml_regions_persist_{idx}")
                seasons_pick = st.multiselect("Seasons", seasons_all, default=[s for s in (filters.get("seasons") or []) if s in seasons_all], key=f"ml_seasons_persist_{idx}")
                carriers_pick = st.multiselect("Carrier Type", carriers_all, default=[c for c in (filters.get("carriers") or []) if c in carriers_all], key=f"ml_carriers_persist_{idx}")
 
                # --- guards: MANDATE context ---
                if not sku_pick or not vendors_pick:
                    st.warning("Please choose a SKU and at least one vendor."); st.stop()
                if len(currencies_pick) != 1:
                    st.warning("Please choose exactly one currency."); st.stop()
                need = []
                if len(regions_pick)  != 1: need.append("region")
                if len(seasons_pick)  != 1: need.append("season")
                if len(carriers_pick) != 1: need.append("carrier")
                if need:
                    st.warning("Please pick exactly one " + ", ".join(need) + "."); st.stop()
 
                # Base filtered slice
                base = apply_filters_v2(
                    df_ml,
                    skus=[sku_pick],
                    vendors=vendors_pick, regions=regions_pick, seasons=seasons_pick,
                    carriers=carriers_pick, currencies=currencies_pick
                )
                if base.empty:
                    st.warning("No rows after applying filters. Adjust your selections.")
                    st.stop()
 
                # --- NEW: get target quantity from orchestrator (or parse), and infer (qmin, qmax, step) from data
                # try orchestrator-provided target first
                target_from_orc = sub_result.get("target_qty")
                # fallback parse from the combined original+task question (stored in sub_result['question'])
                target_from_text = extract_target_qty_from_question(sub_result.get("question", "")) if not target_from_orc else target_from_orc
                qmin_v, qmax_v, qstep_v, target_v, qty_grid = infer_qty_defaults(base, target_from_text)
 
                # also honor explicit qty_prefs from orchestrator if present and numeric
                qp = sub_result.get("qty_prefs") or {}
                try:
                    if isinstance(qp.get("min"), (int, float)) and isinstance(qp.get("max"), (int, float)) and qp["max"] > qp["min"]:
                        qmin_v, qmax_v = int(qp["min"]), int(qp["max"])
                        qstep_v = int(qp.get("step") or qstep_v)
                        qty_grid = list(range(qmin_v, qmax_v + 1, max(1, qstep_v)))
                except Exception:
                    pass
 
                # Show the inferred quantities (no manual inputs)
                st.caption(f"Quantity window inferred from data: **Q∈[{qmin_v}, {qmax_v}]**, step **{qstep_v}**; comparing at **Q={target_v}**.")
 
                # --- one-time auto-run if all required picks are present ---
                auto_key = f"_vol_auto_ran_{idx}"
                want_run = st.button("Train, compare & plot", key=f"ml_run_persist_{idx}") or (not st.session_state.get(auto_key, False))
                if want_run:
                    rows = []
                    curves = {}
                    min_rows = 5  # thin-slice threshold
 
                    for v in vendors_pick:
                        # vendor-specific slice (respect quantity window)
                        df_v = base[base["Vendor_ID"].astype(str) == str(v)]
                        df_v_rng = df_v[(df_v["Quantity"] >= qmin_v) & (df_v["Quantity"] <= qmax_v)]
 
                        if df_v_rng.shape[0] < min_rows:
                            # thin slice: observed points only
                            if df_v_rng.empty:
                                st.warning(f"Skipping vendor {v}: no rows in Q∈[{qmin_v},{qmax_v}].")
                                continue
                            pts = df_v_rng.copy()
                            try:
                                grid_set = set(int(q) for q in qty_grid)
                                pts2 = pts[pts["Quantity"].astype(int).isin(grid_set)]
                                pts = pts2 if not pts2.empty else df_v_rng.copy()
                            except Exception:
                                pts = df_v_rng.copy()
                            pts = (
                                pts.groupby("Quantity", as_index=False)["Unit_Price_USD"]
                                .mean().rename(columns={"Unit_Price_USD": "Predicted_Unit_Price"})
                                .sort_values("Quantity")
                            )
                            curves[v] = {"curve": pts}
                            rows.append({"Vendor_ID": v, "Predicted_Unit_Price@Q": float('nan')})
                            st.info(f"Vendor {v}: plotted observed points only (insufficient rows: {df_v_rng.shape[0]}).")
                            continue
 
                        # sufficient data: train per vendor
                        if model_key == "volume_discount_enriched":
                            automl, pre, feats, cat_feats = discount_curve_training_enriched(df_v_rng)
                            ctx = {
                                "Region": regions_pick[0], "Season": seasons_pick[0],
                                "Carrier_Type": carriers_pick[0], "Currency": currencies_pick[0],
                            }
                            curve_df = discount_curve_grid(automl, pre, feats, qty_grid, ctx=ctx, cat_feats=cat_feats)
                            entry = {"curve": curve_df, "pre": pre, "feats": feats, "automl": automl, "cat_feats": cat_feats}
                            price = predict_for_vendor_enriched(entry, target_v, ctx)
                        else:
                            automl, pre, feats = discount_curve_training(df_v_rng)
                            curve_df = discount_curve_grid(automl, pre, feats, qty_grid)
                            entry = {"curve": curve_df, "pre": pre, "feats": feats, "automl": automl}
                            price = predict_for_vendor(entry, target_v)
 
                        curves[v] = entry
                        rows.append({"Vendor_ID": v, "Predicted_Unit_Price@Q": price})
 
                    if not curves:
                        st.warning("No vendor had enough data to visualize.")
                        st.stop()
 
                    # --- show winner FIRST ---
                    res_df = pd.DataFrame(rows).dropna(subset=["Predicted_Unit_Price@Q"])
                    winner_text = "No comparable predictions."
                    if not res_df.empty:
                        best_idx = res_df["Predicted_Unit_Price@Q"].idxmin()
                        best_vendor = res_df.loc[best_idx, "Vendor_ID"]
                        best_price = res_df.loc[best_idx, "Predicted_Unit_Price@Q"]
                        winner_text = f"**Best at Q={target_v}**: {best_vendor} at **{best_price:.4f} {currencies_pick[0]}**"
                        st.success(winner_text)
 
                    # then show chart + table
                    title = "Vendor Discount Curves (Enriched)" if model_key == "volume_discount_enriched" else "Vendor Discount Curves (Quantity only)"
                    fig = fig_discount_multi(curves, title=title)
                    if fig is not None:
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
 
                    st.subheader(f"Predicted price at Q={target_v}")
                    st.dataframe(res_df.sort_values("Predicted_Unit_Price@Q") if not res_df.empty else pd.DataFrame(rows))
 
                    st.session_state[auto_key] = True

# TAB 6: Decision Orchestrator
# TAB 6: Decision Orchestrator
with tabs[5]:
    logger.info("Entered Decision Support tab.")
    st.header("Ask a Question")

    # --- Sidebar-ish controls for Decision sessions (mirror of Orchestrator, but separate keys) ---
    decision_sessions = list_decision_sessions(user_id)
    if decision_sessions:
        picked_dec = st.selectbox(
            "Open Decision session",
            options=[s["id"] for s in decision_sessions],
            format_func=lambda sid: next((s["title"] for s in decision_sessions if s["id"] == sid), sid),
            key="decision_session_picker",
        )
        if st.session_state.get("decision_session_id") != picked_dec:
            rec = load_decision_session(user_id, picked_dec)
            st.session_state["decision_session_id"] = picked_dec
            st.session_state["decision_messages"] = rec.get("messages", [])
            st.session_state["decision_turns"] = rec.get("turns", [])
            st.session_state["decision_followups"] = rec.get("followups", [])
            st.rerun()

    # Create new
    if st.button("➕ New Decision chat", key="new_decision_chat"):
        sid = new_decision_session(user_id, title="New decision chat")
        st.session_state["decision_session_id"] = sid
        st.session_state["decision_session_picker"] = sid
        st.session_state["decision_messages"] = []
        st.session_state["decision_turns"] = []
        st.session_state["_decision_ephemeral_session_id"] = sid
        st.rerun()

    # Delete current
    cur_dec = st.session_state.get("decision_session_id")
    if cur_dec and st.button("🗑️ Delete this Decision chat", key="delete_decision_chat"):
        if delete_decision_session(user_id, cur_dec):
            st.session_state["decision_session_id"] = None
            st.session_state["decision_messages"] = []
            st.session_state["decision_turns"] = []
            st.session_state.pop("decision_last_result", None)
            st.session_state.pop("_decision_force_llm_next", None)
            try:
                clear_llm_memory(cur_dec)
            except Exception:
                pass
            st.rerun()
        else:
            st.error("Could not delete this Decision chat.")

    # ---- Chat UI ----
    user_q_dec = st.chat_input("Ask your decision question…", key="decision_q")
    if "decision_messages" not in st.session_state:
        st.session_state["decision_messages"] = []
    if "decision_turns" not in st.session_state:
        st.session_state["decision_turns"] = []

    # Render history + per-turn followups, aligned
    turns = st.session_state.get("decision_turns", [])
    turn_idx = 0
    pending_user = None
    for m in st.session_state["decision_messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "user":
                pending_user = m["content"]
            elif m["role"] == "assistant" and pending_user is not None:
                if turn_idx < len(turns):
                    fu = (turns[turn_idx] or {}).get("followups", [])
                    if fu:
                        st.markdown("**Follow-ups:**")
                        st.markdown("\n".join(f"- {q}" for q in fu))
                turn_idx += 1
                pending_user = None

    # ---- Handle new turn ----
    if user_q_dec:
        st.session_state["decision_messages"].append({"role": "user", "content": user_q_dec})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    sid = st.session_state.get("decision_session_id")
                    # Force next turn to pure LLM if prior decision run gathered context
                    if st.session_state.get("_decision_force_llm_next"):
                        result = orchestrator(
                            user_q_dec,
                            chat_history=[(u["content"], a["content"]) for u, a in zip(
                                [m for m in st.session_state["decision_messages"] if m["role"]=="user"][:-1],
                                [m for m in st.session_state["decision_messages"] if m["role"]=="assistant"]
                            )],
                            session_id=sid or st.session_state.get("_decision_ephemeral_session_id") or "decision-default",
                            force_agent="llm",
                        )
                        st.session_state["_decision_force_llm_next"] = False
                    else:
                        result = decision_orchestrator(
                            user_q_dec,
                            chat_history=[(u["content"], a["content"]) for u, a in zip(
                                [m for m in st.session_state["decision_messages"] if m["role"]=="user"][:-1],
                                [m for m in st.session_state["decision_messages"] if m["role"]=="assistant"]
                            )],
                            session_id=sid or st.session_state.get("_decision_ephemeral_session_id") or "decision-default",
                        )

                    final = result.get("final_answer","")
                    sugs_obj = (result or {}).get("suggested_followups") or {}
                    sugs = sugs_obj.get("followups", [])
                    st.markdown(final or "_(no answer)_")

                    # Build & mirror a structured turn
                    import time
                    turn = {
                        "user": user_q_dec,
                        "assistant": final,
                        "followups": sugs,
                        "ts": time.time(),
                    }
                    st.session_state["decision_turns"] = st.session_state.get("decision_turns", []) + [turn]
                    st.session_state["decision_messages"].append({"role": "assistant", "content": final})
                    st.session_state["decision_last_result"] = result

                    # Persist to Decision sessions
                    if not sid:
                        sid = new_decision_session(user_id, title=(user_q_dec[:60]+"…") if len(user_q_dec)>60 else user_q_dec)
                        st.session_state["decision_session_id"] = sid
                        st.session_state["decision_session_picker"] = sid
                    rec = load_decision_session(user_id, sid)
                    title = rec.get("title") or ((user_q_dec[:60]+"…") if len(user_q_dec)>60 else user_q_dec) or "Decision chat"

                    save_decision_session(
                        user_id,
                        sid,
                        st.session_state["decision_messages"],
                        title=title,
                        followups=sugs,
                        turns=[turn],
                    )

                    # After a Decision run that pulled data (sql/rag/web), force next turn to LLM
                    sub_agents = [r.get("agent") for r in (result.get("results", []) or [])]
                    if any(a in ("sql","rag","web") for a in sub_agents):
                        st.session_state["_decision_force_llm_next"] = True

                    # Migrate LLM memory
                    old_ephem = st.session_state.get("_decision_ephemeral_session_id")
                    if old_ephem and old_ephem != sid:
                        try:
                            migrate_llm_memory(old_ephem, sid)
                            st.session_state["_decision_ephemeral_session_id"] = sid
                        except Exception as e:
                            logger.warning(f"Could not migrate Decision LLM memory: {e}")

                except Exception as e:
                    st.error(f"Sorry—decision run failed: {e}")

 
    # === Always render ML parameter UI from the last result (persists across reruns) ===
    active_result = st.session_state.get("decision_last_result")
    if active_result:
        for idx, sub_result in enumerate(active_result.get("results", [])):
            if sub_result.get("agent") != "ml" or sub_result.get("error"):
                continue
 
            model_key = sub_result.get("model")
            ui = (sub_result.get("ui") or {})
            label = ui.get("label", model_key)
            defaults = ui.get("params", {}) or {}
 
            st.subheader(f"🧪 {label} — Parameters & Training")
 
            # Tiny hydrator: if user uploaded a CSV already (via key="ml_csv"), load it now
            if "ml_df" not in st.session_state:
                upl = st.session_state.get("ml_csv")
                if upl is not None:
                    df_tmp = load_csv(upl)
                    missing = check_columns(df_tmp, REQUIRED_COLUMNS)
                    if not missing:
                        st.session_state["ml_df"] = df_tmp
 
            if "ml_df" not in st.session_state:
                st.warning("No CSV loaded. Go to the **ML Data** tab and upload your transactions CSV first.")
                continue
            df_ml = st.session_state["ml_df"]
 
            # ---------- filters parsed from the orchestrator ----------
            filters_raw = sub_result.get("filters", {}) or {}
            filters = canonicalize_filters(df_ml, filters_raw)
            missing_ctx = sub_result.get("missing_context", []) or []
            if missing_ctx and sub_result.get("model") in ("volume_discount_basic", "volume_discount_enriched"):
                st.warning("Missing in question: " + ", ".join(missing_ctx) + ". Use the pickers below.")
 
            # ---- Price Anomaly Detection ----
            if model_key == "price_anomaly":
                c1, c2 = st.columns(2)
                with c1:
                    time_budget_s = st.number_input(
                        "AutoML time budget (seconds)", min_value=10, max_value=600,
                        value=int(defaults.get("time_budget_s", 60)), step=5, key=f"ml_t_persist_{idx}"
                    )
                with c2:
                    percentile = st.slider(
                        "Anomaly threshold (percentile of |residual|)", 80, 99,
                        int(defaults.get("percentile", 95)), key=f"ml_p_persist_{idx}"
                    )
 
                # Model source selector + save toggle
                src_col, save_col = st.columns([0.6, 0.4])
                with src_col:
                    model_source = st.radio(
                        "Model source",
                        ["Use existing trained model", "Train on uploaded CSV"],
                        index=0, key=f"pa_src_{idx}", horizontal=True
                    )
                with save_col:
                    save_chk = st.checkbox("Save/overwrite trained model", value=False, key=f"pa_save_{idx}")
 
                # PREVIEW & TOGGLE FILTERS
                will_filter = any([filters.get("vendors"), filters.get("skus"),
                                filters.get("regions"), filters.get("seasons"),
                                filters.get("carriers"), filters.get("currencies")])
                if will_filter:
                    st.caption(
                        f"Using filters from question → "
                        f"Vendors: {filters['vendors'] or 'ALL'}, "
                        f"SKUs: {filters['skus'] or 'ALL'}, "
                        f"Region: {filters['regions'] or 'ALL'}, "
                        f"Season: {filters['seasons'] or 'ALL'}, "
                        f"Carrier: {filters['carriers'] or 'ALL'}, "
                        f"Currency: {filters['currencies'] or 'ALL'}"
                    )
                use_question_filters = st.checkbox(
                    "Apply filters extracted from the question",
                    value=will_filter, key=f"pa_useflt_{idx}"
                )
 
                if st.button("Run Price Anomaly Detection", key=f"ml_run_persist_{idx}"):
 
                    if get_price_model is not None:
                        use_pretrained = (model_source == "Use existing trained model")
                        bundle = get_price_model(
                            use_pretrained=use_pretrained, df=df_ml,
                            time_budget_s=int(time_budget_s), save_new=save_chk
                        )
                    else:
                        bundle = fit_price_model(df_ml, time_budget_s=int(time_budget_s))
 
                    if bundle is None:
                        st.error("No saved model found and no data to train. Upload a CSV or switch to 'Train on uploaded CSV'.")
                    else:
                        # (2) TRUE MULTI-SKU FILTERING
                        if use_question_filters and filters["skus"]:
                            skus_to_run = filters["skus"]
                        else:
                            skus_to_run = [None]  # single pass (no SKU filter)
 
                        outs = []
                        df_used_for_scoring = []
                        for sku_single in skus_to_run:
                            df_run = apply_filters(
                                df_ml.copy(),
                                sku=sku_single,
                                vendors=filters["vendors"] if use_question_filters else None,
                                regions=filters["regions"] if use_question_filters else None,
                                seasons=filters["seasons"] if use_question_filters else None,
                                carriers=filters["carriers"] if use_question_filters else None,
                                currencies=filters["currencies"] if use_question_filters else None,
                            )
                            if df_run.empty:
                                continue
                            out, thr = detect_anomalies(df_run, bundle, percentile=int(percentile))
                            outs.append(out)
                            df_used_for_scoring.append(df_run)
 
                        if not outs:
                            st.error("No rows after applying filters."); st.stop()
 
                        out = pd.concat(outs, ignore_index=True)
                        df_used = pd.concat(df_used_for_scoring, ignore_index=True)
 
                        mae, rmse, r2 = score_price_model(bundle, df_used)
                        st.write(f"**MAE:** {mae:.4f} | **RMSE:** {rmse:.4f} | **R²:** {r2:.4f}")
 
                        st.subheader("Flagged Anomalies")
                        thr = out["Residual_Abs"].quantile(int(percentile)/100.0)  # display-only
                        st.write(f"Anomaly threshold |residual| (display): **{thr:.4f}**")
                        st.dataframe(out[out["Anomaly_Flag"] == 1].reset_index(drop=True))
 
                        fig_residuals_scatter(out, threshold=thr)
 
                        # (5) WHY? Feature importance
                        show_price_importance(bundle, df_used)
 
            # ---- SKU Segmentation ----
            elif model_key == "sku_segmentation":
                use_filters_for_cluster = st.checkbox(
                    "Use question filters for clustering (context-specific)",
                    value=any([filters["regions"], filters["seasons"], filters["carriers"], filters["currencies"], filters["skus"], filters["vendors"]]),
                    key=f"vendor_cluster_ctx_{idx}"
                )
 
                # optionally slice before building features
                df_for_feats = apply_filters_v2(
                    df_ml,
                    skus=filters["skus"] if use_filters_for_cluster else None,
                    vendors=filters["vendors"] if use_filters_for_cluster else None,
                    regions=filters["regions"] if use_filters_for_cluster else None,
                    seasons=filters["seasons"] if use_filters_for_cluster else None,
                    carriers=filters["carriers"] if use_filters_for_cluster else None,
                    currencies=filters["currencies"] if use_filters_for_cluster else None,
                )
                if use_filters_for_cluster and df_for_feats.empty:
                    st.warning("No rows for the chosen context; falling back to global clustering.")
                    df_for_feats = df_ml
 
                k = st.number_input("Number of clusters (k)", min_value=2, max_value=12,
                                    value=int(defaults.get("n_clusters", 3)), key=f"ml_k_persist_{idx}")
                feats_df = sku_features(df_for_feats)
                default_cols = [c for c in feats_df.columns if c != "SKU_ID"]
                feature_cols = st.multiselect("Features to use", default_cols, default=default_cols, key=f"ml_feats_persist_{idx}")
                if st.button("Run SKU Segmentation", key=f"ml_run_persist_{idx}"):
                    clustered = cluster_table_train(feats_df[["SKU_ID"] + feature_cols], n_clusters=int(k), feature_cols=feature_cols)
                    st.subheader("Clustered SKUs")
                    st.dataframe(clustered)
                    fig = fig_sku_clusters(clustered.rename(columns={"Total_Quantity": "Total_Quantity"}))
                    if fig is not None:
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    # show asked SKUs
                    if filters.get("skus"):
                        asked = clustered[clustered["SKU_ID"].astype(str).isin(filters["skus"])]
                        if not asked.empty:
                            st.subheader("Requested SKUs and their clusters")
                            st.dataframe(asked[["SKU_ID", "Cluster"]])
 
            # ---- Vendor Segmentation ----
            elif model_key == "vendor_segmentation":
                use_filters_for_cluster = st.checkbox(
                    "Use question filters for clustering (context-specific)",
                    value=any([filters["regions"], filters["seasons"], filters["carriers"], filters["currencies"], filters["skus"]]),
                    key=f"vendor_cluster_ctx_{idx}"
                )
                df_for_feats = apply_filters_v2(
                    df_ml,
                    skus=filters["skus"] if use_filters_for_cluster else None,
                    vendors=filters["vendors"] if use_filters_for_cluster else None,
                    regions=filters["regions"] if use_filters_for_cluster else None,
                    seasons=filters["seasons"] if use_filters_for_cluster else None,
                    carriers=filters["carriers"] if use_filters_for_cluster else None,
                    currencies=filters["currencies"] if use_filters_for_cluster else None,
                )
                if use_filters_for_cluster and df_for_feats.empty:
                    st.warning("No rows for the chosen context; falling back to global clustering.")
                    df_for_feats = df_ml
 
                k = st.number_input("Number of clusters (k)", min_value=2, max_value=12,
                                    value=int(defaults.get("n_clusters", 3)), key=f"ml_k_persist_{idx}")
                feats_df = vendor_features(df_for_feats)
                default_cols = [c for c in feats_df.columns if c != "Vendor_ID"]
                feature_cols = st.multiselect("Features to use", default_cols, default=default_cols, key=f"ml_feats_persist_{idx}")
                if st.button("Run Vendor Segmentation", key=f"ml_run_persist_{idx}"):
                    clustered = cluster_table_train(feats_df[["Vendor_ID"] + feature_cols], n_clusters=int(k), feature_cols=feature_cols)
                    st.subheader("Clustered Vendors")
                    st.dataframe(clustered)
                    fig = fig_vendor_clusters(clustered)
                    if fig is not None:
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    if filters.get("vendors"):
                        asked = clustered[clustered["Vendor_ID"].astype(str).isin(filters["vendors"])]
                        if not asked.empty:
                            st.subheader("Requested Vendors and their clusters")
                            st.dataframe(asked[["Vendor_ID", "Cluster"]])
 
            # ---- Volume-based Discount (basic/enriched) ----
            elif model_key in ("volume_discount_basic", "volume_discount_enriched"):
                # REQUIRED picks (pre-filled)
                sku_values = sorted(df_ml["SKU_ID"].astype(str).unique().tolist())
                sku_default = filters["skus"][0] if filters["skus"] else None
                sku_index = sku_values.index(sku_default) if sku_default in sku_values else 0
                sku_pick = st.selectbox("SKU (required)", sku_values, index=sku_index, key=f"ml_sku_persist_{idx}")
 
                vendors_avail = sorted(df_ml[df_ml["SKU_ID"].astype(str) == sku_pick]["Vendor_ID"].astype(str).unique().tolist())
                vendor_defaults = [v for v in (filters.get("vendors") or []) if v in vendors_avail] or vendors_avail
                vendors_pick = st.multiselect("Vendors (required – one or more)", vendors_avail, default=vendor_defaults, key=f"ml_vendors_persist_{idx}")
 
                currencies_avail = sorted(df_ml["Currency"].astype(str).unique().tolist())
                curr_defaults = [c for c in (filters.get("currencies") or []) if c in currencies_avail]
                currencies_pick = st.multiselect("Currency (required – choose exactly one)", currencies_avail, default=curr_defaults[:1] if curr_defaults else [], key=f"ml_currs_persist_{idx}")
 
                # Always render context pickers (you enforce exactly-one)
                regions_all = sorted(df_ml["Region"].astype(str).unique().tolist())
                seasons_all = sorted(df_ml["Season"].astype(str).unique().tolist())
                carriers_all = sorted(df_ml["Carrier_Type"].astype(str).unique().tolist())
 
                regions_pick = st.multiselect("Regions", regions_all, default=[r for r in (filters.get("regions") or []) if r in regions_all], key=f"ml_regions_persist_{idx}")
                seasons_pick = st.multiselect("Seasons", seasons_all, default=[s for s in (filters.get("seasons") or []) if s in seasons_all], key=f"ml_seasons_persist_{idx}")
                carriers_pick = st.multiselect("Carrier Type", carriers_all, default=[c for c in (filters.get("carriers") or []) if c in carriers_all], key=f"ml_carriers_persist_{idx}")
 
                # --- guards: MANDATE context ---
                if not sku_pick or not vendors_pick:
                    st.warning("Please choose a SKU and at least one vendor."); st.stop()
                if len(currencies_pick) != 1:
                    st.warning("Please choose exactly one currency."); st.stop()
                need = []
                if len(regions_pick)  != 1: need.append("region")
                if len(seasons_pick)  != 1: need.append("season")
                if len(carriers_pick) != 1: need.append("carrier")
                if need:
                    st.warning("Please pick exactly one " + ", ".join(need) + "."); st.stop()
 
                # Base filtered slice
                base = apply_filters_v2(
                    df_ml,
                    skus=[sku_pick],
                    vendors=vendors_pick, regions=regions_pick, seasons=seasons_pick,
                    carriers=carriers_pick, currencies=currencies_pick
                )
                if base.empty:
                    st.warning("No rows after applying filters. Adjust your selections.")
                    st.stop()
 
                # --- NEW: get target quantity from orchestrator (or parse), and infer (qmin, qmax, step) from data
                # try orchestrator-provided target first
                target_from_orc = sub_result.get("target_qty")
                # fallback parse from the combined original+task question (stored in sub_result['question'])
                target_from_text = extract_target_qty_from_question(sub_result.get("question", "")) if not target_from_orc else target_from_orc
                qmin_v, qmax_v, qstep_v, target_v, qty_grid = infer_qty_defaults(base, target_from_text)
 
                # also honor explicit qty_prefs from orchestrator if present and numeric
                qp = sub_result.get("qty_prefs") or {}
                try:
                    if isinstance(qp.get("min"), (int, float)) and isinstance(qp.get("max"), (int, float)) and qp["max"] > qp["min"]:
                        qmin_v, qmax_v = int(qp["min"]), int(qp["max"])
                        qstep_v = int(qp.get("step") or qstep_v)
                        qty_grid = list(range(qmin_v, qmax_v + 1, max(1, qstep_v)))
                except Exception:
                    pass
 
                # Show the inferred quantities (no manual inputs)
                st.caption(f"Quantity window inferred from data: **Q∈[{qmin_v}, {qmax_v}]**, step **{qstep_v}**; comparing at **Q={target_v}**.")
 
                # --- one-time auto-run if all required picks are present ---
                auto_key = f"_vol_auto_ran_{idx}"
                want_run = st.button("Train, compare & plot", key=f"ml_run_persist_{idx}") or (not st.session_state.get(auto_key, False))
                if want_run:
                    rows = []
                    curves = {}
                    min_rows = 5  # thin-slice threshold
 
                    for v in vendors_pick:
                        # vendor-specific slice (respect quantity window)
                        df_v = base[base["Vendor_ID"].astype(str) == str(v)]
                        df_v_rng = df_v[(df_v["Quantity"] >= qmin_v) & (df_v["Quantity"] <= qmax_v)]
 
                        if df_v_rng.shape[0] < min_rows:
                            # thin slice: observed points only
                            if df_v_rng.empty:
                                st.warning(f"Skipping vendor {v}: no rows in Q∈[{qmin_v},{qmax_v}].")
                                continue
                            pts = df_v_rng.copy()
                            try:
                                grid_set = set(int(q) for q in qty_grid)
                                pts2 = pts[pts["Quantity"].astype(int).isin(grid_set)]
                                pts = pts2 if not pts2.empty else df_v_rng.copy()
                            except Exception:
                                pts = df_v_rng.copy()
                            pts = (
                                pts.groupby("Quantity", as_index=False)["Unit_Price_USD"]
                                .mean().rename(columns={"Unit_Price_USD": "Predicted_Unit_Price"})
                                .sort_values("Quantity")
                            )
                            curves[v] = {"curve": pts}
                            rows.append({"Vendor_ID": v, "Predicted_Unit_Price@Q": float('nan')})
                            st.info(f"Vendor {v}: plotted observed points only (insufficient rows: {df_v_rng.shape[0]}).")
                            continue
 
                        # sufficient data: train per vendor
                        if model_key == "volume_discount_enriched":
                            automl, pre, feats, cat_feats = discount_curve_training_enriched(df_v_rng)
                            ctx = {
                                "Region": regions_pick[0], "Season": seasons_pick[0],
                                "Carrier_Type": carriers_pick[0], "Currency": currencies_pick[0],
                            }
                            curve_df = discount_curve_grid(automl, pre, feats, qty_grid, ctx=ctx, cat_feats=cat_feats)
                            entry = {"curve": curve_df, "pre": pre, "feats": feats, "automl": automl, "cat_feats": cat_feats}
                            price = predict_for_vendor_enriched(entry, target_v, ctx)
                        else:
                            automl, pre, feats = discount_curve_training(df_v_rng)
                            curve_df = discount_curve_grid(automl, pre, feats, qty_grid)
                            entry = {"curve": curve_df, "pre": pre, "feats": feats, "automl": automl}
                            price = predict_for_vendor(entry, target_v)
 
                        curves[v] = entry
                        rows.append({"Vendor_ID": v, "Predicted_Unit_Price@Q": price})
 
                    if not curves:
                        st.warning("No vendor had enough data to visualize.")
                        st.stop()
 
                    # --- show winner FIRST ---
                    res_df = pd.DataFrame(rows).dropna(subset=["Predicted_Unit_Price@Q"])
                    winner_text = "No comparable predictions."
                    if not res_df.empty:
                        best_idx = res_df["Predicted_Unit_Price@Q"].idxmin()
                        best_vendor = res_df.loc[best_idx, "Vendor_ID"]
                        best_price = res_df.loc[best_idx, "Predicted_Unit_Price@Q"]
                        winner_text = f"**Best at Q={target_v}**: {best_vendor} at **{best_price:.4f} {currencies_pick[0]}**"
                        st.success(winner_text)
 
                    # then show chart + table
                    title = "Vendor Discount Curves (Enriched)" if model_key == "volume_discount_enriched" else "Vendor Discount Curves (Quantity only)"
                    fig = fig_discount_multi(curves, title=title)
                    if fig is not None:
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
 
                    st.subheader(f"Predicted price at Q={target_v}")
                    st.dataframe(res_df.sort_values("Predicted_Unit_Price@Q") if not res_df.empty else pd.DataFrame(rows))
 
                    st.session_state[auto_key] = True

# TAB7: RAG only
with tabs[6]:
    logger.info("Entered RAG Chat tab.")
    # Safety: ensure chain & memory exist even if user never opened Tab 2
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = _get_crc()
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    cur_rag = st.session_state.get("rag_session_id")
    if cur_rag and st.button("Delete this RAG chat"):
        if delete_session(user_id, cur_rag):
            st.session_state["rag_session_id"] = None
            st.session_state["rag_messages"] = []
            st.rerun()
        else:
            st.error("Could not delete current chat.")

    # --- 0) Render history first (so switching sessions shows past turns) ---
    for m in st.session_state.get("rag_messages", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask about your data", key="rag_chat_input")
    if not user_msg:
        # nothing new to process; we've already rendered history
        pass
    else:
        # --- 2) Append and render the user's message ---
        logger.info(f"User RAG chat message: {user_msg}")
        st.session_state.rag_messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # --- 3) Smalltalk shortcut ---
        kind = detect_smalltalk_kind(user_msg)
        if kind:
            logger.info(f"Detected smalltalk kind: {kind}")
            reply = choose_smalltalk(kind, st.session_state.setdefault("_smalltalk_seen", {}))
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.rag_messages.append({"role": "assistant", "content": reply})

            # ---- Auto-save after this turn ----
            sid = st.session_state.get("rag_session_id")
            if not sid:
                title = (user_msg[:60] + "…") if len(user_msg) > 60 else user_msg
                sid = new_session(user_id, title=title or "Chat")
                st.session_state["rag_session_id"] = sid
            jsonable_msgs = jsonable_messages(st.session_state.rag_messages)
            rec = load_session(user_id, sid)
            title = rec.get("title") or ((user_msg[:60] + "…") if len(user_msg) > 60 else user_msg) or "Chat"
            save_session(user_id, sid, jsonable_msgs, title=title)

        else:
            # --- 4) Build chat history pairs (last 4 pairs) for CRC ---
            chat_pairs, pending_user = [], None
            for m in st.session_state.rag_messages[:-1]:  # exclude the just-typed user message's assistant (not there yet)
                if m["role"] == "user":
                    pending_user = m["content"]
                elif m["role"] == "assistant" and pending_user is not None:
                    chat_pairs.append((pending_user, m["content"]))
                    pending_user = None
            chat_pairs = chat_pairs[-4:]

            # --- 5) Run QA chain ---
            req_id = str(uuid.uuid4()); t0 = time.perf_counter()
            logger.info("Running RAG QA chain.")
            try:
                result = st.session_state.rag_chain({"question": user_msg, "chat_history": chat_pairs})
                dur_ms = int((time.perf_counter() - t0) * 1000)
                logger.info(f"RAG QA chain completed in {dur_ms} ms.")

                # normalize results
                sources_raw = result.get("source_documents", []) or []
                answer = result.get("answer") or result.get("result") or ""
                sources = [
                    Document(page_content=src.get("page_content", ""), metadata=src.get("metadata", {}))
                    if isinstance(src, dict) and not hasattr(src, "page_content") else src
                    for src in sources_raw
                ]

                # --- 6) Show assistant answer ---
                with st.chat_message("assistant"):
                    st.markdown(answer)

                # --- 7) Answer-aware filtering + optional ImageAgent source ---
                filtered_sources = filter_sources_by_answer(sources, answer, user_msg)
                final_sources = append_imageagent_source(filtered_sources, user_msg)

                # --- 8) RAG Sources (already filtered) ---
                with st.expander("RAG Sources"):
                    if not final_sources:
                        st.warning("No matching passages found in your indexed data.")
                    else:
                        render_sources(user_msg, final_sources, key_seed=f"live_{_qhash(user_msg)}")

                # --- 9) Relevant PDF (RAG pages + ImageAgent PDF if present) ---
                with st.expander("Relevant PDF"):
                    pdf_bytes = build_relevant_pdf(final_sources, user_msg)
                    if pdf_bytes:
                        unique_key = f"download_combined_pdf_{_qhash(user_msg)}_{uuid.uuid4()}"
                        st.download_button(
                            label="Download Combined PDF",
                            data=pdf_bytes,
                            file_name="combined_sources.pdf",
                            mime="application/pdf",
                            key=unique_key
                        )
                        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                        st.markdown(
                            f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="600px"></iframe>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("No PDFs available to combine.")

                # --- 10) Persist assistant turn in memory ---
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": final_sources,   # keep Document objects in-memory for this run
                    "question": user_msg,
                    "latency_ms": dur_ms,
                })

                # --- 11) Auto-save this chat session (pack to JSON first) ---
                sid = st.session_state.get("rag_session_id")
                if not sid:
                    title = (user_msg[:60] + "…") if len(user_msg) > 60 else user_msg
                    sid = new_session(user_id, title=title or "Chat")
                    st.session_state["rag_session_id"] = sid

                rec = load_session(user_id, sid)
                title = rec.get("title") or ((user_msg[:60] + "…") if len(user_msg) > 60 else user_msg) or "Chat"
                jsonable_msgs = jsonable_messages(st.session_state.rag_messages)
                save_session(user_id, sid, jsonable_msgs, title=title)

            except Exception as e:
                logger.error(f"Error running RAG QA chain: {e}")
                st.error(f"Error running RAG QA chain: {e}")


# TAB 8: ML Data — upload CSV only
with tabs[7]:
    logger.info("Entered ML Data tab.")
    st.header("ML Data — Upload CSV")
    st.caption("Expected columns: " + ", ".join(REQUIRED_COLUMNS))
 
    csv_file_ml = st.file_uploader("Upload transactions CSV", type=["csv"], key="ml_csv")
    if csv_file_ml:
        df_tmp = load_csv(csv_file_ml)
        missing = check_columns(df_tmp, REQUIRED_COLUMNS)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.session_state["ml_df"] = df_tmp
            st.success(f"Loaded {len(df_tmp):,} rows for ML.")
            st.dataframe(df_tmp.head(20))
            if st.button("Clear loaded CSV"):
                st.session_state.pop("ml_df", None)
                st.rerun()