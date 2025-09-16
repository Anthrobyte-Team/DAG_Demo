import os, re, json, toml, tiktoken
from typing import List, Dict, Any
from openai import OpenAI
import logging
from sqlalchemy import text
from langchain_openai import ChatOpenAI as LCChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# SQL Agent
from SQLAgent.chain import write_query, execute_query, generate_answer, generate_chart_code
from SQLAgent.db_connection import engine
from SQLAgent.schema import State

# RAG Agent
from RAGAgentCloud.rag_answer import get_rag_answer
from RAGAgentCloud.sources_utils import filter_sources_by_answer
from sqlalchemy import inspect as inspect

# Web Search Agent
from WebSearchAgent.main import web_search
from WebSearchAgent.llm import summarize_news

# ML Agent
from MLAgent.data_utils import extract_target_qty_from_question as _fallback_extract_q
from MLAgent.prompt_template import ML_FILTER_EXTRACTOR_SYSTEM

# Orchestrator Prompt
from orchestrator_prompt_template import PLANNER_SYSTEM, CONSOLIDATOR_SYSTEM, FOLLOWUP_SYSTEM_PROMPT, COMPACT_SYSTEM, CONDENSE_SYSTEM, DECISION_SYSTEM

# logging
from logging_setup import setup_logging
setup_logging()
logger = logging.getLogger("app.orchestrator")

logger.info("Orchestrator setup module loaded.")

# Load Secrets
secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.streamlit', 'secrets.toml')
try:
    secrets = toml.load(secrets_path)
    logger.info("Secrets loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load secrets: {e}")
    raise

OPENAI_API_KEY = secrets["openai"]["api_key"]
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Global context for the Orchestrator LLM =====
RAG_SUMMARIES = {}
RAG_SUMMARY_BLOCK = ""

def _load_rag_summaries() -> None:
    """
    Load /mnt/data/summary.json into globals:
      - RAG_SUMMARIES: raw dict
      - RAG_SUMMARY_BLOCK: compact bullet list for planner context
    """
    global RAG_SUMMARIES, RAG_SUMMARY_BLOCK
    try:
        # Robust pathing: prefer /mnt/data/summary.json; fallback to file alongside this module
        primary = "/mnt/data/summary.json"
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary.json")
        path = primary if os.path.exists(primary) else fallback

        if not os.path.exists(path):
            logger.warning("RAG summary file not found at %s (and fallback). Planner will proceed without it.", primary)
            RAG_SUMMARIES, RAG_SUMMARY_BLOCK = {}, ""
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            RAG_SUMMARIES = data
            # Build a compact human-readable block (limit items & per-item size to keep prompts lean)
            items = []
            max_items = 12
            max_chars = 600
            for i, (k, v) in enumerate(list(data.items())[:max_items]):
                short = (v or "")
                short = short if len(short) <= max_chars else (short[:max_chars] + "…")
                items.append(f"- {k}: {short}")
            RAG_SUMMARY_BLOCK = "Internal NTPC summaries (RAG pre-digested):\n" + ("\n".join(items) if items else "(none)")
            logger.info("Loaded %d RAG summary items for planner context.", len(RAG_SUMMARIES))
        else:
            logger.warning("summary.json is not a dict; ignoring.")
            RAG_SUMMARIES, RAG_SUMMARY_BLOCK = {}, ""
    except Exception as e:
        logger.error("Failed to load RAG summaries: %s", e)
        RAG_SUMMARIES, RAG_SUMMARY_BLOCK = {}, ""

def _planner_context_block() -> str:
    """
    Compose a single Context block for the planner:
      - Tables/columns from csv_master_metadata
      - Pre-digested internal RAG summaries
    """
    try:
        tables = build_tables_prompt() or ""
    except Exception:
        tables = ""
    parts = []
    if tables.strip():
        parts.append("Tables & columns (csv_master_metadata):\n" + tables.strip())
    if RAG_SUMMARY_BLOCK.strip():
        parts.append(RAG_SUMMARY_BLOCK.strip())
    return "\n\n".join(parts)

# Load once at import
_load_rag_summaries()

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)

def _pairs_to_text(pairs):
    parts = []
    for u, a in pairs or []:
        parts.append(f"USER: {u}\nASSISTANT: {a}")
    return "\n\n".join(parts)

def compact_history(chat_pairs, session_id: str, target_tokens: int = 1200,
                    min_tail_pairs: int = 8, max_tail_pairs: int = 20):
    """
    Returns a compact list of (user, assistant) pairs:
    [("Earlier summary", "<summary>"), ...tail_pairs...]
    Summarizes the head, keeps a dynamic tail under a token budget.
    """
    if not chat_pairs:
        return []

    total_tokens = _estimate_tokens(_pairs_to_text(chat_pairs))
    if total_tokens <= target_tokens:
        return chat_pairs

    # dynamic tail growth
    tail = []
    for n in range(min_tail_pairs, min(len(chat_pairs), max_tail_pairs) + 1):
        tail = chat_pairs[-n:]
        t = _estimate_tokens(_pairs_to_text(tail))
        if t >= target_tokens * 0.6:
            break

    head = chat_pairs[:-len(tail)] if tail else chat_pairs
    head_txt = _pairs_to_text(head)

    try:
        summ = _llm_for_agent.invoke([
            ("system", COMPACT_SYSTEM),
            ("human", head_txt)
        ])
        summary = (summ.content or "").strip()
    except Exception:
        summary = "Earlier conversation summary unavailable."

    compacted = [("Earlier conversation summary:", summary)]
    compacted.extend(tail)
    return compacted

def condense_followup(latest_question: str, chat_history=None) -> str:
    if not chat_history:
        return latest_question
    compact_pairs = compact_history(chat_history, session_id="__condense__", target_tokens=900)
    hist_txt = _pairs_to_text(compact_pairs)
    msgs = [
        ("system", CONDENSE_SYSTEM),
        ("human", f"History (summary+tail):\n{hist_txt}\n\nLatest:\n{latest_question}\n\nReturn standalone:")
    ]
    try:
        out = _llm_for_agent.invoke(msgs)
        return (out.content or "").strip() or latest_question
    except Exception:
        return latest_question

def maybe_summarize_llm_memory(session_id: str, target_tokens: int = 3500, keep_tail_pairs: int = 12):
    """Compress older LLM-agent messages: running summary + tail."""
    hist = _get_session_history(session_id)
    if not hist.messages:
        return
    text = "\n\n".join((m.type.upper() + ": " + (m.content or "")) for m in hist.messages)
    if _estimate_tokens(text) <= target_tokens:
        return
    tail_msgs = hist.messages[-keep_tail_pairs:]
    head_msgs = hist.messages[:-keep_tail_pairs]
    head_txt = "\n\n".join((m.type.upper() + ": " + (m.content or "")) for m in head_msgs)
    try:
        summ = _llm_for_agent.invoke([
            ("system", "Summarize earlier chat. Preserve entities (vendors/SKUs), definitions, metrics, and decisions."),
            ("human", head_txt)
        ])
        summary = f"(Earlier LLM chat summary) {summ.content.strip()}"
    except Exception:
        summary = "(Earlier LLM chat summary unavailable.)"
    hist.messages = []
    hist.add_ai_message(summary)
    for m in tail_msgs:
        hist.add_message(m)


_llm_for_agent = LCChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=OPENAI_API_KEY,
)

_llm_agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a knowledgeable procurement assistant inside an orchestrator. "
     "Maintain context across this chat and build on prior answers. "
     "If key parameters (e.g., definitions, metrics, SKUs, vendors, timeframe) are missing "
     "and would materially change the answer, ask up to 2 focused follow-up questions and STOP. "
     "Otherwise, answer directly. Never invent numbers; when referencing earlier SQL/RAG outputs, "
     "say “as per earlier answer”."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

_llm_agent_chain = _llm_agent_prompt | _llm_for_agent

# Per-session chat history store
_history_store = {}

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    hist = _history_store.get(session_id)
    if hist is None:
        hist = InMemoryChatMessageHistory()
        _history_store[session_id] = hist
    return hist

_llm_agent_with_mem = RunnableWithMessageHistory(
    _llm_agent_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def clear_llm_memory(session_id: str) -> None:
    """Clear stored LLM chat memory for a given orchestrator session."""
    _history_store.pop(session_id, None)

def migrate_llm_memory(old_session_id: str, new_session_id: str) -> None:
    """Move chat memory from a temporary session id to the final persisted one."""
    if not old_session_id or old_session_id == new_session_id:
        return
    old = _history_store.get(old_session_id)
    if old is None:
        return
    new = _get_session_history(new_session_id)
    for m in old.messages:
        new.add_message(m)
    _history_store.pop(old_session_id, None)


def build_tables_prompt():
    logger.info("Building tables prompt.")
    try:
        from SQLAgent.db_connection import engine
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name, column_names, description FROM csv_master_metadata
            """))
            rows = result.fetchall()
            descs = []
            for r in rows:
                columns = r[1]
                if isinstance(columns, str):
                    import json
                    try:
                        columns = json.loads(columns)
                    except Exception:
                        logger.warning("Failed to parse columns as JSON in build_tables_prompt.")
                        columns = []
                descs.append(
                    f"Table: {r[0]}\nDescription: {r[2] or ''}\nColumns: {', '.join(columns)}"
                )
            logger.info("Tables prompt built successfully.")
            return "\n".join(descs)
    except Exception as e:
        logger.error(f"Error building tables prompt: {e}")
        return ""

def _extract_tables_from_sql(sql: str) -> list[str]:
    """
    Naive, robust-enough table extractor from a single SQL string.
    Grabs identifiers after FROM and JOIN, strips quotes/aliases.
    """
    if not sql:
        return []
    # Find tokens after FROM/ JOIN
    patt = re.compile(r'(?i)\b(?:from|join)\s+([`"\[\]\w\.]+)')
    raw = patt.findall(sql)
    out = []
    for t in raw:
        t = (t or "").strip()
        if not t or t.startswith("("):  # skip subqueries
            continue
        # strip quoting and schema alias like schema.table or "table" AS t
        t = t.strip('`"[]')
        t = t.split(".")[-1]            # drop schema if present
        t = t.split()[0]                # drop alias if present
        t = t.rstrip(",")
        if t and t not in out:
            out.append(t)
    return out

def _schema_prompt_for_tables(tables: list[str]) -> str:
    """
    Fetch column names for concrete tables using SQLAlchemy inspector.
    If a table isn't found, skip it (no PRAGMA fallback).
    """
    if not tables:
        return ""
    try:
        insp = inspect(engine)
        chunks = []
        for tbl in tables:
            try:
                cols = [c["name"] for c in insp.get_columns(tbl)]
            except Exception:
                # Table not found or inaccessible — skip it
                continue
            col_txt = ", ".join(cols) if cols else "(no columns)"
            chunks.append(f"Table: {tbl}\nColumns: {col_txt}")
        return "\n".join(chunks)
    except Exception:
        return ""

def suggest_followups(
    original_question: str,
    final_answer: str,
    results: list,
    n_questions: int = 3
) -> dict:
    import json as _json

    # Collect context from sub-results
    rag_source_lines, sql_bits, web_bits = [], [], []

    # Helper: build a compact ID like file.pdf#p3 from a LangChain Document
    def _source_id_from_doc(doc):
        meta = getattr(doc, "metadata", {}) or {}
        name = meta.get("file_name") or meta.get("display_name") or meta.get("source") or ""
        page = meta.get("page")
        try:
            if isinstance(page, int) and page >= 0:
                return f"{name}#p{page + 1}"
        except Exception:
            pass
        return name or "source"

    # Extract tables from SQL, gather snippets, and collect RAG sources
    used_tables = []
    for r in results or []:
        agent = r.get("agent")
        if agent == "sql":
            ans = (r.get("answer") or "").strip()
            if ans:
                sql_bits.append(ans[:350])
            sql_text = r.get("sql") or r.get("query") or ""
            for t in _extract_tables_from_sql(sql_text):
                if t not in used_tables:
                    used_tables.append(t)

        elif agent == "rag":
            # Prefer filtered sources (by final answer + original question)
            try:
                srcs = r.get("sources") or []
                filtered = filter_sources_by_answer(srcs, final_answer, original_question) or []
                for d in filtered[:6]:
                    sid = _source_id_from_doc(d)
                    snip = (d.page_content or "")[:180].replace("\n", " ")
                    rag_source_lines.append(f"- {sid}: {snip}{'…' if len(d.page_content or '') > 180 else ''}")
            except Exception:
                # Fallback to prebuilt citations if filtering fails
                for c in (r.get("citations") or [])[:6]:
                    sid = (c.get("source_id") or "").strip()
                    sn = (c.get("snippet") or "")[:180].replace("\n", " ")
                    if sid or sn:
                        rag_source_lines.append(f"- {sid}: {sn}{'…' if len(c.get('snippet','')) > 180 else ''}")

        elif agent == "web":
            summ = (r.get("answer") or "")[:300]
            if summ:
                web_bits.append(f"{summ}")

    # Build schema prompt from actually used tables (will be empty if none or not found)
    used_schema_prompt = _schema_prompt_for_tables(used_tables)
    schema_block = f"\nRelevant tables & columns (used by prior SQL):\n{used_schema_prompt}\n" if used_schema_prompt else ""

    # Compose LLM context
    sections = [
        f"Original:\n{original_question}",
        f"Final Answer:\n{final_answer}",
    ]
    if sql_bits:
        sections.append("SQL nuggets:\n- " + "\n- ".join(sql_bits[:3]))
    if rag_source_lines:
        sections.append("Filtered RAG sources:\n" + "\n".join(rag_source_lines))
    if web_bits:
        sections.append("Web context:\n- " + "\n- ".join(web_bits[:2]))
    if schema_block:
        sections.append(schema_block.strip())

    context = "\n\n".join(sections) + f"\n\nSuggest {n_questions} next questions."

    logger.info(f"Suggesting follow-ups with used tables: {used_tables} | rag_sources={len(rag_source_lines)}")

    # Ask the LLM for STRICT JSON
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ]
        )
        data = _json.loads(resp.choices[0].message.content or "{}")
        items = data.get("followups") or []

        # De-dup + trim
        seen, out = set(), []
        for q in items:
            qt = (q or "").strip()
            if qt and qt.lower() not in seen:
                out.append(qt)
                seen.add(qt.lower())

        # Binary vs multi heuristics (keep if you already have is_binary_followup)
        bin_count = sum(1 for q in out if 'yes' == q.lower() or 'no' == q.lower())  # lightweight check
        mode = "binary" if (out and bin_count >= max(1, len(out) // 2)) else "multi"

        if mode == "binary":
            out = out[:1]
        else:
            out = out[:n_questions]

        return {"followups": out, "followup_mode": mode}

    except Exception as e:
        logger.error(f"Error suggesting follow-ups: {e}")
        return {"followups": [], "followup_mode": "multi"}

# --- DAG Helpers ---
def _normalize_plan(plan: dict) -> dict:
    """Ensure every sub-question has a 'depends_on' list and valid agent."""
    subs = plan.get("sub_questions", [])
    for i, sq in enumerate(subs):
        sq.setdefault("depends_on", [])
        # basic validation
        if sq.get("agent") not in ("sql", "rag", "web", "llm"):
            raise ValueError(f"Invalid agent at idx {i}: {sq.get('agent')}")
        if not isinstance(sq["depends_on"], list) or any(
            (not isinstance(j, int) or j < 0 or j >= len(subs) or j == i)
            for j in sq["depends_on"]
        ):
            raise ValueError(f"Bad depends_on at idx {i}: {sq['depends_on']}")
    return plan

def create_dag(plan: dict) -> list[int]:
    """
    Return a topological order (list of node indices) from a plan with depends_on.
    Raises ValueError on cycles.
    """
    plan = _normalize_plan(plan)
    n = len(plan["sub_questions"])
    indeg = [0]*n
    children = [[] for _ in range(n)]
    for i, sq in enumerate(plan["sub_questions"]):
        for j in sq["depends_on"]:
            children[j].append(i)
            indeg[i] += 1
    # Kahn's algorithm
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        raise ValueError("Plan contains a cycle (not a DAG).")
    return order

def _summarize_dep(results_by_idx, deps):
    parts = []
    for j in deps:
        r = results_by_idx.get(j, {})
        a = r.get("agent", "?").upper()
        ans = r.get("answer") or r.get("rows") or r.get("sql") or ""
        parts.append(f"[{a}] {ans}")
    return "\n".join(parts)[:4000]

def execute_dag(plan: dict, *, session_id: str, chat_history):
    """
    Execute sub-questions in topological order; return a list of sub-results
    ordered by the topo order (not original index).
    """
    order = create_dag(plan)
    subs = plan["sub_questions"]
    results_by_idx = {}

    # helper: call the right agent (reuse your existing agent functions)
    def _call(agent: str, q: str):
        if agent == "sql":
            return sql_agent(q)
        elif agent == "rag":
            return rag_agent(q, chat_history=chat_history)
        elif agent == "web":
            return web_agent(q)
        elif agent == "llm":
            return llm_agent(q, session_id=session_id, chat_history=chat_history)
        else:
            return {"agent": agent, "question": q, "error": "Unknown agent"}

    for i in order:
        agent = subs[i]["agent"]
        q = subs[i]["question"]
        deps = subs[i].get("depends_on", [])

        # --- Optional context injection (Step 3) ---
        if deps:
            ctx = _summarize_dep(results_by_idx, deps)
            q = f"Context for this task:\n{ctx}\n\nTask:\n{q}"
        # ------------------------------------------

        results_by_idx[i] = _call(agent, q)


    # Return results in topo order (so downstream UI shows the true execution order)
    return [results_by_idx[i] for i in order]

# SQL Agent
def sql_agent(question: str) -> Dict[str, Any]:
    logger.info(f"SQL agent called with question: {question}")
    try:
        tables_prompt = build_tables_prompt()
        logger.info("Tables prompt built for SQL agent.")
        state: State = {
            "question": question,
            "tables_prompt": tables_prompt,
        }

        # Step 1: Generate SQL query
        query_out = write_query(state)
        logger.info(f"SQL query generated: {query_out.get('query')}")
        state.update(query_out)

        # Step 2: Execute SQL query
        result_out = execute_query(state)
        logger.info("SQL query executed.")
        state.update(result_out)

        # Step 3: Generate natural language answer
        answer_out = generate_answer(state)
        logger.info("SQL answer generated.")
        state.update(answer_out)

        # Step 4: Optionally generate chart code
        chart_out = generate_chart_code(state)
        logger.info("Chart code generated.")
        state.update(chart_out)

        logger.info("SQL agent completed successfully.")
        return {
            "agent": "sql",
            "question": question,
            "sql": state.get("query"),
            "rows": state.get("result"),
            "answer": state.get("answer"),
            "chart_code": state.get("chart_code"),
        }
    except Exception as e:
        logger.error(f"SQL agent error: {e}")
        return {
            "agent": "sql",
            "question": question,
            "error": str(e),
        }

# RAG Agent
def rag_agent(question: str, chat_history=None) -> Dict[str, Any]:
    logger.info(f"RAG agent called with question: {question}")
    try:
        result = get_rag_answer(question, chat_history=chat_history)
        answer = result["answer"]
        sources = result["sources"]

        # Format citations for orchestrator
        citations = []
        for src in sources:
            meta = src.metadata or {}
            source_id = (
                f"{meta.get('file_name','') or meta.get('display_name','') or meta.get('source','')}"
                f"#p{meta.get('page')+1}" if isinstance(meta.get('page'), int) and meta.get('page') >= 0 else ""
            )
            snippet = src.page_content[:160] + ("…" if len(src.page_content) > 160 else "")
            citations.append({"source_id": source_id, "snippet": snippet})

        logger.info("RAG agent completed successfully.")
        return {
            "agent": "rag",
            "question": question,
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "notes": ["live-rag-response"]
        }
    except Exception as e:
        logger.error(f"RAG agent error: {e}")
        return {
            "agent": "rag",
            "question": question,
            "error": str(e),
            "citations": [],
            "sources": [],
            "notes": ["live-rag-response"]
        }

# Web Agent
def web_agent(question: str, num_results: int = 3) -> Dict[str, Any]:
    logger.info(f"Web agent called with question: {question}")
    try:
        results = web_search(question, num_results=num_results)
        logger.info(f"Web search returned {len(results) if results else 0} results.")
        formatted_results = [f"{r['title']}\n{r['link']}\n{r['snippet']}" for r in results]

        # Use summarizer if results exist
        if formatted_results:
            summary = summarize_news(formatted_results)
            logger.info("Web search summary generated.")
        else:
            summary = "No web results found."
            logger.warning("No web results found for web agent.")

        logger.info("Web agent completed successfully.")
        return {
            "agent": "web",
            "question": question,
            "answer": summary,
            "sources": results,
        }
    except Exception as e:
        logger.error(f"Web agent error: {e}")
        return {
            "agent": "web",
            "question": question,
            "error": str(e),
            "sources": [],
        }
    
# New LLM Agent
def llm_agent(question: str, session_id: str, chat_history=None) -> Dict[str, Any]:
    logger.info(f"LLM agent (with memory) called. session_id={session_id}")
    try:
        # Seed the memory once with prior pairs so the LLM sees the recent context
        hist = _get_session_history(session_id)
        if chat_history and len(hist.messages) == 0:
            for u, a in chat_history:
                hist.add_user_message(u)
                hist.add_ai_message(a)

        result_msg = _llm_agent_with_mem.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        answer = getattr(result_msg, "content", str(result_msg))

        # NEW: control LLM memory growth each turn
        maybe_summarize_llm_memory(session_id)
        logger.info("LLM agent completed successfully.")
        return {
            "agent": "llm",
            "question": question,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"LLM agent error: {e}")
        return {
            "agent": "llm",
            "question": question,
            "error": str(e)
        }


# ML model catalog
ML_MODELS = {
    "price_anomaly": {
        "label": "Price Anomaly Detection",
        "params": {"time_budget_s": 60, "percentile": 95}
    },
    "sku_segmentation": {
        "label": "SKU Segmentation (k-means)",
        "params": {"n_clusters": 3}
    },
    "vendor_segmentation": {
        "label": "Vendor Segmentation (k-means)",
        "params": {"n_clusters": 3}
    },
    "volume_discount_basic": {
        "label": "Volume-based Discount (Quantity only)",
        "params": {"qmin": 10, "qmax": 100, "qstep": 10, "target_qty": 50}
    },
    "volume_discount_enriched": {
        "label": "Volume-based Discount (Enriched)",
        "params": {"qmin": 10, "qmax": 100, "qstep": 10, "target_qty": 50}
    },
}

def parse_ml_filters(question: str, client: OpenAI) -> dict:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ML_FILTER_EXTRACTOR_SYSTEM},
                {"role": "user", "content": question}
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logging.getLogger("app").warning(f"ML filter parsing failed: {e}")
        return {}
 
def _ensure_target_qty(parsed: dict, question: str) -> dict:
    if not isinstance(parsed, dict):
        return {}
    tq = parsed.get("target_qty")
    if tq in (None, "", 0):
        guess = _fallback_extract_q(question)
        if guess:
            parsed["target_qty"] = int(guess)
    return parsed
   
def ml_agent(question: str) -> Dict[str, Any]:
    logger = logging.getLogger("app")
    logger.info(f"ML agent called with question: {question}")
    try:
        parsed = parse_ml_filters(question, client) or {}
        parsed = _ensure_target_qty(parsed, question)  # <--- ensure target_qty
 
        task = parsed.get("task")
        task_map = {
            "price_anomaly": "price_anomaly",
            "vendor_segmentation": "vendor_segmentation",
            "sku_segmentation": "sku_segmentation",
            "volume_discount": "volume_discount_enriched",
        }
        model_key = task_map.get(task, "price_anomaly")
        spec = ML_MODELS[model_key]
        return {
            "agent": "ml",
            "question": question,
            "model": model_key,
            "ui": {"label": spec["label"], "params": spec["params"]},
            "filters": parsed.get("filters", {}) or {},
            "target_qty": parsed.get("target_qty"),
            "qty_prefs": parsed.get("qty_prefs") or {},
            "missing_context": parsed.get("missing_context", []) or [],
            "notes": ["render-ml-ui-in-orchestrator-tab"]
        }
    except Exception as e:
        logger.error(f"ML agent error: {e}")
        return {"agent": "ml", "question": question, "error": str(e)}

# LLM helpers
def llm_plan(question: str) -> Dict[str, Any]:
    logger.info(f"LLM planner called with question: {question}")
    try:
        context_block = _planner_context_block()
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f'Question: "{question}"\n\n'
                        f'Context for planning (do not answer yet; just pick agents):\n'
                        f'{context_block or "(no additional context)"}'
                    )
                }
            ]
        )
        plan = json.loads(resp.choices[0].message.content)
        logger.info(f"LLM planner returned plan with {len(plan.get('sub_questions', []))} sub-questions.")
        # Light validation: enforce only sql/rag and single tag per item
        if "sub_questions" not in plan or not isinstance(plan["sub_questions"], list) or len(plan["sub_questions"]) == 0:
            logger.error("Planner returned no sub_questions.")
            raise ValueError("Planner returned no sub_questions.")
        for sq in plan["sub_questions"]:
            if sq.get("agent") not in ("sql", "rag", "web", "llm","ml"):
                logger.error(f'Invalid agent tag: {sq.get("agent")} (must be "sql", "rag", "web", "llm" or "ml").')
                raise ValueError(f'Invalid agent tag: {sq.get("agent")} (must be "sql", "rag", "web", "llm" or "ml").')
        logger.info("LLM planner completed successfully.")
        return plan
    except Exception as e:
        logger.error(f"LLM planner error: {e}")
        raise

def llm_consolidate(original_question: str, sub_answers: List[str]) -> str:
    logger.info("LLM consolidator called.")
    try:
        prompt = f"Original question: {original_question}\n\nSub-answers:\n- " + "\n- ".join(sub_answers) + \
                 "\n\nWrite the final concise answer:"
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": CONSOLIDATOR_SYSTEM},
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("LLM consolidator completed successfully.")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM consolidator error: {e}")
        raise

# Orchestrator powered by LLM
def orchestrator(question: str, chat_history=None, session_id: str = "default", force_agent: str | None = None) -> Dict[str, Any]:
    try:
        # 0) Compact history, 1) condense question
        compact_pairs = compact_history(chat_history or [], session_id=session_id, target_tokens=1200)
        standalone_q = condense_followup(question, chat_history=compact_pairs)

        # >>> NEW: force LLM path when requested
        if force_agent == "llm":
            logger.info(f"Orchestrator is forcing the LLM agent path for question: '{standalone_q}'") # <<< ADDED LOG
            llm_res = llm_agent(standalone_q, session_id=session_id, chat_history=compact_pairs)
            final_answer = llm_res.get("answer", "") if isinstance(llm_res, dict) else str(llm_res)
            followups = suggest_followups(
                original_question=standalone_q,
                final_answer=final_answer,
                results=[{"agent":"llm","question":standalone_q,"answer":final_answer}],
                n_questions=3
            )
            return {
                "original_question": standalone_q,
                "sub_questions": [{"agent":"llm","question":standalone_q}],
                "results": [llm_res],
                "final_answer": final_answer,
                "suggested_followups": followups,
            }


        # 2) Planner on the standalone version
        plan = llm_plan(standalone_q)
        logger.info("Plan generated.")

        # --- DAG PRINTS ---
        print("\n--- PLANNER JSON ---\n" + json.dumps(plan, indent=2))
        print("\n--- DAG NODES ---")
        for idx, sub in enumerate(plan.get("sub_questions", [])):
            print(f"{idx}: agent={sub.get('agent')} | question={sub.get('question')}")
        print("\n--- DAG EDGES (u -> v) ---")
        for idx, sub in enumerate(plan.get("sub_questions", [])):
            for dep in sub.get("depends_on", []):
                print(f"{dep} -> {idx}")

        # 3) Execute plan as a DAG
        results = execute_dag(plan, session_id=session_id, chat_history=compact_pairs)

        print("\n--- EXECUTION RESULTS ---")
        for idx, r in enumerate(results):
            agent = r.get("agent")
            q = r.get("question")
            ans = r.get("answer")
            print(f"Node {idx} [{agent}]: [{ans if ans else 'No result'} for: {q}]")

        # 4) Consolidate (unchanged)
        sub_answer_lines: List[str] = []
        for r in results:
            if r.get("agent") == "sql":
                sub_answer_lines.append(f"SQL Answer: {r.get('answer','')}")
            elif r.get("agent") == "rag":
                ans = r.get("answer", "")
                cites = ", ".join(c.get("source_id","") for c in r.get("citations", [])[:2])
                sub_answer_lines.append(f"RAG Answer: {ans} [{cites}]")
            elif r.get("agent") == "web":
                sub_answer_lines.append(f"Web Answer: {r.get('answer','')}")
            elif r.get("agent") == "llm":
                sub_answer_lines.append(f"LLM Answer: {r.get('answer','')}")
            elif r.get("agent") == "ml":
                label = (r.get("ui", {}) or {}).get("label") or r.get("model") or "ML task"
                sub_answer_lines.append(f"ML Plan: {label} (parameters will be chosen and trained in Orchestrator UI)")
            else:
                sub_answer_lines.append(f"{r.get('agent')}: (no result)")

        final_answer = llm_consolidate(standalone_q, sub_answer_lines)

        # NEW: generate suggested follow-ups
        followups = suggest_followups(
            original_question=standalone_q,
            final_answer=final_answer,
            results=results,
            n_questions=3
        )

        return {
            "original_question": plan["original_question"],
            "sub_questions": plan["sub_questions"],
            "results": results,
            "final_answer": final_answer,
            "suggested_followups": followups,     # <— NEW FIELD
        }

    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        return {
            "original_question": question,
            "sub_questions": [],
            "results": [],
            "final_answer": str(e)
        }

# == NEW ==
def decision_plan(question: str) -> dict:
    """
    Same as llm_plan, but uses DECISION_SYSTEM.
    """
    logger.info(f"[Decision] planner called with question: {question}")
    context_block = _planner_context_block()  # reuse your existing helper
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DECISION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f'Question: "{question}"\n\n'
                    f'Context for planning (do not answer yet; just pick agents):\n'
                    f'{context_block or "(no additional context)"}'
                )
            }
        ]
    )
    return json.loads(resp.choices[0].message.content)

# == NEW ==
def decision_orchestrator(
    question: str,
    chat_history=None,
    session_id: str = "decision-default",
    force_agent: str | None = None,
) -> dict:
    try:
        chat_history = chat_history or []
        # Condense
        compact_pairs = compact_history(chat_history, session_id=session_id, target_tokens=1200)
        standalone_q = condense_followup(question, chat_history=compact_pairs)

        # PLAN with decision prompt
        plan = decision_plan(standalone_q)
        logger.info("[Decision] Plan generated.")

        # EXECUTE via the common DAG runner (ensures llm_agent gets session_id)
        results = execute_dag(plan, session_id=session_id, chat_history=compact_pairs)

        # CONSOLIDATE
        sub_answer_lines = []
        for r in results:
            if r.get("agent") == "sql":
                sub_answer_lines.append(f"SQL Answer: {r.get('answer','')}")
            elif r.get("agent") == "rag":
                ans = r.get("answer","")
                cites = ", ".join(c.get("source_id","") for c in r.get("citations", [])[:2])
                sub_answer_lines.append(f"RAG Answer: {ans} [{cites}]")
            elif r.get("agent") == "web":
                sub_answer_lines.append(f"Web Answer: {r.get('answer','')}")
            elif r.get("agent") == "llm":
                sub_answer_lines.append(f"LLM Answer: {r.get('answer','')}")
            elif r.get("agent") == "ml":
                label = (r.get("ui", {}) or {}).get("label") or r.get("model") or "ML task"
                sub_answer_lines.append(f"ML Plan: {label}")
            else:
                sub_answer_lines.append(f"{r.get('agent')}: (no result)")

        final_answer = llm_consolidate(standalone_q, sub_answer_lines)

        followups = suggest_followups(
            original_question=standalone_q,
            final_answer=final_answer,
            results=results,
            n_questions=3
        )

        return {
            "original_question": plan.get("original_question", standalone_q),
            "sub_questions": plan.get("sub_questions", []),
            "results": results,
            "final_answer": final_answer,
            "suggested_followups": followups,
        }
    except Exception as e:
        logger.error(f"[Decision] Orchestrator error: {e}")
        return {
            "original_question": question,
            "sub_questions": [],
            "results": [],
            "final_answer": str(e)
        }