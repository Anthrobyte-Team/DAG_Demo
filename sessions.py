import os, json, uuid
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.schema import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ----- internal fs helpers -----
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _session_dir_for_user(user_id: str | int) -> str:
    d = os.path.join(SESSIONS_DIR, str(user_id))
    _ensure_dir(d)
    return d

def _orc_session_dir_for_user(user_id: str | int) -> str:
    d = os.path.join(SESSIONS_DIR, str(user_id), "orchestrator")
    _ensure_dir(d)
    return d

# ----- RAG source packing helpers (to make messages JSON-safe) -----
def _pack_sources_for_session(docs: List[Document]) -> List[Dict[str, Any]]:
    packed: List[Dict[str, Any]] = []
    for d in docs or []:
        meta = dict(d.metadata or {})
        packed.append({
            "page_content": d.page_content or "",
            "metadata": {
                "source_type":  meta.get("source_type"),
                "file_url":     meta.get("file_url"),
                "file_id":      meta.get("file_id"),
                "file_name":    meta.get("file_name"),
                "display_name": meta.get("display_name"),
                "pdf_title":    meta.get("pdf_title"),
                "page":         meta.get("page"),
                "title":        meta.get("title"),
                "canonical_url":meta.get("canonical_url"),
                "source_url":   meta.get("source_url"),
                "source":       meta.get("source"),
                "type":         meta.get("type"),
                "description":  meta.get("description"),
            }
        })
    return packed

# def _unpack_sources_from_session(packed: List[Dict[str, Any]]) -> List[Document]:
#     docs: List[Document] = []
#     for it in packed or []:
#         docs.append(Document(
#             page_content=it.get("page_content") or "",
#             metadata=it.get("metadata") or {}
#         ))
#     return docs

def jsonable_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deep-copy messages replacing Document lists with JSON-friendly dicts."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        mm = deepcopy(m)
        if isinstance(mm.get("sources"), list) and mm["sources"]:
            first = mm["sources"][0]
            # only pack if they are Document objects
            if isinstance(first, Document):
                mm["sources"] = _pack_sources_for_session(mm["sources"])  # type: ignore[arg-type]
        out.append(mm)
    return out

# ----- RAG sessions (file-per-chat) -----
def list_sessions(user_id: str | int) -> List[Dict[str, Any]]:
    d = _session_dir_for_user(user_id)
    items: List[Dict[str, Any]] = []
    for fname in os.listdir(d):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(d, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            items.append({
                "id": meta.get("id"),
                "title": meta.get("title", meta.get("id")),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "path": fpath,
            })
        except Exception:
            pass
    items.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return items

def new_session(user_id: str | int, title: str = "New chat") -> str:
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    record = {"id": sid, "title": title, "created_at": now, "updated_at": now, "messages": []}
    fpath = os.path.join(_session_dir_for_user(user_id), f"{sid}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return sid

def load_session(user_id: str | int, session_id: str) -> Dict[str, Any]:
    fpath = os.path.join(_session_dir_for_user(user_id), f"{session_id}.json")
    if not os.path.exists(fpath):
        return {"id": session_id, "messages": [], "title": "Recovered chat"}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_session(user_id: str | int, session_id: str, messages: list, title: Optional[str] = None) -> None:
    fpath = os.path.join(_session_dir_for_user(user_id), f"{session_id}.json")
    now = datetime.utcnow().isoformat()
    record = {"id": session_id, "title": title or "Chat", "created_at": None, "updated_at": now, "messages": messages}
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                old = json.load(f)
            record["created_at"] = old.get("created_at") or now
            record["title"] = title or old.get("title") or "Chat"
        except Exception:
            record["created_at"] = now
    else:
        record["created_at"] = now
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

def delete_session(user_id: str | int, session_id: str) -> bool:
    try:
        fpath = os.path.join(_session_dir_for_user(user_id), f"{session_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
            return True
    except Exception:
        pass
    return False

# ----- Orchestrator sessions (separate subfolder) -----
def list_orc_sessions(user_id: str | int) -> List[Dict[str, Any]]:
    d = _orc_session_dir_for_user(user_id)
    items: List[Dict[str, Any]] = []
    for fname in os.listdir(d):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(d, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            items.append({
                "id": meta.get("id"),
                "title": meta.get("title", meta.get("id")),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "path": fpath,
            })
        except Exception:
            pass
    items.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return items

def new_orc_session(user_id: str | int, title: str = "New orchestrator chat") -> str:
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    record = {
        "id": sid,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "followups": [],   # << add
        "turns": []        # << add
    }
    fpath = os.path.join(_orc_session_dir_for_user(user_id), f"{sid}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return sid


def load_orc_session(user_id: str | int, session_id: str) -> Dict[str, Any]:
    fpath = os.path.join(_orc_session_dir_for_user(user_id), f"{session_id}.json")
    if not os.path.exists(fpath):
        return {"id": session_id, "messages": [], "title": "Recovered orchestrator chat"}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_orc_session(
    user_id: str | int,
    session_id: str,
    messages: list,
    title: Optional[str] = None,
    followups: Optional[list[str]] = None,      # << add
    turns: Optional[list[dict]] = None           # << add
) -> None:
    fpath = os.path.join(_orc_session_dir_for_user(user_id), f"{session_id}.json")
    now = datetime.utcnow().isoformat()
    record = {
        "id": session_id,
        "title": title or "Orchestrator chat",
        "created_at": None,
        "updated_at": now,
        "messages": messages,
    }
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                old = json.load(f)
            record["created_at"] = old.get("created_at") or now
            record["title"] = title or old.get("title") or "Orchestrator chat"
            # carry forward prior structured fields
            record["followups"] = old.get("followups", [])
            record["turns"] = old.get("turns", [])
        except Exception:
            record["created_at"] = now
            record["followups"] = []
            record["turns"] = []
    else:
        record["created_at"] = now
        record["followups"] = []
        record["turns"] = []

    # append or replace structured data
    if followups is not None:
        record["followups"] = list(followups)  # latest suggestions at top-level
    if turns:
        record["turns"].extend(turns)          # append this turn

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def delete_orc_session(user_id: str | int, session_id: str) -> bool:
    try:
        fpath = os.path.join(_orc_session_dir_for_user(user_id), f"{session_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
            return True
    except Exception:
        pass
    return False

# ----- Decision sessions (separate subfolder) -----
def _decision_session_dir_for_user(user_id: str | int) -> str:
    d = os.path.join(_session_dir_for_user(user_id), "decision_sessions")
    os.makedirs(d, exist_ok=True)
    return d

def list_decision_sessions(user_id: str | int) -> List[Dict[str, Any]]:
    d = _decision_session_dir_for_user(user_id)
    items: List[Dict[str, Any]] = []
    for fname in os.listdir(d):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(d, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            items.append({
                "id": meta.get("id"),
                "title": meta.get("title", meta.get("id")),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "path": fpath,
            })
        except Exception:
            pass
    items.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return items

def new_decision_session(user_id: str | int, title: str = "New decision chat") -> str:
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    record = {
        "id": sid,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "followups": [],
        "turns": []
    }
    fpath = os.path.join(_decision_session_dir_for_user(user_id), f"{sid}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return sid

def load_decision_session(user_id: str | int, session_id: str) -> Dict[str, Any]:
    fpath = os.path.join(_decision_session_dir_for_user(user_id), f"{session_id}.json")
    if not os.path.exists(fpath):
        return {"id": session_id, "messages": [], "title": "Recovered decision chat"}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_decision_session(
    user_id: str | int,
    session_id: str,
    messages: list,
    title: Optional[str] = None,
    followups: Optional[list[str]] = None,
    turns: Optional[list[dict]] = None
) -> None:
    fpath = os.path.join(_decision_session_dir_for_user(user_id), f"{session_id}.json")
    now = datetime.utcnow().isoformat()
    record = {
        "id": session_id,
        "title": title or "Decision chat",
        "created_at": None,
        "updated_at": now,
        "messages": messages,
    }
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                old = json.load(f)
            record["created_at"] = old.get("created_at") or now
            record["title"] = title or old.get("title") or "Decision chat"
            record["followups"] = old.get("followups", [])
            record["turns"] = old.get("turns", [])
        except Exception:
            record["created_at"] = now
            record["followups"] = []
            record["turns"] = []
    else:
        record["created_at"] = now
        record["followups"] = []
        record["turns"] = []

    if followups is not None:
        record["followups"] = list(followups)
    if turns:
        record["turns"].extend(turns)

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

def delete_decision_session(user_id: str | int, session_id: str) -> bool:
    try:
        fpath = os.path.join(_decision_session_dir_for_user(user_id), f"{session_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
            return True
    except Exception:
        pass
    return False
