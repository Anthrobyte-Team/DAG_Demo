import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

SMALLTALK = {
    "greet": [
        "Hey! 👋 What shall we explore from your indexed docs?",
        "Hello! Ready to search your PDFs or links?",
        "Hi there — ask me about anything you’ve indexed.",
        "Hey! I can pull answers with sources from your indexed content."
    ],
    "thanks": [
        "You’re welcome! Want to dig into another topic?",
        "Anytime! I’m here if you need more.",
        "Happy to help — got another question?",
        "Glad it helped! What next?"
    ],
    "bye": [
        "Bye! 👋 Come back anytime.",
        "See you! I’ll keep your session warm.",
        "Catch you later!",
        "Goodbye! Ping me when you’re back."
    ],
    "ack": [
        "Got it. When you’re ready, ask away.",
        "👍 Noted — what next?",
        "Cool — what else can I find?",
        "Okay — want me to search your docs?"
    ],
}

def _time_greeting() -> str:
    hr = datetime.now().hour
    if 5 <= hr < 12: return "Good morning! "
    if 12 <= hr < 17: return "Good afternoon! "
    if 17 <= hr < 22: return "Good evening! "
    return "Hello!"

def detect_smalltalk_kind(text: str):
    if not text: return None
    t = text.strip().lower()
    if "?" in t: return None
    toks = t.split(); joined = " ".join(toks)
    if any(w in joined for w in ["bye","goodbye","see you","ciao","later"]): return "bye"
    if "thank" in joined or "thx" in joined or joined == "ty": return "thanks"
    if toks[:1] and toks[0] in {"hi","hello","hey","yo","namaste","hola"}: return "greet"
    if joined.startswith(("good morning","good afternoon","good evening","good night")):
        return "greet" if "night" not in joined else "bye"
    if len(toks) <= 3: return "ack"
    return None

def choose_smalltalk(kind: str, seen: dict | None = None) -> str:
    seen = seen or {}
    choices = SMALLTALK.get(kind, SMALLTALK["ack"])
    last_idx = seen.get(kind, -1)
    pool = [i for i in range(len(choices)) if i != last_idx] or [0]
    pick = random.choice(pool)
    seen[kind] = pick
    base = choices[pick]
    return _time_greeting() + base if kind == "greet" else base