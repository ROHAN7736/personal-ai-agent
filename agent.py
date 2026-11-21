"""
Personal AI Agent: free, local-first, token-lean.

    CLI:       python agent.py        (/help /clear /model <name> /tokens)
    Web UI:    streamlit run app.py
    REST API:  uvicorn api:app --reload
    Telegram:  python telegram_bot.py

Providers (all OpenAI-compatible): ollama (default, local), groq, gemini.
Config via environment variables: PROVIDER, MODEL, OLLAMA_URL, GROQ_API_KEY,
GEMINI_API_KEY, AGENT_DB, AGENT_FILES, AGENT_LOG.
"""
import ast
import datetime
import json
import logging
import operator
import os
import pathlib
import sqlite3
import threading

from openai import OpenAI

# ---------- Config ----------
PROVIDER = os.getenv("PROVIDER", "ollama")
PROVIDERS = {
    "ollama": (os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), "ollama", "qwen2.5:3b"),
    "groq": ("https://api.groq.com/openai/v1", os.getenv("GROQ_API_KEY", ""), "llama-3.1-8b-instant"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        os.getenv("GEMINI_API_KEY", ""),
        "gemini-2.0-flash",
    ),
}
BASE_URL, API_KEY, DEFAULT_MODEL = PROVIDERS[PROVIDER]
MODEL = os.getenv("MODEL") or DEFAULT_MODEL
client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "missing-key", timeout=180)

# ---------- Token-saving knobs ----------
HISTORY_WINDOW = 6        # only the last N messages are sent each turn
HISTORY_STORE_LIMIT = 50  # messages kept per session in the database
MAX_STEPS = 4             # max tool-calling rounds per question
MAX_OUT_TOKENS = 400      # cap on every reply
TOOL_OUTPUT_LIMIT = 1200  # characters of any tool result sent back to the model

SYSTEM = (
    "You are a concise personal assistant. Answer briefly. "
    "Do not use tools for greetings or small talk. "
    "Questions about the user (their preferences, plans, facts they told you): call recall_notes first. "
    "When the user asks you to remember something: call save_note. "
    "Math: call calc. Current facts or news: call web_search."
)

logging.basicConfig(
    filename=os.getenv("AGENT_LOG", "agent.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("agent")

# ---------- Storage (one shared connection, guarded by a lock) ----------
DB_PATH = os.getenv("AGENT_DB", "agent.db")
pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
DB = sqlite3.connect(DB_PATH, check_same_thread=False)
_lock = threading.Lock()
FILES_DIR = pathlib.Path(os.getenv("AGENT_FILES", "files")).resolve()
FILES_DIR.mkdir(parents=True, exist_ok=True)


def _db(sql: str, params: tuple = ()) -> list:
    with _lock:
        rows = DB.execute(sql, params).fetchall()
        DB.commit()
        return rows


_db("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, ts TEXT, text TEXT)")
_db("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, sid TEXT, role TEXT, content TEXT)")


# ---------- Tools ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calc(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval").body))


def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=3)
    return "\n".join(f"{r['title']}: {r['body']} ({r['href']})" for r in results) or "No results."


def save_note(text: str) -> str:
    _db("INSERT INTO notes (ts, text) VALUES (?, ?)", (datetime.datetime.now().isoformat(), text))
    return "Saved."


def recall_notes(keyword: str = "") -> str:
    rows = _db("SELECT ts, text FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT 5", (f"%{keyword}%",))
    return "\n".join(f"{ts[:10]}: {t}" for ts, t in rows) or "No notes."


def read_file(name: str) -> str:
    path = (FILES_DIR / name).resolve()
    if FILES_DIR not in path.parents:
        return "Access denied."
    return path.read_text(errors="ignore")[:TOOL_OUTPUT_LIMIT] if path.exists() else "File not found."


def now() -> str:
    return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")


def _tool(name, desc, props=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in (props or [])},
                "required": required or [],
            },
        },
    }


TOOLS = [
    _tool("web_search", "Search the web for current info", ["query"], ["query"]),
    _tool("calc", "Evaluate a math expression", ["expression"], ["expression"]),
    _tool("save_note", "Save a note or fact about the user", ["text"], ["text"]),
    _tool("recall_notes", "Find saved notes by keyword", ["keyword"]),
    _tool("read_file", "Read a text file from the files/ folder", ["name"], ["name"]),
    _tool("now", "Get current date and time"),
]
REGISTRY = {f.__name__: f for f in (web_search, calc, save_note, recall_notes, read_file, now)}


# ---------- Saved conversation history (per session id) ----------
def load_history(sid: str) -> list:
    rows = _db("SELECT role, content FROM messages WHERE sid=? ORDER BY id DESC LIMIT ?", (sid, HISTORY_WINDOW))
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_turn(sid: str, user: str, assistant: str) -> None:
    _db("INSERT INTO messages (sid, role, content) VALUES (?, 'user', ?)", (sid, user))
    _db("INSERT INTO messages (sid, role, content) VALUES (?, 'assistant', ?)", (sid, assistant))
    _db(
        "DELETE FROM messages WHERE sid=? AND id NOT IN "
        "(SELECT id FROM messages WHERE sid=? ORDER BY id DESC LIMIT ?)",
        (sid, sid, HISTORY_STORE_LIMIT),
    )


def clear_history(sid: str) -> None:
    _db("DELETE FROM messages WHERE sid=?", (sid,))


# ---------- Agent loop ----------
total_tokens = 0


def run(user_msg: str, history: list) -> str:
    """Answer one message. `history` is mutated (user + assistant turns appended)."""
    global total_tokens
    history.append({"role": "user", "content": user_msg})
    recent = history[-HISTORY_WINDOW:]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    msgs = [{"role": "system", "content": SYSTEM}] + recent

    for _ in range(MAX_STEPS):
        r = client.chat.completions.create(
            model=MODEL, messages=msgs, tools=TOOLS,
            max_tokens=MAX_OUT_TOKENS, temperature=0.3,
        )
        if r.usage:
            total_tokens += r.usage.total_tokens
        m = r.choices[0].message
        if not m.tool_calls:
            answer = m.content or ""
            history.append({"role": "assistant", "content": answer})
            return answer

        msgs.append(m.model_dump(exclude_none=True))
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                log.info("tool call: %s %s", tc.function.name, args)
                out = REGISTRY[tc.function.name](**args)
            except Exception as e:  # keep the agent alive on tool errors
                log.warning("tool error in %s: %s", tc.function.name, e)
                out = f"Tool error: {e}"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)[:TOOL_OUTPUT_LIMIT]})

    return "I couldn't finish in the allowed steps. Try rephrasing."


def chat(session_id: str, text: str) -> str:
    """Stateful entry point used by the API and Telegram bot: loads and saves history."""
    history = load_history(session_id)
    answer = run(text, history)
    save_turn(session_id, text, answer)
    return answer


# ---------- CLI ----------
HELP = "Commands: /help  /clear  /model <name>  /tokens  exit"

if __name__ == "__main__":
    print(f"Agent ready | {PROVIDER}:{MODEL}\n{HELP}")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q == "/help":
            print(HELP)
        elif q == "/clear":
            clear_history("cli")
            print("History cleared.")
        elif q == "/tokens":
            print(f"Tokens used this session: {total_tokens}")
        elif q.startswith("/model"):
            parts = q.split(maxsplit=1)
            if len(parts) == 2:
                MODEL = parts[1].strip()
            print(f"Model: {PROVIDER}:{MODEL}")
        else:
            try:
                print("Agent:", chat("cli", q))
                print(f"[tokens used this session: {total_tokens}]")
            except Exception as e:
                log.error("chat failed: %s", e)
                print(f"Error: {e}")
