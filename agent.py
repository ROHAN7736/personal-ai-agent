"""
Personal AI Agent: free, local-first, token-lean.

Setup:
    pip install openai ddgs
    ollama pull qwen2.5:3b          # free, local, zero tokens billed
    python agent.py

Switch to a free cloud API when you need more power:
    PROVIDER=groq   GROQ_API_KEY=...   python agent.py
    PROVIDER=gemini GEMINI_API_KEY=... python agent.py
Override the model with MODEL=<name> (check the provider's current free models).
"""
import ast
import datetime
import json
import operator
import os
import pathlib
import sqlite3

from openai import OpenAI

# ---------- Config (all OpenAI-compatible, so one client works for all) ----------
PROVIDER = os.getenv("PROVIDER", "ollama")
PROVIDERS = {
    "ollama": ("http://localhost:11434/v1", "ollama", "qwen2.5:3b"),
    "groq": ("https://api.groq.com/openai/v1", os.getenv("GROQ_API_KEY", ""), "llama-3.1-8b-instant"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        os.getenv("GEMINI_API_KEY", ""),
        "gemini-2.0-flash",
    ),
}
BASE_URL, API_KEY, DEFAULT_MODEL = PROVIDERS[PROVIDER]
MODEL = os.getenv("MODEL", DEFAULT_MODEL)
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---------- Token-saving knobs ----------
HISTORY_WINDOW = 6      # only last N messages are sent each turn
MAX_STEPS = 4           # max tool-calling rounds per question
MAX_OUT_TOKENS = 400    # cap on every reply
TOOL_OUTPUT_LIMIT = 1200  # characters of any tool result sent back to the model

SYSTEM = (
    "You are a concise personal assistant. Answer briefly. "
    "Use tools only when needed: search for current facts, calc for math, "
    "notes to remember or recall user info."
)

# ---------- Storage ----------
DB = sqlite3.connect("agent.db")
DB.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, ts TEXT, text TEXT)")
FILES_DIR = pathlib.Path("files").resolve()
FILES_DIR.mkdir(exist_ok=True)


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
    DB.execute("INSERT INTO notes (ts, text) VALUES (?, ?)", (datetime.datetime.now().isoformat(), text))
    DB.commit()
    return "Saved."


def recall_notes(keyword: str = "") -> str:
    rows = DB.execute(
        "SELECT ts, text FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT 5", (f"%{keyword}%",)
    ).fetchall()
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


# ---------- Agent loop ----------
total_tokens = 0


def run(user_msg: str, history: list) -> str:
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
                out = REGISTRY[tc.function.name](**args)
            except Exception as e:  # keep the agent alive on tool errors
                out = f"Tool error: {e}"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)[:TOOL_OUTPUT_LIMIT]})

    return "I couldn't finish in the allowed steps. Try rephrasing."


if __name__ == "__main__":
    print(f"Agent ready | {PROVIDER}:{MODEL} | type 'exit' to quit")
    hist: list = []
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print("Agent:", run(q, hist))
            print(f"[tokens used this session: {total_tokens}]")
