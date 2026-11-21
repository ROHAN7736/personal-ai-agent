"""
Local PDF search for the agent. Drop PDFs into the docs/ folder.
Uses Ollama embeddings (free, local). Index is cached on disk and rebuilt
only when the PDFs change, so repeat questions cost no extra embedding time.

Needs:  pip install pypdf numpy   and   ollama pull nomic-embed-text
"""
import json
import pathlib

import numpy as np
from openai import OpenAI
from pypdf import PdfReader

DOCS = pathlib.Path("docs")
DOCS.mkdir(exist_ok=True)
STORE = pathlib.Path("docs_index")  # creates docs_index.json + docs_index.npy
EMBED_MODEL = "nomic-embed-text"
CHUNK, OVERLAP, TOP_K = 500, 80, 3

# Embeddings always use local Ollama, even if chat uses Groq/Gemini.
_ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=180)
_cache: dict = {}


def _signature() -> list:
    return [[p.name, p.stat().st_size, int(p.stat().st_mtime)] for p in sorted(DOCS.glob("*.pdf"))]


def _chunks(path: pathlib.Path) -> list:
    out = []
    for page_no, page in enumerate(PdfReader(path).pages, 1):
        text = " ".join((page.extract_text() or "").split())
        for i in range(0, len(text), CHUNK - OVERLAP):
            piece = text[i : i + CHUNK].strip()
            if len(piece) > 40:
                out.append({"file": path.name, "page": page_no, "text": piece})
    return out


def _embed(texts: list) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), 32):
        r = _ollama.embeddings.create(model=EMBED_MODEL, input=texts[i : i + 32])
        vecs += [d.embedding for d in r.data]
    v = np.array(vecs, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _load(sig: list):
    meta_f, vec_f = STORE.with_suffix(".json"), STORE.with_suffix(".npy")
    if meta_f.exists() and vec_f.exists():
        meta = json.loads(meta_f.read_text())
        if meta["sig"] == sig:
            return meta["chunks"], np.load(vec_f)
    chunks = [c for p in sorted(DOCS.glob("*.pdf")) for c in _chunks(p)]
    if not chunks:
        return [], None
    vecs = _embed(["search_document: " + c["text"] for c in chunks])
    meta_f.write_text(json.dumps({"sig": sig, "chunks": chunks}))
    np.save(vec_f, vecs)
    return chunks, vecs


def search(query: str) -> str:
    sig = _signature()
    if _cache.get("sig") != sig:
        _cache["idx"], _cache["sig"] = _load(sig), sig
    chunks, vecs = _cache["idx"]
    if not chunks:
        return "No searchable PDF text found in the docs/ folder."
    q = _embed(["search_query: " + query])[0]
    top = np.argsort(-(vecs @ q))[:TOP_K]
    return "\n".join(f"[{chunks[i]['file']} p{chunks[i]['page']}] {chunks[i]['text']}" for i in top)
