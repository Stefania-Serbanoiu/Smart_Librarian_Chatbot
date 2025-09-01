#backend/db.py
import json
import chromadb
from typing import List, Dict, Any
from chromadb.utils import embedding_functions
from pathlib import Path
from .config import OPENAI_API_KEY, EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME, DATA_FILE


def _load_books() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")
    return json.loads(Path(DATA_FILE).read_text(encoding="utf-8"))


def _seed_if_empty(collection):
    if collection.count() > 0:
        return
    books = _load_books()
    ids, docs, metas = [], [], []
    for b in books:
        themes_str = ", ".join(b.get("themes", []))
        ids.append(b["id"])
        docs.append(f"Title: {b['title']}\nSummary: {b['summary']}\nThemes: {themes_str}")
        metas.append({"title": b["title"], "themes": themes_str})
    collection.add(ids=ids, documents=docs, metadatas=metas)


def get_collection():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )
    _seed_if_empty(col)
    return col


def search(query: str, top_k: int = 4):
    col = get_collection()
    res = col.query(query_texts=[query], n_results=top_k)
    hits = []
    for i in range(len(res.get("ids", [[]])[0])):
        hits.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
        })
    return hits
