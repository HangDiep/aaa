import os, json
import numpy as np
import faiss
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from db_connector import search_faq_like


load_dotenv()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "./data/faiss_index.bin")
META_PATH = os.getenv("META_PATH", "./data/meta.jsonl")
TOP_K = int(os.getenv("TOP_K", 5))


_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(INDEX_PATH)
_meta = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        _meta.append(json.loads(line))


# tiện ích
def _encode(text: str):
    v = _model.encode([text], normalize_embeddings=True)
    return np.array(v, dtype=np.float32)


def semantic_search(query: str, k: int = TOP_K) -> List[Dict]:
    vq = _encode(query)
    scores, idx = _index.search(vq, k)
    hits = []
    for score, i in zip(scores[0], idx[0]):
        if i < 0:
            continue
        item = {"score": float(score), "meta": _meta[i]}
        hits.append(item)
    return hits


def hybrid_search(query: str, k: int = TOP_K) -> Dict[str, List[Dict]]:
    sem = semantic_search(query, k)
    kw = search_faq_like(query, limit=k)
# có thể thêm BM25 ở đây
    return {"semantic": sem, "faq_like": kw}