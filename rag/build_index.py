import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from db_connector import fetch_faqs


load_dotenv()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "./data/faiss_index.bin")
META_PATH = os.getenv("META_PATH", "./data/meta.jsonl")
NOTION_DUMP= os.getenv("NOTION_DUMP", "./data/notion_raw.jsonl")


model = SentenceTransformer(MODEL_NAME)


corpus = []
meta = []


# 1) Notion dump
if os.path.exists(NOTION_DUMP):
    with open(NOTION_DUMP, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = (obj.get("title") or "").strip()
                if text:
                    corpus.append(text)
                    meta.append({"source": "notion", **obj})
            except Exception:
                pass


# 2) FAQ DB
faqs = fetch_faqs()
for r in faqs:
    text = f"Q: {r['question']}\nA: {r['answer']}"
corpus.append(text)
meta.append({"source": "faq", **r})


print("Total docs:", len(corpus))


# encode
emb = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)


# FAISS index (cosine → dùng IndexFlatIP với vector chuẩn hoá)
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb.astype(np.float32))


# save
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print("Index saved.")