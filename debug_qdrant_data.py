import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "rag", ".env")
load_dotenv(ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print(f"Inspecting 'sch_' items in knowledge_base...")

# Fetch 5 items from source_table = 'sch_'
res, _ = client.scroll(
    collection_name="knowledge_base",
    scroll_filter=Filter(
        must=[
            FieldCondition(key="source_table", match=MatchValue(value="sch_"))
        ]
    ),
    limit=5,
    with_payload=True
)

if not res:
    print("No items found for source_table='sch_'")
else:
    print(f"Found {len(res)} items. Checking payloads:")
    for point in res:
        payload = point.payload
        id_ngnh = payload.get("id_ngnh")
        print(f"ID: {point.id}")
        print(f"  Title: {payload.get('title') or payload.get('name')}")
        print(f"  id_ngnh: {id_ngnh} (Type: {type(id_ngnh)})")
        print("-" * 20)
