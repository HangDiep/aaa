import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load env variables
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, "rag", ".env")
load_dotenv(env_path)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "knowledge_base"

def main():
    print(f"🔌 Connecting to Qdrant: {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        if not client.collection_exists(COLLECTION):
            print(f"❌ Collection '{COLLECTION}' does not exist.")
            return

        print(f"🔍 Scanning collection '{COLLECTION}' for tables...")
        
        unique_tables = {} # table_name -> count
        next_offset = None
        total_points = 0
        
        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION,
                limit=500,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            
            for p in points:
                total_points += 1
                payload = p.payload or {}
                tbl = payload.get("source_table", "UNKNOWN")
                unique_tables[tbl] = unique_tables.get(tbl, 0) + 1
                    
            print(f"  Processed {total_points} points...", end="\r")
            
            if next_offset is None:
                break
        
        print(f"\n✅ Scan complete. Total items: {total_points}\n")
        print("📊 Tables found in Qdrant:")
        print(f"{'Table Name':<30} | {'Count':<10}")
        print("-" * 45)
        for tbl, count in sorted(unique_tables.items()):
            print(f"{tbl:<30} | {count:<10}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
