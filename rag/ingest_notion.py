import os, json
from pathlib import Path
from dotenv import load_dotenv
from notion_client import Client, errors  # type: ignore

# Luôn nạp .env nằm cạnh file
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID  = os.getenv("NOTION_DATABASE_ID")
OUT_PATH     = os.getenv("NOTION_DUMP", "./data/notion_raw.jsonl")

if not NOTION_TOKEN or not NOTION_TOKEN.startswith("secret_"):
    raise SystemExit("❌ NOTION_TOKEN thiếu hoặc không hợp lệ. Kiểm tra rag/.env (không để dấu ngoặc).")

if not DATABASE_ID or len(DATABASE_ID.replace("-", "")) != 32:
    raise SystemExit("❌ NOTION_DATABASE_ID thiếu hoặc sai. Lấy 32 ký tự trước '?v=' trong Copy link DB.")

client = Client(auth=NOTION_TOKEN)

def page_to_text(page: dict) -> str:
    # Lấy title; có thể mở rộng để lấy children
    props = page.get("properties", {})
    title = ""
    for p in props.values():
        if p.get("type") == "title":
            title = " ".join(t.get("plain_text", "") for t in p.get("title", [])).strip()
            break
    return title or page.get("id", "")

def fetch_database(database_id: str):
    results, has_more, start_cursor = [], True, None
    while has_more:
        resp = client.databases.query(database_id=database_id, start_cursor=start_cursor)
        results.extend(resp["results"])
        has_more    = resp.get("has_more", False)
        start_cursor= resp.get("next_cursor")
    return results

if __name__ == "__main__":
    # Thử retrieve DB để báo lỗi sớm (share/quyền/ID)
    try:
        _ = client.databases.retrieve(database_id=DATABASE_ID)
    except errors.APIResponseError as e:
        msg = "❌ Notion 401 Unauthorized. Kiểm tra:\n" \
              "- Token đúng dạng secret_ và thuộc đúng workspace.\n" \
              "- Database đã Share → Invite integration → Can read.\n" \
              "- NOTION_DATABASE_ID đúng (32 ký tự trước '?v=')."
        raise SystemExit(f"{msg}\nDetail: {e}")

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages = fetch_database(DATABASE_ID)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            item = {
                "id":    p.get("id"),
                "title": page_to_text(p),
                "url":   p.get("url"),
                "source":"notion",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(pages)} records to {out_path}")
