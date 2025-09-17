from dotenv import load_dotenv
from notion_client import Client
from datetime import datetime, timezone
import os

load_dotenv('D:/HTML/chat2/rag/.env')
client = Client(auth=os.getenv("NOTION_TOKEN"))
DBID = os.getenv("NOTION_DATABASE_ID")

def rt(text):
    # rich_text tiện dụng
    return [{"type":"text","text":{"content": text or ""}}]

def upsert_faq(question: str, answer: str,
               category: str|None = None,
               language: str = "Tiếng Việt",
               approved: bool = True):
    """
    Tìm row có Question == question.
    - Nếu có: update Answer/Category/Language/Approved/Last Updated
    - Nếu không: create mới.
    """
    # 1) Tìm exact-match theo Question
    res = client.databases.query(
        database_id=DBID,
        filter={"property":"Question","rich_text":{"equals": question}}
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    props = {
        "Question": {"rich_text": rt(question)},
        "Answer":   {"rich_text": rt(answer)},
        "Approved": {"checkbox": approved},
        "Language": {"select": {"name": language}},
        "Last Updated": {"date": {"start": now_iso}},
        # "Tên" (title) có thể dùng chính Question cho dễ nhìn
        "Tên": {"title": rt(question[:200])},
    }
    if category:
        props["Category"] = {"select": {"name": category}}

    if res.get("results"):
        page_id = res["results"][0]["id"]
        client.pages.update(page_id=page_id, properties=props)
        return ("updated", page_id)
    else:
        page = client.pages.create(parent={"database_id": DBID}, properties=props)
        return ("created", page["id"])

import json

if __name__ == "__main__":
    with open("data/faqs.json", "r", encoding="utf-8") as f:
        faqs = json.load(f)

    for faq in faqs:
        status, pid = upsert_faq(
            question=faq["question"],
            answer=faq["answer"],
            category=faq.get("category"),
            language=faq.get("language", "Tiếng Việt"),
            approved=faq.get("approved", True)
        )
        print("✅", status, pid, "|", faq["question"])
    print("✅ Done.")