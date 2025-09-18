DROP TABLE IF EXISTS conversations;
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    bot_reply TEXT,
    intent_tag TEXT,
    confidence REAL,
    time TEXT
);
CREATE TABLE IF NOT EXISTS faq (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE,   -- id Notion page_id
    question TEXT,
    answer TEXT,
    updated_at TEXT            -- ngày cập nhật cuối
);
