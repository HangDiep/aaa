DROP TABLE IF EXISTS conversations;
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    bot_reply TEXT,
    intent_tag TEXT,
    confidence REAL,
    time TEXT
);
