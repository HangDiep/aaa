import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sqlite3
import os

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# --------- Load data from FAQ.DB (Notion Data) ----------
DB_PATH = "faq.db"
if not os.path.exists(DB_PATH):
    print(f"❌ Không tìm thấy {DB_PATH}. Hãy chạy sync_faq.py trước!")
    exit()

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Lấy danh sách Category (Nhãn)
cur.execute("SELECT DISTINCT category FROM faq WHERE category IS NOT NULL")
tags = [row[0] for row in cur.fetchall()]
tags = sorted(tags)
print(f"📋 Các Category tìm thấy: {tags}")

# Lấy dữ liệu train: Dùng ANSWER (hoặc Question + Answer) để học
# Cô giáo yêu cầu: "Mô hình cần học từ CÁC CÂU TRẢ LỜI chuẩn"
cur.execute("SELECT answer, category FROM faq WHERE category IS NOT NULL")
rows = cur.fetchall()
conn.close()

all_words = []
xy = []

for answer, category in rows:
    if not answer or not category:
        continue
        
    # Tokenize câu trả lời
    tokens = tokenize(answer)
    all_words.extend(tokens)
    xy.append((tokens, category))

# Thêm dữ liệu từ intents.json (BOOKS, MAJORS, GREETING mở rộng)
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

for intent in intents['intents']:
    tag = intent['tag']
    # Nếu tag chưa có trong DB (ví dụ BOOKS, MAJORS), thêm vào
    if tag.upper() not in [t.upper() for t in tags]:
        tags.append(tag)
    
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

# Đảm bảo tags unique và sorted
tags = sorted(list(set(tags)))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"✅ Đã tải {len(xy)} mẫu dữ liệu từ Notion để train.")

# --------- Build training data ----------
x_train = []
y_train = []
for (tokenized_sentence, tag) in xy:
    bag = bag_of_words(tokenized_sentence, all_words)   # np.float32 vector
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = x_train    # (N, input_size) float32
        self.y_data = y_train    # (N,) int64
        self.n_samples = len(x_train)

    def __getitem__(self, index):
        # trả về torch tensor đúng dtype
        return torch.from_numpy(self.x_data[index]), torch.tensor(self.y_data[index], dtype=torch.long)

    def __len__(self):
        return self.n_samples

# --------- Hyperparams ----------
batch_size  = 8
hidden_size = 8
input_size  = x_train.shape[1]
output_size = len(tags)
learning_rate = 1e-3
num_epochs = 1000

print(f"input_size={input_size}, output_size={output_size}, tags={tags}")

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --------- Train loop ----------
print("🚀 Bắt đầu train model...")
for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}

FILE="data.pth"
torch.save(data,FILE)
print(f'✅ Training hoàn tất! Model đã lưu vào {FILE}')
