// =============================
// Config
// =============================
const CHAT_API_URL = localStorage.getItem("CHAT_API_URL") || "/chat"; // cùng origin → không CORS
// Voice Server đã được mount vào view/app.py (port 8000)
const WS_URL = "ws://127.0.0.1:8000/ws";

const apiStatusEl = document.getElementById("apiStatus");
if (apiStatusEl) apiStatusEl.textContent = CHAT_API_URL ? CHAT_API_URL : "offline";
// =============================
// State
// =============================
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const emptyState = document.getElementById("emptyState");
const btnNew = document.getElementById("btnNew");
const btnRecord = document.getElementById("btnRecord");
// THÊM 3 DÒNG NÀY – QUAN TRỌNG NHẤT
const imageInput = document.getElementById("imageInput");          // input file thật
const pickImageBtn = document.getElementById("pickImage");         // nút bấm
const imagePreview = document.getElementById("imagePreview");      // vùng preview

const transcript = [];
let sending = false;
let ws = null;
let mediaRecorder = null;
let audioChunks = [];
// =============================
// Utils / UI helpers
// =============================
function formatTime(d = new Date()) {
  return d.toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function msgTemplate(role, text, time) {
  const content = role === "bot" ? (text || "") : escapeHtml(text || "").replace(/\n/g, "<br/>");
  return `
    <article class="msg ${role}">
      <div class="avatar" aria-hidden="true">${role === "bot" ? "🤖" : "🧑"}</div>
      <div>
        <div class="bubble">${escapeHtml(text || "").replace(/\n/g, "<br/>")}</div>
        <div class="meta">${role === "bot" ? "Bot" : "Bạn"} · ${time || formatTime()}</div>
      </div>
    </article>`;
}
function render() {
  if (!chat) return;
  chat.innerHTML = "";
  if (!transcript.length) {
    if (emptyState) chat.appendChild(emptyState);
  } else {
    transcript.forEach((row) => {
      chat.insertAdjacentHTML("beforeend", msgTemplate("user", row.user_message, row.time));
      chat.insertAdjacentHTML("beforeend", msgTemplate("bot", row.bot_reply, row.time));
    });
  }
  chat.scrollTop = chat.scrollHeight;
}
function persist() {
  // KHÔNG LƯU GÌ – F5 / đóng tab là mất
}

async function safeParse(res) {
  const txt = await res.text(); // luôn đọc text trước
  try {
    return JSON.parse(txt); // nếu là JSON hợp lệ
  } catch {
    return { answer: txt }; // nếu không phải JSON -> dùng text làm answer
  }
}
// =============================
// Offline mock
// =============================
function offlineMock(q) {
  const cancelWords = ["hủy", "huỷ", "huy", "cancel", "thoát", "dừng", "đổi chủ đề", "doi chu de"];
  if (cancelWords.includes(q.toLowerCase())) {
    return "Đã hủy luồng hiện tại. Bạn muốn hỏi gì tiếp?";
  }
  if (/mở cửa|giờ mở/.test(q.toLowerCase())) return "Thư viện mở cửa 7:30–17:00, Thứ 2–Thứ 6.";
  if (/mượn.*sách|muon sach/.test(q.toLowerCase())) return "Bạn cần thẻ sinh viên để mượn sách. Đến quầy thủ thư để hỗ trợ nhé!";
  return "Chế độ offline: mình chưa hiểu, hãy kết nối API để có câu trả lời chính xác.";
}
// =============================
// Send logic
// =============================
async function send() {
  if (sending) return;
  const text = (input && input.value ? input.value : "").trim();
  const imageFile = imageInput ? imageInput.files[0] : null;
  if (!text && !imageFile) return; // Không gửi nếu cả hai rỗng
  sending = true;
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.textContent = "Đang gửi...";
  }
  if (input) input.value = "";
  if (imageInput) imageInput.value = "";
  if (imagePreview) imagePreview.innerHTML = "";  // Xóa preview sau khi gửi
  const now = new Date();
  const record = { 
    user_message: text || "[Ảnh]", bot_reply: `<span class="typing"><span>.</span><span>.</span><span>.</span></span>`, 
    time: formatTime(now) };
  transcript.push(record);
  persist();
  render();
  let reply = "";
  try {
    if (CHAT_API_URL) {
      const fd = new FormData();
      fd.append("message", text);
      if (imageFile) fd.append("image", imageFile); // Append file nếu có
      const res = await fetch(CHAT_API_URL || "/chat", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await safeParse(res); // an toàn với cả text lẫn JSON
      reply = (data && data.answer) || "";
    } else {
      reply = offlineMock(text);
    }
  } catch (err) {
    reply = `Không gọi được API (${err.message}). Mẹo: thiết lập CHAT_API_URL trong localStorage, ví dụ: localStorage.setItem('CHAT_API_URL', 'http://127.0.0.1:8000/chat')`;
  }
  record.bot_reply = reply || "Xin lỗi, mình chưa hiểu ý bạn.";
  persist();
  render();
  sending = false;
  if (sendBtn) {
    sendBtn.disabled = false;
    sendBtn.textContent = "Gửi";
  }
}

// =============================
// WebSocket Voice Recognition
// =============================
function initWebSocket() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => console.log("WS connected");
  ws.onerror = () => console.log("WS error");
  ws.onclose = () => console.log("WS closed");

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.sender === "user") {
      transcript.push({
        user_message: msg.text,
        bot_reply: "",
        time: formatTime()
      });
    }

    if (msg.sender === "bot") {
      transcript.push({
        user_message: "",
        bot_reply: msg.text,
        time: formatTime()
      });
    }

    persist();
    render();
  };
}

// Start WebSocket
initWebSocket();

// =============================
// Voice Recording (WebSocket Stream)
// =============================
if (btnRecord) btnRecord.addEventListener("click", async () => {

  if (!mediaRecorder) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
          const buffer = await event.data.arrayBuffer();
          const bytes = new Uint8Array(buffer);
          let binary = "";
          bytes.forEach(b => binary += String.fromCharCode(b));
          const base64 = btoa(binary);
          ws.send(base64);
        }
      };
    } catch (e) {
      alert("Không thể truy cập microphone: " + e.message);
      return;
    }
  }

  if (mediaRecorder.state === "inactive") { mediaRecorder.start(300); // gửi 0.3s một lần 
  btnRecord.textContent = "⏹ Dừng"; 
  btnRecord.classList.add("recording"); } 
  else { 
    mediaRecorder.stop(); 
    btnRecord.textContent = "🎤"; 
    btnRecord.classList.remove("recording"); }
});

// =============================
// Events
// =============================
if (sendBtn) sendBtn.addEventListener("click", send);
if (input) {
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
}
document.querySelectorAll(".chip").forEach((ch) => {
  ch.addEventListener("click", () => {
    if (!input) return;
    input.value = ch.dataset.fill || "";
    input.focus();
  });
});
// =============================
// Xử lý chọn ảnh + preview + gửi kèm tin nhắn
// =============================
if (pickImageBtn && imageInput) {
  pickImageBtn.addEventListener("click", (e) => {
    e.preventDefault();
    imageInput.click();
  });
}

if (imageInput) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    imagePreview.innerHTML = `
     <div class="thumb">
      <img src="${url}">
    </div>
    
    <div class="meta">
        Đã chọn: <strong>${escapeHtml(file.name)}</strong>
        (${(file.size / 1024).toFixed(1)} KB)
      </div>

      <span class="remove" id="removeImage">Hủy</span>
    `;

    document.getElementById("removeImage").onclick = () => {
      imageInput.value = "";
      imagePreview.innerHTML = "";
    };

    // Tự động focus vào ô nhập để người dùng gõ thêm caption nếu muốn
    input && input.focus();
  });
}

// Cho phép gửi bằng Enter (không Shift) dù có ảnh hay không
if (input) {
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
}

// Hàm send – đã đúng, chỉ bổ sung hiển thị ảnh trong lịch sử chat
async function send() {
  if (sending) return;
  const text = input?.value?.trim() || "";
  const imageFile = imageInput?.files[0];

  if (!text && !imageFile) return;

  sending = true;
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.textContent = "Đang gửi...";
  }

  // Xóa input + preview ngay lập tức để tránh gửi 2 lần
  if (input) input.value = "";
  if (imageInput) imageInput.value = "";
  if (imagePreview) imagePreview.innerHTML = "";

  const now = new Date();
  const userMessage = text || "[Đã gửi một ảnh]";

  // Hiển thị tin nhắn người dùng (có ảnh nếu có)
  let userHtml = msgTemplate("user", text || "📷 Đã gửi ảnh", formatTime(now));
  if (imageFile) {
    const imgUrl = URL.createObjectURL(imageFile);
    userHtml = `
      <article class="msg user">
        <div class="avatar">Người dùng</div>
        <div>
          <div class="bubble">
            ${text ? escapeHtml(text) + "<br/><br/>" : ""}
            <img src="${imgUrl}" style="max-width:100%; border-radius:8px; margin-top:8px;">
          </div>
          <div class="meta">Bạn · ${formatTime(now)}</div>
        </div>
      </article>
    `;
  }

  chat.insertAdjacentHTML("beforeend", userHtml);
  chat.scrollTop = chat.scrollHeight;

  // Lưu vào transcript (chỉ text + ghi chú ảnh)
  transcript.push({
    user_message: text || "[ảnh]",
    bot_reply: `<span class="typing"><span>.</span><span>.</span><span>.</span></span>`,
    time: formatTime(now)
  });
  persist();

  let reply = "Xin lỗi, có lỗi xảy ra.";

  try {
    if (CHAT_API_URL) {
      const fd = new FormData();
      fd.append("message", text);
      if (imageFile) fd.append("image", imageFile, imageFile.name);

      const res = await fetch(CHAT_API_URL, { method: "POST", body: fd });
      if (!res.ok) throw new Error("HTTP " + res.status);

      const data = await safeParse(res);
      reply = data?.answer || "Bot không phản hồi.";
    } else {
      reply = offlineMock(text);
    }
  } catch (err) {
    reply = `Lỗi kết nối: ${err.message}`;
  }

  // Thêm phản hồi bot
  chat.insertAdjacentHTML("beforeend", msgTemplate("bot", reply, formatTime(now)));
  transcript[transcript.length - 1].bot_reply = reply;
  persist();
  chat.scrollTop = chat.scrollHeight;

  sending = false;
  if (sendBtn) {
    sendBtn.disabled = false;
    sendBtn.textContent = "Gửi";
  }
}

btnNew.addEventListener("click", () => {
  if (confirm("Bắt đầu phiên chat mới?")) {
    transcript.length = 0;
    persist();
    render();
  }
});
render();