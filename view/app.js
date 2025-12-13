// =============================
// Config
// =============================
const CHAT_API_URL = localStorage.getItem("CHAT_API_URL") || "/chat";
// Voice Server đã được mount vào view/app.py (port 8000)
const WS_URL = "ws://127.0.0.1:8000/ws";

const apiStatusEl = document.getElementById("apiStatus");
if (apiStatusEl) apiStatusEl.textContent = CHAT_API_URL ? CHAT_API_URL : "offline";

// =============================
// Session ID (for conversation memory)
// =============================
let sessionId = localStorage.getItem('chat_session_id');
if (!sessionId) {
  sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  localStorage.setItem('chat_session_id', sessionId);
  console.log('🆕 New session created:', sessionId);
} else {
  console.log('📌 Existing session:', sessionId);
}

// =============================
// State
// =============================
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const emptyState = document.getElementById("emptyState");
const btnNew = document.getElementById("btnNew");
const btnRecord = document.getElementById("btnRecord");

// Transcript: Không lưu lại sau khi F5 (theo yêu cầu)
const transcript = [];

let sending = false;
let ws = null;
let mediaRecorder = null;
let audioChunks = [];

// =============================
// Utils
// =============================
function formatTime(d = new Date()) {
  return d.toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
    year: "numeric"
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}

function msgTemplate(role, text, time) {
  const content = role === "bot" ? (text || "") : escapeHtml(text || "").replace(/\n/g, "<br/>");
  return `
    <article class="msg ${role}">
      <div class="avatar">${role === "bot" ? "🤖" : "🧑"}</div>
      <div>
        <div class="bubble">${content}</div>
        <div class="meta">${role === "bot" ? "Bot" : "Bạn"} · ${time || formatTime()}</div>
      </div>
    </article>`;
}

function render() {
  chat.innerHTML = "";
  if (!transcript.length) {
    chat.appendChild(emptyState);
  } else {
    transcript.forEach((row) => {
      chat.insertAdjacentHTML("beforeend", msgTemplate("user", row.user_message, row.time));
      chat.insertAdjacentHTML("beforeend", msgTemplate("bot", row.bot_reply, row.time));
    });
  }
  chat.scrollTop = chat.scrollHeight;
}

function persist() {
  // Đã tắt lưu transcript theo yêu cầu (F5 là mất)
}

async function safeParse(res) {
  const txt = await res.text();
  try { return JSON.parse(txt); }
  catch { return { answer: txt }; }
}

// =============================
// Send logic
// =============================
async function send() {
  if (sending) return;

  const text = input.value.trim();
  if (!text) return;

  sending = true;
  sendBtn.disabled = true;
  sendBtn.textContent = "Đang gửi...";
  input.value = "";

  const now = new Date();
  const record = {
    user_message: text,
    bot_reply: `<span class="typing"><span>.</span><span>.</span><span>.</span></span>`,
    time: formatTime(now)
  };

  transcript.push(record);
  persist();
  render();

  let reply = "";

  // CHAT
  try {
    const fd = new FormData();
    fd.append("message", text);
    fd.append("session_id", sessionId);  // ✅ Send session ID

    const res = await fetch(CHAT_API_URL, {
      method: "POST",
      body: fd
    });

    const data = await safeParse(res);
    reply = data.answer || data.output || "Không có phản hồi.";

  } catch (e) {
    reply = "Không gọi được API: " + e.message;
  }

  record.bot_reply = reply;
  persist();
  render();

  sending = false;
  sendBtn.disabled = false;
  sendBtn.textContent = "Gửi";
}

if (sendBtn) sendBtn.addEventListener("click", send);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// =============================
// OCR Logic
// =============================
const btnOCR = document.getElementById("btnOCR");
const ocrInput = document.getElementById("ocrInput");

if (btnOCR) btnOCR.addEventListener("click", () => ocrInput.click());

if (ocrInput) ocrInput.addEventListener("change", async () => {
  const file = ocrInput.files[0];
  if (!file) return;

  const record = {
    user_message: "[Ảnh gửi lên để OCR]",
    bot_reply: `<span class="typing"><span>.</span><span>.</span><span>.</span></span>`,
    time: formatTime()
  };

  transcript.push(record);
  persist();
  render();

  let reply = "";
  const fd = new FormData();
  fd.append("image", file);

  try {
    // Post to /ocr endpoint (assuming same host)
    const res = await fetch("/ocr", { method: "POST", body: fd });
    const data = await safeParse(res);
    reply = data.answer;
  } catch (e) {
    reply = "Lỗi OCR: " + e.message;
  }

  record.bot_reply = reply || "Không đọc được văn bản.";
  persist();
  render();
});

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

  if (mediaRecorder.state === "inactive") {
    mediaRecorder.start(300);   // gửi 0.3s một lần
    btnRecord.textContent = "⏹ Dừng";
    btnRecord.classList.add("recording");
  } else {
    mediaRecorder.stop();
    btnRecord.textContent = "🎤 Ghi âm";
    btnRecord.classList.remove("recording");
  }
});

// =============================
// Export & New
// =============================

btnNew.addEventListener("click", () => {
  if (confirm("Bắt đầu phiên chat mới?")) {
    transcript.length = 0;
    persist();
    render();

    // Reset session ID (Keep logic from diep)
    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('chat_session_id', sessionId);
    console.log('🔄 New session started:', sessionId);
  }
});

// Init
render();
