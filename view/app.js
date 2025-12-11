// =============================
// Config
// =============================
const CHAT_API_URL = localStorage.getItem("CHAT_API_URL") || "/chat";

const apiStatusEl = document.getElementById("apiStatus");
if (apiStatusEl) apiStatusEl.textContent = CHAT_API_URL ? CHAT_API_URL : "offline";

// =============================
// State
// =============================
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const emptyState = document.getElementById("emptyState");
const btnExport = document.getElementById("btnExport");
const btnNew = document.getElementById("btnNew");

const transcript = JSON.parse(localStorage.getItem("chat_transcript") || "[]");
let sending = false;

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
  localStorage.setItem("chat_transcript", JSON.stringify(transcript));
}

async function safeParse(res) {
  const txt = await res.text();
  try {
    return JSON.parse(txt);
  } catch {
    return { answer: txt };
  }
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

  try {
    const fd = new FormData();
    fd.append("message", text);

    const res = await fetch(CHAT_API_URL, {
      method: "POST",
      body: fd
    });

    const data = await safeParse(res);
    reply = data.answer;
  } catch (err) {
    reply = "Không gọi được API: " + err.message;
  }

  record.bot_reply = reply || "Xin lỗi, mình chưa hiểu.";
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
// OCR logic
// =============================
const btnOCR = document.getElementById("btnOCR");
const ocrInput = document.getElementById("ocrInput");

btnOCR.addEventListener("click", () => ocrInput.click());

ocrInput.addEventListener("change", async () => {
  const file = ocrInput.files[0];
  if (!file) return;

  const record = {
    user_message: "[Ảnh gửi lên để OCR]",
    bot_reply: `<span class="typing"><span>.</span><span>.</span><span>.</span></span>`,
    time: formatTime()
  };
  transcript.push(record);
  render();
  persist();

  const fd = new FormData();
  fd.append("image", file);

  let reply = "";

  try {
    const res = await fetch(CHAT_API_URL + "/ocr", { method: "POST", body: fd });
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
// Voice Recording logic
// =============================
let mediaRecorder = null;
let audioChunks = [];
const btnRecord = document.getElementById("btnRecord");

btnRecord.addEventListener("click", async () => {
  if (!mediaRecorder) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      audioChunks = [];

      const record = {
        user_message: "[Ghi âm gửi lên]",
        bot_reply: `<span class="typing"><span>.</span><span>.</span><span>.</span></span>`,
        time: formatTime()
      };
      transcript.push(record);
      render();
      persist();

      const fd = new FormData();
      fd.append("audio", audioBlob, "voice.webm");

      let reply = "";
      try {
        const res = await fetch(CHAT_API_URL + "/speech", {
          method: "POST",
          body: fd
        });
        const data = await safeParse(res);
        reply = data.answer;
      } catch (e) {
        reply = "Lỗi giọng nói: " + e.message;
      }

      record.bot_reply = reply || "Không nhận diện được giọng nói.";
      persist();
      render();
    };
  }

  if (mediaRecorder.state === "inactive") {
    mediaRecorder.start();
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
btnExport.addEventListener("click", () => {
  const payload = transcript.map((r) => ({
    user_message: r.user_message,
    bot_reply: r.bot_reply,
    time: r.time
  }));
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `chat_transcript_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

btnNew.addEventListener("click", () => {
  if (confirm("Bắt đầu phiên chat mới?")) {
    transcript.length = 0;
    persist();
    render();
  }
});

// Init
render();
