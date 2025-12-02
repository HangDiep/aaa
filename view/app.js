// =============================
// Config
// =============================
const CHAT_API_URL = localStorage.getItem("CHAT_API_URL") || "/chat"; // cùng origin → không CORS
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
const btnClear = document.getElementById("btnClear");
// THÊM 3 DÒNG NÀY – QUAN TRỌNG NHẤT
const imageInput = document.getElementById("imageInput");          // input file thật
const pickImageBtn = document.getElementById("pickImage");         // nút bấm
const imagePreview = document.getElementById("imagePreview");      // vùng preview

const transcript = JSON.parse(localStorage.getItem("chat_transcript") || "[]");
let sending = false;
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
  localStorage.setItem("chat_transcript", JSON.stringify(transcript));
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
  const record = { user_message: text || "[Ảnh]", bot_reply: "…", time: formatTime(now) };
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
if (btnExport) {
  btnExport.addEventListener("click", () => {
    const payload = transcript.map((r) => ({
      user_message: r.user_message,
      bot_reply: r.bot_reply,
      intent_tag: null,
      confidence: null,
      time: r.time,
    }));
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), {
      href: url,
      download: `chat_transcript_${Date.now()}.json`,
    });
    a.click();
    URL.revokeObjectURL(url);
  });
}
if (btnClear) {
  btnClear.addEventListener("click", () => {
    if (confirm("Xóa toàn bộ phiên chat hiện tại?")) {
      transcript.splice(0, transcript.length);
      persist();
      render();
    }
  });
}
// =============================
// Xử lý chọn ảnh + preview
// =============================
if (pickImageBtn && imageInput) {
  pickImageBtn.addEventListener("click", () => {
    imageInput.click();
  });
}

if (imageInput) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) {
      imagePreview.innerHTML = "";
      return;
    }
    const url = URL.createObjectURL(file);
    imagePreview.innerHTML = `
      <div style="padding:8px 0; color:#22d3ee; font-size:13px">
        Đã chọn: <strong>${escapeHtml(file.name)}</strong> (${(file.size/1024).toFixed(1)} KB)
        <span style="margin-left:12px; color:#94a3b8; cursor:pointer; text-decoration:underline" onclick="this.parentElement.parentElement.innerHTML=''">
          Hủy
        </span>
      </div>
      <img src="${url}" style="max-width:100%; max-height:300px; border-radius:8px; border:1px solid #334155">
    `;
  });
}
render();