const CHAT_API_URL = "/chat";

const toggleBtn = document.getElementById("toggleBtn");
const widget = document.getElementById("chatWidget");
const chatBody = document.getElementById("chatBody");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const fileInput = document.getElementById("fileInput");
const cameraInput = document.getElementById("cameraInput");
const attachBtn = document.getElementById("attachBtn");
const cameraBtn = document.getElementById("cameraBtn");
const micBtn = document.getElementById("micBtn");
const preview = document.getElementById("preview");

let mediaRecorder;
let audioChunks = [];

/* ----- TOGGLE ----- */
toggleBtn.onclick = () => widget.classList.toggle("open");

/* ----- QUICK CHIP ----- */
document.querySelectorAll(".quick-chip").forEach(chip => {
  chip.onclick = () => {
    input.value = chip.dataset.text;
    input.focus();
  };
});

/* ========== IMAGE UPLOAD ========== */
attachBtn.onclick = () => fileInput.click();
fileInput.onchange = e => handleFile(e.target.files[0]);

cameraBtn.onclick = () => cameraInput.click();
cameraInput.onchange = e => handleFile(e.target.files[0]);

function handleFile(file) {
  if (!file) return;

  const url = URL.createObjectURL(file);
  preview.innerHTML = `
    <div>Đã chọn: <strong>${file.name}</strong></div>
    <img src="${url}">
    <div onclick="clearPreview()" style="margin-top:6px;cursor:pointer;">Hủy</div>
  `;
  preview.file = file;
}

window.clearPreview = () => {
  preview.innerHTML = "";
  preview.file = null;
  preview.audioBlob = null;
  fileInput.value = cameraInput.value = "";
};

/* ========== AUDIO RECORDING ========== */
micBtn.onclick = async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    micBtn.classList.remove("active");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      const url = URL.createObjectURL(blob);

      preview.innerHTML = `
        <div>Đã ghi âm</div>
        <audio controls src="${url}"></audio>
        <div onclick="clearPreview()" style="cursor:pointer;margin-top:6px;">Hủy</div>
      `;
      preview.audioBlob = blob;

      stream.getTracks().forEach(t => t.stop());
    };

    mediaRecorder.start();
    micBtn.classList.add("active");
  } catch (err) {
    alert("Không thể mở micro: " + err.message);
  }
};

/* ========== SEND MESSAGE ========== */
async function sendMessage() {
  let text = input.value.trim();
  const img = preview.file;
  const audio = preview.audioBlob;

  if (!text && !img && !audio) return;

  addMessage("user", text || (img ? "📷 Đã gửi ảnh" : "🎤 Đã gửi ghi âm"), img ? URL.createObjectURL(img) : null);

  input.value = "";
  clearPreview();

  const typing = document.createElement("div");
  typing.className = "message bot";
  typing.innerHTML = `<div class="bubble"><div>Đang soạn...</div></div>`;
  chatBody.appendChild(typing);
  chatBody.scrollTop = chatBody.scrollHeight;

  const form = new FormData();
  form.append("message", text);
  if (img) form.append("image", img);
  if (audio) form.append("audio", audio, "voice.webm");

  try {
    const res = await fetch(CHAT_API_URL, { method: "POST", body: form });
    const data = await res.json();
    typing.remove();
    addMessage("bot", data.answer || "Bot chưa phản hồi.");
  } catch (err) {
    typing.remove();
    addMessage("bot", "Không thể kết nối server.");
  }
}

function addMessage(sender, text, imgUrl) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  
  let html = `<div class="bubble">${text.replace(/\n/g, "<br>")}`;
  if (imgUrl) html += `<img src="${imgUrl}">`;
  html += `</div>`;

  div.innerHTML = html;
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
}

sendBtn.onclick = sendMessage;

input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
