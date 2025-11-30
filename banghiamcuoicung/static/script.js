// static/script.js
const ws = new WebSocket('ws://localhost:8000/ws');
const chatbox = document.getElementById('chatbox');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');

let mediaRecorder = null;
let stream = null;
let audioBuffer = []; // Lưu tạm các đoạn âm thanh

// === NHẬN TEXT TỪ SERVER ===
ws.onmessage = (event) => {
    const text = event.data.trim();
    if (!text) return;
    const msg = document.createElement('div');
    msg.className = 'message';
    msg.innerHTML = `<strong>Bạn:</strong> ${text}`;
    chatbox.appendChild(msg);
    chatbox.scrollTop = chatbox.scrollHeight;
};

ws.onopen = () => {
    status.textContent = 'Kết nối thành công!';
    status.style.color = '#4ade80';
};

ws.onerror = () => {
    status.textContent = 'Lỗi kết nối!';
    status.style.color = '#f87171';
};

// === BẮT ĐẦU THU ÂM ===
startBtn.onclick = async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

        audioBuffer = []; // Reset buffer khi bắt đầu

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioBuffer.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            // Gửi toàn bộ đoạn thu âm khi dừng
            const blob = new Blob(audioBuffer, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.onload = () => {
                const base64data = reader.result.split(',')[1];
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(base64data);
                }
            };
            reader.readAsDataURL(blob);

            // Dừng stream
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            mediaRecorder = null;
        };

        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        startBtn.textContent = 'Đang thu...';
        status.textContent = 'Đang thu – Nói đi!';
        status.style.color = '#60a5fa';
    } catch (err) {
        alert('Micro lỗi: ' + err.message);
    }
};

// === DỪNG THU ÂM → GỬI VÀ HIỆN TEXT ===
stopBtn.onclick = () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    startBtn.textContent = 'Bắt đầu';
    status.textContent = 'Đã dừng – Đang xử lý...';
    status.style.color = '#3b82f6';
};

// Dừng khi đóng tab
window.onbeforeunload = () => {
    if (mediaRecorder) mediaRecorder.stop();
    if (stream) stream.getTracks().forEach(t => t.stop());
};