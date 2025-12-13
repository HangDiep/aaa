// static/script.js

window.addEventListener("DOMContentLoaded", () => {
    console.log("✅ DOM loaded, init app...");

    const chatbox = document.getElementById("chatbox");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const status = document.getElementById("status");

    if (!chatbox || !startBtn || !stopBtn || !status) {
        console.error("❌ Thiếu phần tử HTML! Kiểm tra lại index.html");
        return;
    }

    let ws = null;
    let mediaRecorder = null;
    let stream = null;
    let audioBuffer = [];

    // ===== KẾT NỐI WEBSOCKET =====
    function connectWebSocket() {
        console.log("🔌 Đang kết nối WebSocket...");
        ws = new WebSocket("ws://localhost:9000/ws");

        ws.onopen = () => {
            console.log("✅ WebSocket connected");
            status.textContent = "Kết nối thành công!";
            status.style.color = "#4ade80";
        };

        ws.onerror = (err) => {
            console.error("❌ WebSocket error:", err);
            status.textContent = "Lỗi kết nối WebSocket!";
            status.style.color = "#f87171";
        };

        ws.onclose = () => {
            console.warn("⚠️ WebSocket closed");
            status.textContent = "WebSocket đã đóng. Refresh lại trang nếu cần.";
            status.style.color = "#fbbf24";
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("📩 Nhận message:", data);

                const msg = document.createElement("div");
                msg.className = "message";

                if (data.sender === "user") {
                    msg.innerHTML = `<strong>Tôi:</strong> ${data.text}`;
                } else if (data.sender === "bot") {
                    msg.innerHTML = `<strong>Bot:</strong> ${data.text}`;
                } else {
                    msg.innerHTML = `<strong>?</strong> ${event.data}`;
                }

                chatbox.appendChild(msg);
                chatbox.scrollTop = chatbox.scrollHeight;

            } catch (err) {
                console.error("❌ Lỗi parse JSON:", err, event.data);
            }
        };
    }

    connectWebSocket();

    // ===== BẮT ĐẦU THU ÂM =====
    startBtn.onclick = async () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            console.log("⏺ Đã thu rồi, bỏ qua");
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Trình duyệt không hỗ trợ thu âm.");
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

            audioBuffer = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    audioBuffer.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                console.log("🛑 Dừng thu, đang gửi dữ liệu...");
                const blob = new Blob(audioBuffer, { type: "audio/webm" });
                const reader = new FileReader();

                reader.onload = () => {
                    const result = reader.result || "";
                    const base64data = result.split(",")[1];

                    if (!base64data) {
                        console.error("❌ Không đọc được base64.");
                        status.textContent = "Lỗi đọc âm thanh.";
                        status.style.color = "#f87171";
                        return;
                    }

                    if (!ws || ws.readyState !== WebSocket.OPEN) {
                        console.error("❌ WebSocket chưa mở.");
                        status.textContent = "WebSocket chưa sẵn sàng.";
                        status.style.color = "#f87171";
                        return;
                    }

                    ws.send(base64data);
                    console.log("📤 Đã gửi audio lên server");
                    status.textContent = "Đã gửi – đang xử lý...";
                    status.style.color = "#3b82f6";
                };

                reader.readAsDataURL(blob);

                if (stream) {
                    stream.getTracks().forEach((t) => t.stop());
                }
                stream = null;
                mediaRecorder = null;
            };

            mediaRecorder.start();
            console.log("⏺ Bắt đầu thu âm...");
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.textContent = "Đang thu...";
            status.textContent = "Đang thu – Nói đi!";
            status.style.color = "#60a5fa";

        } catch (err) {
            console.error("❌ Lỗi micro:", err);
            alert("Micro lỗi: " + err.message);
        }
    };

    // ===== DỪNG THU ÂM =====
    stopBtn.onclick = () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }

        startBtn.disabled = false;
        stopBtn.disabled = true;
        startBtn.textContent = "Bắt đầu";
        status.textContent = "Đã dừng – đang xử lý...";
        status.style.color = "#3b82f6";
    };

    // Dọn dẹp
    window.addEventListener("beforeunload", () => {
        if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
        if (stream) stream.getTracks().forEach((t) => t.stop());
        if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    });
});
