/*
==========================================
HO TÊN: Đỗ Thị Hồng Điệp
MSSV: 23103014
ĐỒ ÁN: Chatbot Dynamic Router - TTN University
NGÀY NỘP: 21/12/2025
Copyright © 2025. All rights reserved.
==========================================
*/

const toggleMenu = () => {
    const sidebar = document.getElementById("sideMenu");
    const overlay = document.querySelector(".overlay-menu");

    sidebar.classList.toggle("open");
    overlay.classList.toggle("active");
};

// Đóng menu khi bấm vào link (tùy chọn nhưng rất nên có)
document.querySelectorAll(".sidebar-nav a").forEach(link => {
    link.addEventListener("click", () => {
        sidebar.classList.remove("open");
        overlay.classList.remove("active");
    });
});
// ================= CHAT POPUP =================
const chatBubble = document.getElementById("chatBubble");
const chatPopup  = document.getElementById("chatPopup");
const closeChat  = document.getElementById("closeChat");

if (chatBubble && chatPopup) {
    chatBubble.addEventListener("click", () => {
        chatPopup.style.display = "flex";
        chatBubble.style.display = "none";
    });
}

if (closeChat) {
    closeChat.addEventListener("click", () => {
        chatPopup.style.display = "none";
        chatBubble.style.display = "flex";
    });
}
