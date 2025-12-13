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