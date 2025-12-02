@echo off
setlocal
rem ===== timestamp vào log để biết có chạy hay không =====
echo [START %date% %time%] >> D:\HTML\a\sync_log.txt

rem --- đường dẫn Python & script ---
set "PYEXE=C:\Users\HOME\AppData\Local\Programs\Python\Python310\python.exe"
set "SCRIPT=C:\Users\HOME\Desktop\aaa\sync_faq.py"

rem --- chạy và ghi cả stdout + stderr vào log ---
"%PYEXE%" "%SCRIPT%"  >> C:\Users\HOME\Desktop\aaa\sync_log.txt 2>&1

echo [END   %date% %time%] >> C:\Users\HOME\Desktop\aaa\sync_log.txt
endlocal