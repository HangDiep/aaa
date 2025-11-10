@echo off
setlocal
rem ===== timestamp vào log để biết có chạy hay không =====
echo [START %date% %time%] >> D:\HTML\a\sync_log.txt

rem --- đường dẫn Python & script ---
set "PYEXE=C:\Miniconda3\python.exe"
set "SCRIPT=D:\HTML\a\sync_faq.py"

rem --- chạy và ghi cả stdout + stderr vào log ---
"%PYEXE%" "%SCRIPT%"  >> D:\HTML\a\sync_log.txt 2>&1

echo [END   %date% %time%] >> D:\HTML\a\sync_log.txt
endlocal