@echo off
set PYEXE=C:\Miniconda3\python.exe   
set WORKDIR=D:\HTML\a
cd /d %WORKDIR%
%PYEXE% sync_faq.py



echo === [%date% %time%] Bắt đầu đồng bộ... >> sync_log.txt
"%PYEXE%" sync_faq.py >> sync_log.txt 2>&1
echo === [%date% %time%] Kết thúc đồng bộ === >> sync_log.txt
exit
