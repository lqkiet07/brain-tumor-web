@echo off
setlocal
cd /d "%~dp0"

set "TF_CPP_MIN_LOG_LEVEL=2"

if not exist ".venv\Scripts\python.exe" (
    echo CHUA CAI DAT MOI TRUONG. Hay chay setup.bat truoc.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run_demo.py
