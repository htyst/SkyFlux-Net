@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Local environment is missing. Run the dependency installer first.
    pause
    exit /b 1
)
echo Starting the local prediction service. Please wait for the ready message.
echo The browser will open http://127.0.0.1:8765 after startup.
echo Loading PyTorch for the first time may take a minute.
echo Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" -u server.py --open-browser
if errorlevel 1 (
    echo.
    echo Startup failed. See the error above.
    echo If dependencies are missing, run the dependency installer in this folder.
    pause
    exit /b 1
)
pause
