@echo off
setlocal
cd /d "%~dp0"
echo This is a layout preview only. Prediction APIs are unavailable.
echo Open http://127.0.0.1:8766 in your browser.
echo Press Ctrl+C to stop the server.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m http.server 8766 --bind 127.0.0.1 --directory static
) else (
    set "PYTHON_CMD="
    for %%V in (3.12 3.11 3.10) do (
        if not defined PYTHON_CMD (
            py -%%V --version >nul 2>nul
            if not errorlevel 1 set "PYTHON_CMD=py -%%V"
        )
    )
    if not defined PYTHON_CMD (
        python -c "import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))" >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
    if not defined PYTHON_CMD (
        echo A working Python 3.10, 3.11, or 3.12 installation was not found.
        echo Install Python first, then run this preview again.
        pause
        exit /b 1
    )
    %PYTHON_CMD% -m http.server 8766 --bind 127.0.0.1 --directory static
)
pause
