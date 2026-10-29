@echo off
setlocal
cd /d "%~dp0"
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
    echo Install 64-bit Python 3.10-3.12, then run this installer again.
    pause
    exit /b 1
)
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo Could not create virtual environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Dependencies installed. Run the site launcher in this folder.
pause
exit /b 0
:fail
echo Installation failed. Check network access and Python version.
pause
exit /b 1
