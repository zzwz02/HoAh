@echo off
rem Creates .venv and installs dependencies. HoAh-lite.vbs runs this automatically on first start.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
if errorlevel 1 (
  echo Could not create the Python environment. Install Python 3.12+ from https://www.python.org/downloads/
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)
