@echo off
setlocal EnableExtensions

REM 1) Go to the project folder (where app.py lives)
cd /d "%~dp0"
echo [i] Working dir: %CD%

REM 2) Activate your venv
if not exist ".venv\Scripts\activate.bat" (
  echo [X] venv not found at .venv\Scripts\activate.bat
  echo     Create it:  python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

REM 3) Turn ON Local AI chat for this run
set "ENABLE_LOCAL_AI=true"

REM 4) Start Ollama if available (okay if missing)
where ollama >nul 2>nul
if errorlevel 1 (
  echo [!] Ollama not found on PATH. Chat UI will show a notice until you install/pull a model.
) else (
  start "" /B ollama serve
)

REM 5) Launch Streamlit
echo [i] Starting Streamlit...
streamlit run app.py

REM Keep window open if Streamlit exits
pause
