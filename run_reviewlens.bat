@echo off
title ReviewLens NLP Intelligence Platform
echo ============================================================
echo   ReviewLens: Smart Product Review Intelligence Platform
echo ============================================================
echo.

cd /d "%~dp0"

:: Check for virtual environment or create one
if not exist "venv" (
    echo [1/3] Creating Python virtual environment (venv)...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat 2>nul || (
    echo Note: Using system Python environment...
)

echo [2/3] Verifying dependencies and NLTK corpora...
python setup_nltk.py >nul 2>&1

echo [3/3] Launching ReviewLens Streamlit Application on port 8501...
start "" http://localhost:8501
python -m streamlit run app.py --server.port=8501 --server.headless=false

pause
