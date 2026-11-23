@echo off
title ReviewLens NLP Intelligence Platform
echo ============================================================
echo  Starting ReviewLens: Smart Product Review Intelligence
echo ============================================================
echo.
cd /d "%~dp0"
echo Opening browser at http://localhost:8501 ...
start "" http://localhost:8501
python -m streamlit run app.py --server.port=8501
pause
