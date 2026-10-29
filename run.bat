@echo off
rem AgroVision AI - start the web app at http://127.0.0.1:8000
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
start "" http://127.0.0.1:8000
%PY% backend\app.py
