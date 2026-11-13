@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONPYCACHEPREFIX=%~dp0.runtime\cache\pycache"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 with PowerShell to create the isolated environment first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m emotion_assistant
if errorlevel 1 pause
endlocal
