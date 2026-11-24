@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m photo_resizer
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m photo_resizer
) else (
    uv run photo-resizer
)
endlocal
