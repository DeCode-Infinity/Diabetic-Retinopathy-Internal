@echo off
REM Waits until the backend's /health endpoint responds, then starts ngrok.
REM Called by launch_netra_ai.bat — not meant to be run standalone unless
REM NGROK_DOMAIN is set as an environment variable first.

echo Waiting for backend to be ready...

:waitloop
curl -s -o nul -w "%%{http_code}" http://localhost:8000/health > "%TEMP%\netra_health.txt" 2>nul
set /p HEALTH_CODE=<"%TEMP%\netra_health.txt"
if not "%HEALTH_CODE%"=="200" (
    timeout /t 2 /nobreak >nul
    goto waitloop
)

echo Backend is up. Starting ngrok...
ngrok http --domain=%NGROK_DOMAIN% 8000
