@echo off
REM ============================================================
REM Netra AI — Launch Script
REM Opens backend (uvicorn) and ngrok tunnel in separate windows.
REM EDIT the two paths/values below before first use.
REM ============================================================

REM --- EDIT THIS: path to the folder containing inference_api.py ---
set BACKEND_DIR=C:\Users\hp\Downloads\Diabetic-Retinopathy-Internal\Back End

REM --- EDIT THIS: your ngrok static domain (no https://) ---
set NGROK_DOMAIN=outgrow-genetics-stylist.ngrok-free.dev

echo Launching Windows Terminal with backend + ngrok tabs...

wt -w 0 new-tab -d "%BACKEND_DIR%" --title "Netra AI - Backend" cmd /k "uvicorn inference_api:app --host 0.0.0.0 --port 8000" ; new-tab -d "%BACKEND_DIR%" --title "Netra AI - ngrok" cmd /k "set NGROK_DOMAIN=%NGROK_DOMAIN% && wait_and_start_ngrok.bat"

echo.
echo Two tabs launched in Windows Terminal: Backend and ngrok.
echo Frontend URL: check your Vercel deployment (VITE_API_URL already points to %NGROK_DOMAIN%)
echo.
echo If nothing opened, run the wt command above directly in a terminal to see the error.
pause
