@echo off
for /l %%i in (1,1,5) do (
  for /f "tokens=5" %%p in ('netstat -ano ^| findstr LISTENING ^| findstr :8000') do (
    taskkill /PID %%p /T /F >nul 2>nul
  )
  timeout /t 1 /nobreak >nul
)
echo ==== LISTENING :8000 ====
netstat -ano | findstr LISTENING | findstr :8000
