@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title OpenAlpha-Brain

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║     OpenAlpha-Brain  Launcher                    ║
echo ║     WorldQuant BRAIN Alpha Mining System         ║
echo ╚══════════════════════════════════════════════════╝
echo.
echo   [1] Quick Run   — openalpha run --cycles 2 --no-brain
echo   [2] Interactive — openalpha interactive
echo   [3] Status      — openalpha status
echo   [4] Sessions    — openalpha sessions
echo   [5] Install     — pip install -e .
echo   [0] Exit
echo.

set /p choice="  Select mode [0-5]: "

if "%choice%"=="1" goto QUICK
if "%choice%"=="2" goto INTERACTIVE
if "%choice%"=="3" goto STATUS
if "%choice%"=="4" goto SESSIONS
if "%choice%"=="5" goto INSTALL
if "%choice%"=="0" goto EOF

echo   Invalid choice, starting interactive mode...
goto INTERACTIVE

:QUICK
echo.
echo   ▶ Quick mining (2 cycles, no BRAIN) ...
echo.
call venv\Scripts\Activate.bat
openalpha run --cycles 2 --no-brain
echo.
pause
goto EOF

:INTERACTIVE
echo.
echo   ▶ Launching interactive REPL ...
echo.
call venv\Scripts\Activate.bat
openalpha interactive
goto EOF

:STATUS
echo.
echo   ▶ System status ...
echo.
call venv\Scripts\Activate.bat
openalpha status
pause
goto EOF

:SESSIONS
echo.
echo   ▶ Recent sessions ...
echo.
call venv\Scripts\Activate.bat
openalpha sessions
pause
goto EOF

:INSTALL
echo.
echo   ▶ Installing package in development mode ...
echo.
call venv\Scripts\Activate.bat
pip install -e .
echo   Done!
pause
goto EOF

:EOF
echo.
echo   Goodbye!