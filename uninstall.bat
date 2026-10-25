@echo off
rem Removes AI Fuzzy Select (see README.md): double-click this file. From a command prompt,
rem uninstaller options can follow, e.g.  uninstall.bat --keep-checkpoints
setlocal
cd /d "%~dp0"
set PY=
py -3 -c "" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "" >nul 2>nul && set "PY=python"
if not defined PY (
    echo The uninstaller needs Python 3: install it from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
%PY% uninstall_ai_select.py %*
echo.
pause
