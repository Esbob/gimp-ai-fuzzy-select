@echo off
rem Removes AI Fuzzy Select (see README.md): double-click this file. From a command prompt,
rem uninstaller options can follow, e.g.  uninstall.bat --keep-checkpoints
setlocal
cd /d "%~dp0"
set PY=
py -3 -c "" >nul 2>nul && set "PY=py -3"
rem Python 3.14 installed for this user by install.bat while py isn't on the PATH yet
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PY="%LOCALAPPDATA%\Programs\Python\Python314\python.exe""
)
rem Not the Microsoft Store's python (it keeps the files it creates under AppData apart)
if not defined PY (
    python -c "import sys; sys.exit('WindowsApps' in sys.base_prefix)" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo The uninstaller needs Python 3: install it from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
%PY% uninstall_ai_select.py %*
echo.
pause
