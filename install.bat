@echo off
rem Installs AI Fuzzy Select (see README.md): double-click this file. From a command prompt,
rem installer options can follow, e.g.  install.bat --model large --model small
setlocal
cd /d "%~dp0"
set PY=
rem The py launcher comes with Python from python.org; prefer the newest supported version.
for %%v in (3.14 3.13 3.12 3.11 3.10) do (
    if not defined PY py -%%v -c "" >nul 2>nul && set "PY=py -%%v"
)
if not defined PY (
    python -c "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo AI Fuzzy Select needs Python 3.14 ^(3.10 to 3.13 also work, 3.15 does not yet^).
    echo Install it from https://www.python.org/downloads/release/python-3149/
    echo and then run this again.
    echo.
    pause
    exit /b 1
)
echo Using %PY%
%PY% install_ai_select.py %*
echo.
pause
