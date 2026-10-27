@echo off
rem Installs AI Fuzzy Select (see README.md): double-click this file. From a command prompt,
rem installer options can follow, e.g.  install.bat --model large --model small
setlocal
cd /d "%~dp0"
rem Without a suitable Python, this downloads Python from python.org and installs it for the
rem user (asking first); the download is checked against python.org's SHA-256.
set "PY_VERSION=3.14.10"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe"
set "PY_SHA256=92480239264744fe0386a1bd8275786fefa126e3129e56a68e22b52c73329965"
call :find_python
if not defined PY call :get_python
if not defined PY (
    echo.
    pause
    exit /b 1
)
echo Using %PY%
%PY% install_ai_select.py %*
echo.
pause
exit /b

:find_python
set PY=
rem The py launcher comes with Python from python.org; prefer the newest supported version.
for %%v in (3.14 3.13 3.12 3.11 3.10) do (
    if not defined PY py -%%v -c "" >nul 2>nul && set "PY=py -%%v"
)
rem Python 3.14 installed for this user (by :get_python, say) while py isn't on the PATH yet
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PY="%LOCALAPPDATA%\Programs\Python\Python314\python.exe""
)
rem Or python on the PATH, but not the Microsoft Store's: Windows keeps the files it creates
rem under AppData in a private folder, where GIMP doesn't see them
if not defined PY (
    python -c "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14) or 'WindowsApps' in sys.base_prefix)" >nul 2>nul && set "PY=python"
)
exit /b

:get_python
echo AI Fuzzy Select needs Python 3.14 (3.10 to 3.13 also work), from python.org.
echo It can download Python %PY_VERSION% and install it for you: about 32 MB, for your
echo user only, no administrator rights needed.
echo.
choice /c YN /m "Download and install Python %PY_VERSION% now"
if errorlevel 2 goto :manual_python
set "PY_SETUP=%TEMP%\python-%PY_VERSION%-amd64.exe"
echo Downloading %PY_URL%
curl.exe -L --fail --silent --show-error -o "%PY_SETUP%" "%PY_URL%"
if errorlevel 1 (
    echo The download failed. Is the computer connected to the internet?
    goto :manual_python
)
certutil -hashfile "%PY_SETUP%" SHA256 | findstr /i /x "%PY_SHA256%" >nul
if errorlevel 1 (
    del "%PY_SETUP%"
    echo The download is not the expected file: its SHA-256 does not match. Not installed.
    goto :manual_python
)
echo Installing Python %PY_VERSION% for your user...
"%PY_SETUP%" /quiet InstallAllUsers=0 Include_launcher=1 InstallLauncherAllUsers=0 PrependPath=0 AssociateFiles=0 Include_test=0 Shortcuts=0
set "RESULT=%errorlevel%"
del "%PY_SETUP%"
if not "%RESULT%"=="0" (
    echo The Python installer failed with exit code %RESULT%.
    goto :manual_python
)
call :find_python
if not defined PY goto :manual_python
echo Python %PY_VERSION% is installed.
echo.
exit /b

:manual_python
echo.
echo Install Python 3.14 from https://www.python.org/downloads/ yourself: the Windows
echo installer, 64-bit. Not Python 3.15, and not the one from the Microsoft Store.
echo Then run this again.
exit /b
