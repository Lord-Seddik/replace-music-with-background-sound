@echo off
rem Double-click to open the Music remover window. Works from any folder.
set "EXE=%~dp0.venv\Scripts\music-remover-gui.exe"
if not exist "%EXE%" (
    echo Not found: %EXE%
    echo Install the project first: "%~dp0.venv\Scripts\pip.exe" install -e "%~dp0."
    pause
    exit /b 1
)
start "" /D "%~dp0" "%EXE%"
