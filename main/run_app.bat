@echo off
REM ==========================================================
REM  MoS Analyzer One-Click Universal Launcher for Windows
REM ==========================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==========================================================
echo  MoS Analyzer -- Solid Mechanics Studio ^& Learning Lab
echo ==========================================================

where pixi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PIXI_CMD=pixi"
) else (
    if exist "%USERPROFILE%\.pixi\bin\pixi.exe" (
        set "PIXI_CMD=%USERPROFILE%\.pixi\bin\pixi.exe"
    ) else if exist "%LOCALAPPDATA%\pixi\bin\pixi.exe" (
        set "PIXI_CMD=%LOCALAPPDATA%\pixi\bin\pixi.exe"
    ) else (
        echo [INFO] Pixi package manager not found. Installing standalone portable pixi...
        powershell -ExecutionPolicy ByPass -Command "iwr -useb https://pixi.sh/install.ps1 | iex"
        if exist "%USERPROFILE%\.pixi\bin\pixi.exe" (
            set "PIXI_CMD=%USERPROFILE%\.pixi\bin\pixi.exe"
        ) else (
            set "PIXI_CMD=%LOCALAPPDATA%\pixi\bin\pixi.exe"
        )
    )
)

echo [INFO] Resolving dependencies and pre-compiled native libraries...
"%PIXI_CMD%" install

echo.
echo ==========================================================
echo  Starting MoS Studio at http://localhost:8080
echo  Press Ctrl+C in this terminal window to stop the server.
echo ==========================================================
echo.

start "" "http://localhost:8080"
"%PIXI_CMD%" run app

pause
