@echo off
title Telegram Music Bot

echo ================================
echo    TELEGRAM MUSIC BOT
echo ================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan!
    echo Silakan install Python terlebih dahulu.
    pause
    exit /b 1
)

:: Check if .env file exists
if not exist ".env" (
    echo [WARNING] File .env tidak ditemukan!
    echo Menyalin .env.example ke .env...
    copy ".env.example" ".env" >nul
    echo.
    echo [INFO] Silakan edit file .env dan masukkan Telegram Bot Token Anda.
    echo Dapatkan token dari @BotFather di Telegram.
    echo Kemudian jalankan script ini lagi.
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Membuat virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat virtual environment!
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo [INFO] Mengaktifkan virtual environment...
call venv\Scripts\activate.bat

:: Install/update dependencies
echo [INFO] Menginstall dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Gagal menginstall dependencies!
    pause
    exit /b 1
)

:: Check for FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg tidak ditemukan!
    echo Bot mungkin tidak bisa memutar audio.
    echo Silakan install FFmpeg dan tambahkan ke PATH.
    echo.
)

echo [INFO] Memulai bot...
echo ================================
python bot.py

if errorlevel 1 (
    echo.
    echo [ERROR] Bot berhenti dengan error!
)

echo.
pause