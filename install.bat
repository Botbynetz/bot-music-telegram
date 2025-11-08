@echo off
title Universal Music Bot - Auto Installer

echo 🎵 Universal Music Bot - Auto Installer
echo ========================================

:: Check Python installation
echo 📋 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo Download from: https://python.org/downloads
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python found: %PYTHON_VERSION%

:: Check pip
echo 📋 Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found! Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found

:: Install dependencies
echo 📦 Installing dependencies...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

:: Setup environment file
echo ⚙️ Setting up environment file...
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo ✅ Created .env file from template
    ) else (
        echo ❌ .env.example not found
        pause
        exit /b 1
    )
) else (
    echo ℹ️ .env file already exists
)

:: Get bot token
echo.
echo 🤖 TELEGRAM BOT SETUP
echo ====================
echo 1. Chat with @BotFather on Telegram
echo 2. Send: /newbot
echo 3. Follow the instructions
echo 4. Copy your bot token
echo.
set /p BOT_TOKEN="Enter your Telegram Bot Token: "

if not "%BOT_TOKEN%"=="" (
    :: Update .env file
    powershell -Command "(Get-Content .env) -replace 'your_telegram_bot_token_here', '%BOT_TOKEN%' | Set-Content .env"
    echo ✅ Bot token configured
) else (
    echo ⚠️ No token entered. Please edit .env file manually.
)

:: Create directories
echo 📁 Creating directories...
if not exist downloads mkdir downloads
if not exist logs mkdir logs
if not exist data mkdir data
echo ✅ Directories created

:: Final message
echo.
echo 🎉 INSTALLATION COMPLETE!
echo ========================
echo.
echo To start the bot, run:
echo   python clean_universal_music_bot.py
echo.
echo Or use the start script:
echo   start.bat
echo.
echo 📖 Read QUICK_START.md for more information
echo.
echo 🎵 Enjoy your Ultra-High Performance Music Bot!
echo.
pause