@echo off
title Setup Telegram Music Bot

echo ========================================
echo     SETUP TELEGRAM MUSIC BOT
echo ========================================
echo.

echo [INFO] Langkah-langkah mendapatkan Telegram Bot Token:
echo.
echo 1. Buka Telegram dan cari @BotFather
echo 2. Ketik /start untuk memulai
echo 3. Ketik /newbot untuk membuat bot baru
echo 4. Masukkan nama bot (contoh: My Music Bot)
echo 5. Masukkan username bot (harus unik, contoh: my_music_bot)
echo 6. @BotFather akan memberikan token
echo 7. Copy token tersebut
echo.

echo [INFO] Setelah mendapat token:
echo 1. Buka file .env di folder ini
echo 2. Ganti 'your_telegram_bot_token_here' dengan token yang didapat
echo 3. Save file .env
echo 4. Jalankan start.bat
echo.

set /p token="Masukkan Telegram Bot Token (atau Enter untuk skip): "

if "%token%"=="" (
    echo [INFO] Token tidak dimasukkan. Silakan edit file .env manual.
    echo.
) else (
    echo [INFO] Mengupdate file .env dengan token...
    
    :: Backup original .env
    if exist .env (
        copy .env .env.backup >nul 2>&1
    )
    
    :: Update .env file with token
    powershell -Command "(Get-Content .env) -replace 'your_telegram_bot_token_here', '%token%' | Set-Content .env"
    
    echo [SUCCESS] Token berhasil disimpan ke .env!
    echo.
)

echo [INFO] Testing setup...
call "D:/bot musik/.venv/Scripts/python.exe" test_setup.py

echo.
echo ========================================
echo [INFO] Setup selesai!
echo.
echo Untuk menjalankan bot:
echo 1. start.bat (Windows)
echo 2. atau: python bot.py
echo ========================================
pause