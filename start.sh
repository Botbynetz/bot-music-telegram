#!/bin/bash

# Telegram Music Bot - Linux/Mac Startup Script

echo "================================"
echo "    TELEGRAM MUSIC BOT"
echo "================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 tidak ditemukan!"
    echo "Silakan install Python3 terlebih dahulu."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "[WARNING] File .env tidak ditemukan!"
    echo "Menyalin .env.example ke .env..."
    cp ".env.example" ".env"
    echo
    echo "[INFO] Silakan edit file .env dan masukkan Telegram Bot Token Anda."
    echo "Dapatkan token dari @BotFather di Telegram."
    echo "Kemudian jalankan script ini lagi."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Membuat virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Gagal membuat virtual environment!"
        exit 1
    fi
fi

# Activate virtual environment
echo "[INFO] Mengaktifkan virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "[INFO] Menginstall dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Gagal menginstall dependencies!"
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARNING] FFmpeg tidak ditemukan!"
    echo "Bot mungkin tidak bisa memutar audio."
    echo "Install dengan: sudo apt install ffmpeg (Ubuntu/Debian)"
    echo "atau: brew install ffmpeg (macOS)"
    echo
fi

echo "[INFO] Memulai bot..."
echo "================================"
python bot.py

if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Bot berhenti dengan error!"
fi

echo
read -p "Tekan Enter untuk keluar..."