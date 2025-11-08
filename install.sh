#!/bin/bash

# 🎵 Universal Music Bot - Auto Installer
echo "🎵 Universal Music Bot - Auto Installer"
echo "========================================"

# Check Python installation
echo "📋 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found! Please install Python 3.8+ first."
        echo "Download from: https://python.org/downloads"
        exit 1
    else
        PYTHON_CMD=python
    fi
else
    PYTHON_CMD=python3
fi

echo "✅ Python found: $($PYTHON_CMD --version)"

# Check pip
echo "📋 Checking pip..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "❌ pip not found! Please install pip first."
    exit 1
fi

echo "✅ pip found"

# Install dependencies
echo "📦 Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Setup environment file
echo "⚙️ Setting up environment file..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file from template"
    else
        echo "❌ .env.example not found"
        exit 1
    fi
else
    echo "ℹ️ .env file already exists"
fi

# Get bot token
echo ""
echo "🤖 TELEGRAM BOT SETUP"
echo "===================="
echo "1. Chat with @BotFather on Telegram"
echo "2. Send: /newbot"
echo "3. Follow the instructions"
echo "4. Copy your bot token"
echo ""
read -p "Enter your Telegram Bot Token: " BOT_TOKEN

if [ ! -z "$BOT_TOKEN" ]; then
    # Update .env file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/your_telegram_bot_token_here/$BOT_TOKEN/g" .env
    else
        # Linux
        sed -i "s/your_telegram_bot_token_here/$BOT_TOKEN/g" .env
    fi
    echo "✅ Bot token configured"
else
    echo "⚠️ No token entered. Please edit .env file manually."
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p downloads logs data
echo "✅ Directories created"

# Final message
echo ""
echo "🎉 INSTALLATION COMPLETE!"
echo "========================"
echo ""
echo "To start the bot, run:"
echo "  $PYTHON_CMD clean_universal_music_bot.py"
echo ""
echo "Or use the start script:"
echo "  ./start.sh"
echo ""
echo "📖 Read QUICK_START.md for more information"
echo ""
echo "🎵 Enjoy your Ultra-High Performance Music Bot!"