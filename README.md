# 🎵 Universal Music Bot

Bot musik Telegram yang powerful dengan performa ultra-tinggi dan fitur lengkap untuk pencarian dan download musik.

## ✨ Fitur Utama

- 🎶 **Memutar Musik**: Mencari dan memutar musik dari YouTube
- 📋 **Queue System**: Antrian musik dengan berbagai kontrol
- 🔥 **Update Viral**: Update musik viral harian otomatis
- 🎛️ **Kontrol Playback**: Play, pause, skip, stop, volume control
- 🔄 **Repeat & Loop**: Mode repeat dan loop queue
- 🎯 **Search Multiple**: Pencarian dengan hasil multiple
- 📊 **Queue Management**: Melihat, mengatur, dan shuffle queue
- 🎙️ **Voice Chat**: Dukungan voice chat Telegram (opsional)

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 atau lebih baru
- FFmpeg (untuk audio processing)
- Telegram Bot Token

### 2. Installation

```bash
# Clone atau download project
cd "bot musik"

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
copy .env.example .env
# Edit .env file dan masukkan Telegram Bot Token Anda
```

### 3. Setup Telegram Bot

1. Chat dengan [@BotFather](https://t.me/BotFather) di Telegram
2. Ketik `/newbot` dan ikuti instruksi
3. Salin Bot Token yang diberikan
4. Masukkan token ke file `.env`:
   ```
   TELEGRAM_TOKEN=your_bot_token_here
   ```
5. (Opsional) Untuk voice chat, berikan bot permission admin di grup

### 4. Install FFmpeg

#### Windows:
1. Download FFmpeg dari [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract dan tambahkan ke PATH
3. Atau install via chocolatey: `choco install ffmpeg`

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install ffmpeg
```

### 5. Run Bot

```bash
python bot.py
```

## 🎮 Commands

### Basic Music Commands
- `/start` - Mulai menggunakan bot
- `/help` - Bantuan penggunaan
- `/play <song>` - Play musik dari YouTube
- `/pause` - Pause musik
- `/resume` - Resume musik yang di-pause
- `/stop` - Stop musik dan clear queue
- `/skip` - Skip ke lagu selanjutnya

### Queue Management
- `/queue` - Lihat queue saat ini
- `/shuffle` - Shuffle queue
- `/clear` - Clear semua queue
- `/remove <index>` - Remove lagu dari queue

### Advanced Controls
- `/volume <0-100>` - Set volume
- `/repeat` - Toggle repeat mode
- `/loop` - Toggle loop queue mode
- `/nowplaying` - Info lagu yang sedang playing
- `/search <song>` - Cari dengan pilihan multiple

### Viral Music
- `/viral` - Lihat lagu viral saat ini
- `/trending <genre>` - Lagu trending by genre (pop, hip-hop, rock, electronic)

### Voice Chat (Grup)
- `/join` - Info tentang voice chat
- `/leave` - Keluar dari voice chat

## 📁 Project Structure

```
bot musik/
├── bot.py                          # Main bot file untuk Telegram
├── requirements.txt                # Python dependencies
├── .env.example                   # Environment variables template
├── README.md                      # This file
├── start.bat                      # Script startup Windows
├── start.sh                       # Script startup Linux/Mac
├── config/
│   └── settings.py               # Configuration settings
├── src/
│   ├── __init__.py
│   ├── telegram_music_player.py  # Telegram music player logic
│   ├── music_search.py          # YouTube search functionality
│   └── viral_updates.py         # Viral music updates
└── utils/
    ├── __init__.py
    └── music_commands.py         # Command utilities (legacy)
```

## ⚙️ Configuration

Edit file `.env` untuk konfigurasi:

```env
# Wajib
TELEGRAM_TOKEN=your_bot_token

# Opsional
MAX_QUEUE_SIZE=50
DEFAULT_VOLUME=0.5
VIRAL_UPDATE_ENABLED=true
ENABLE_VOICE_CHAT=false
```

## 🎯 Advanced Features

### API Integration (Opsional)

Untuk fitur yang lebih canggih, Anda bisa menambahkan API keys:

1. **Spotify API**: Chart yang lebih akurat
2. **YouTube Data API**: Pencarian yang lebih baik
3. **Last.fm API**: Data musik tambahan

Tambahkan di file `.env`:
```env
SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
YOUTUBE_API_KEY=your_key
LASTFM_API_KEY=your_key
```

### Daily Viral Updates

Bot akan otomatis mengirim update musik viral setiap hari ke chat yang aktif. Waktu dapat diatur di `.env`:

```env
VIRAL_UPDATE_HOUR=9  # 9 AM
VIRAL_UPDATE_ENABLED=true
```

### Voice Chat Integration

Untuk menggunakan fitur voice chat real di Telegram:

1. Install library tambahan:
   ```bash
   pip install py-tgcalls
   ```

2. Set di `.env`:
   ```env
   ENABLE_VOICE_CHAT=true
   ```

3. Berikan bot permission admin di grup
4. Mulai voice chat di grup
5. Invite bot ke voice chat

## 🛠️ Troubleshooting

### Common Issues

1. **"Invalid token"**
   - Periksa Telegram Bot Token
   - Pastikan token dari @BotFather

2. **"Bot tidak merespon"**
   - Pastikan bot sudah di-start dengan `/start`
   - Cek koneksi internet

3. **"Cannot download music"**
   - Periksa instalasi FFmpeg
   - Pastikan yt-dlp up to date

4. **"Voice chat not working"**
   - Install py-tgcalls: `pip install py-tgcalls`
   - Bot harus admin di grup
   - Set ENABLE_VOICE_CHAT=true

### Performance Tips

- Gunakan SSD untuk performa yang lebih baik
- RAM minimal 1GB untuk bot
- Koneksi internet stabil untuk streaming
- Batasi MAX_QUEUE_SIZE untuk menghemat memory

## 📱 Telegram Features

### Inline Keyboards
Bot menggunakan inline keyboard untuk kontrol yang mudah:
- ⏸️ Pause/Resume
- ⏭️ Skip
- 📋 Queue
- 🔀 Shuffle
- 🗑️ Clear

### Private vs Group Chat
- **Private Chat**: Semua fitur tersedia
- **Group Chat**: Fitur voice chat dan kontrol musik
- **Channel**: Tidak mendukung interaksi

### Message Formatting
Bot menggunakan Markdown untuk format pesan yang menarik dengan emoji dan style yang konsisten.

## 📝 Contributing

1. Fork repository
2. Buat branch untuk fitur baru
3. Commit changes
4. Push ke branch
5. Buat Pull Request

## 📄 License

Project ini menggunakan MIT License. Lihat file LICENSE untuk detail.

## 🆘 Support

Jika ada masalah atau pertanyaan:

1. Cek dokumentasi di atas
2. Lihat issues di GitHub
3. Buat issue baru jika diperlukan

## 🔮 Future Features

- [ ] Playlist support
- [ ] Lyrics display
- [ ] Music quiz games
- [ ] User music statistics
- [ ] Custom playlists per user
- [ ] Voice command recognition
- [ ] Multi-language support

---

**Made with ❤️ for Discord music lovers**