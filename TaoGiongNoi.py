from gtts import gTTS
import pygame
import threading
import os

# Khởi tạo pygame để phát âm thanh
pygame.mixer.init()

# Tạo thư mục cache âm thanh nếu chưa có
os.makedirs("voices2", exist_ok=True)

# ✅ Tạo sẵn file âm thanh cho các chữ cái
def generate_voice_cache():
    voice_map = {
        'D': "Chữ D",
        'H': "Chữ H",
        'U': "Chữ U",
        'V': "Chữ V",
    }

    for letter, text in voice_map.items():
        filename = f"voices2/{letter}.mp3"
        if not os.path.exists(filename):
            tts = gTTS(text=text, lang='vi')
            tts.save(filename)
def speak_cached(letter):
    filename = f"voices2/{letter}.mp3"
    def play():
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    threading.Thread(target=play).start()

# Gọi cache khi khởi động
generate_voice_cache()