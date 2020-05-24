import os
from gtts import gTTS
from io import BytesIO


def generate_greeting(username):

    mp3_fp = BytesIO()
    text = f"Hello, {username}!"
    tts = gTTS(text, lang="en")
    tts.write_to_fp(mp3_fp)

    return mp3_fp
