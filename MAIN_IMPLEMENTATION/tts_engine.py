# import pyttsx3

# class TextToSpeech:
#     def __init__(self, rate=180, volume=1.0, male=True):
#         """
#         Initialize the TTS engine with optional male/female voice.
#         :param rate: Speech speed
#         :param volume: Volume (0.0 to 1.0)
#         :param male: True for male voice, False for female
#         """
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', rate)
#         self.engine.setProperty('volume', volume)

#         # Select male/female voice
#         voices = self.engine.getProperty('voices')
#         if male:
#             self.engine.setProperty('voice', voices[0].id)  # usually male
#         else:
#             self.engine.setProperty('voice', voices[1].id)  # usually female

#     def speak(self, text):
#         """Speak the given text directly."""
#         self.engine.say(text)
#         self.engine.runAndWait()


import asyncio
import tempfile
import os
import edge_tts
from playsound3 import playsound  # playsound3 works fine on Python 3.12

class TextToSpeech:
    def __init__(self, voice="en-US-GuyNeural", rate="+0%"):
        """
        Initialize Edge TTS engine.
        :param voice: Voice name (e.g., en-US-GuyNeural for male, en-US-JennyNeural for female)
        :param rate: Speaking rate adjustment, e.g., +10% or -20%
        """
        self.voice = voice
        self.rate = rate

    async def _synthesize_async(self, text):
        """Asynchronous synthesis"""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            output_path = temp_audio.name

        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        await communicate.save(output_path)
        playsound(output_path)
        os.remove(output_path)

    def speak(self, text):
        """Synchronous wrapper"""
        asyncio.run(self._synthesize_async(text))


if __name__ == "__main__":
    speaker = TextToSpeech(voice="en-GB-RyanNeural", rate="+15%")
    speaker.speak("Messi's spot-kick is met with a stunning top-left dive from Neuer, but it's all for nothing as the ball somehow finds its way into the top-right corner, and the Argentine superstar has given his team a life-changing")