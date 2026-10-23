from gtts import gTTS
import os

class CommentaryTTS:
    """
    Converts a generated football commentary line into speech using gTTS.
    """

    def __init__(self, lang='en', slow=False, save_path="commentary.mp3"):
        """
        lang: language code for gTTS ('en' for English)
        slow: whether to speak slowly
        save_path: path to save the audio file
        """
        self.lang = lang
        self.slow = slow
        self.save_path = save_path

    def speak(self, commentary_line):
        if not commentary_line:
            print("No commentary line to speak.")
            return

        try:
            tts = gTTS(text=commentary_line, lang=self.lang, slow=self.slow)
            tts.save(self.save_path)
            print(f"Audio saved to {self.save_path}")
            
            # Play audio (works on most OS)
            if os.name == 'nt':  # Windows
                os.system(f'start {self.save_path}')
            else:  # Mac / Linux
                os.system(f'afplay {self.save_path}' if os.name == 'posix' else f'xdg-open {self.save_path}')

        except Exception as e:
            print(f"TTS error: {e}")


# ---------------- Example Usage ----------------

if __name__ == "__main__":
    commentary_line = "Messi steps up to the penalty, runs up, and with precision sends the ball top-right past Neuer into the net!"
    
    tts_generator = CommentaryTTS(save_path="penalty_commentary.mp3")
    tts_generator.speak(commentary_line)
