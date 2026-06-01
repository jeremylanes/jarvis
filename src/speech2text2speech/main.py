import json
import queue
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model

# Robust path resolution
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model/vosk-model-small-fr-0.22"


class SpeechToText:
    def __init__(self):
        self.q = queue.Queue()
        self.model = Model(str(MODEL_PATH))
        self.rec = KaldiRecognizer(self.model, 16000)
        self.text = ""

    def callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def start_listening(self):
        print("Listening...")
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.callback
        ):
            while True:
                data = self.q.get()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    self.text = result.get("text", "")
                    print(f">> {self.text}")
                else:
                    partial = json.loads(self.rec.PartialResult()).get("partial", "")
                    print(f".. {partial}", end="\r")


class TextToSpeech:
    pass


class SpeechToTextToSpeech(SpeechToText, TextToSpeech):
    pass


if __name__ == "__main__":
    stt = SpeechToText()
    stt.start_listening()
