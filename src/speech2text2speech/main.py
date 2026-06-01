import json
import queue
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model

# Robust path resolution
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model/vosk-model-small-fr-0.22"


class SpeechToText:
    pass


class TextToSpeech:
    pass


class SpeechToTextToSpeech(SpeechToText, TextToSpeech):
    pass


q = queue.Queue()


def callback(indata, frames, time, status):
    q.put(bytes(indata))


# Initialization using absolute path string
model = Model(str(MODEL_PATH))
rec = KaldiRecognizer(model, 16000)

text = ""

with sd.RawInputStream(
    samplerate=16000,
    blocksize=8000,
    dtype="int16",
    channels=1,
    callback=callback
):
    print("Listening...")
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            print(f">> {text}")
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
            print(f".. {partial}", end="\r")
