import io
import json
import logging
import queue
import time
import wave
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from piper.voice import PiperVoice
from vosk import KaldiRecognizer, Model

logger = logging.getLogger(__name__)

# Robust path resolution
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model/vosk-model-small-fr-0.22"


class SpeechToText:
    def __init__(self):
        self.q = queue.Queue()
        self.model = Model(str(MODEL_PATH))
        self.rec = KaldiRecognizer(self.model, 16000)
        self.accumulated_text = []
        self.current_paragraph = ""
        self.last_speech_time = time.time()
        self.is_listening_msg_visible = False
        self.json_path = BASE_DIR / "transcription.json"

    def callback(self, indata, frames, time_info, status):
        self.q.put(bytes(indata))

    def save_transcription(self, final=False):
        if self.current_paragraph.strip():
            self.accumulated_text.append(self.current_paragraph.strip())
            self.current_paragraph = ""

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.accumulated_text, f, ensure_ascii=False, indent=4)

    def show_listening(self):
        print("\rListening...        ", end="\rListening...", flush=True)
        self.is_listening_msg_visible = True

    def hide_listening(self):
        if self.is_listening_msg_visible:
            print("\r" + " " * 20 + "\r", end="", flush=True)
            self.is_listening_msg_visible = False

    def start_listening(self):
        self.show_listening()
        self.last_speech_time = time.time()

        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self.callback
            ):
                while True:
                    try:
                        data = self.q.get(timeout=0.1)
                        # logger.debug(f'DATA - {data}')

                        if self.rec.AcceptWaveform(data):
                            logger.debug('WAVEFORM ACCEPTED')

                            result = json.loads(self.rec.Result())
                            text = result.get("text", "")
                            if text:
                                logger.debug(f'TEXT FOUND - {text}')

                                self.hide_listening()
                                self.current_paragraph += text + " "
                                print(text + " ", end="", flush=True)
                                self.last_speech_time = time.time()
                        else:
                            logger.debug('WAVEFORM NOT ACCEPTED')

                            partial = json.loads(self.rec.PartialResult()).get("partial", "")
                            logger.debug(f'PARTIAL - {partial}')

                            if partial:
                                logger.debug(f'PARTIAL TEXT FOUND - {partial}')

                                self.hide_listening()
                                self.last_speech_time = time.time()

                    except queue.Empty:
                        # logger.debug(f'QUEUE EMPTY')
                        pass

                    # Check for 4 seconds of silence
                    if self.current_paragraph and (time.time() - self.last_speech_time > 3.0):
                        logger.debug('SILENCE > 3s')

                        self.save_transcription()
                        print()  # New line for the new paragraph

                        self.show_listening()
                        self.last_speech_time = time.time()

        except KeyboardInterrupt:
            print("\nStopping listening. Saving in progress...")
            self.save_transcription(final=True)


class TextToSpeech:
    def __init__(self, model_path: str = "model/piper/fr_FR-siwis-medium.onnx"):
        self.voice = PiperVoice.load(model_path)

    def speak(self, text: str):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        sd.play(data, sr)
        sd.wait()


class SpeechToTextToSpeech(SpeechToText, TextToSpeech):
    pass


if __name__ == "__main__":
    import argparse

    from vosk import SetLogLevel

    parser = argparse.ArgumentParser(description="Speech to Text")
    parser.add_argument("--prod", action="store_true", help="Run in production mode (disables logs)")
    args = parser.parse_args()

    # Python logs configuration
    log_level = logging.WARNING if args.prod else logging.DEBUG
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Vosk C++ logs configuration
    if args.prod:
        SetLogLevel(-1)

    stt = SpeechToTextToSpeech()
    # stt.start_listening()

    tts = TextToSpeech()
    tts.speak(
        "Excellente observation, monsieur, si vous voulez changer de planète, il faudra améliorer les exzosystèmes."
    )
