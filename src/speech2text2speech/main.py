"""
Speech-to-Text and Text-to-Speech pipeline module.

This module provides three dataclass-based components for building a local,
offline voice assistant pipeline:

- :class:`SpeechToText`  – captures microphone input, transcribes it with Vosk
  and persists paragraphs to a JSON file.
- :class:`TextToSpeech`  – synthesises speech from text with Piper and plays it
  back through the default audio output.
- :class:`SpeechToTextToSpeech` – a convenience class that composes both
  capabilities through multiple inheritance.

All speech recognition is performed offline using a Vosk model stored under
``model/vosk-model-small-fr-0.22/`` relative to this file.  TTS synthesis
relies on a Piper ONNX model stored under
``model/piper/fr_FR-upmc-medium.onnx``.

**Usage example**::

    # Text-to-speech only
    from speech2text2speech.main import TextToSpeech

    tts = TextToSpeech()
    tts.speak("Hello, world!")

    # Full speech-to-text loop (blocks until Ctrl-C)
    from speech2text2speech.main import SpeechToText

    stt = SpeechToText()
    stt.start_listening()

    # Combined pipeline
    from speech2text2speech.main import SpeechToTextToSpeech

    agent = SpeechToTextToSpeech()
    agent.start_listening()
"""

import io
import json
import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import sounddevice as sd
import soundfile as sf
from piper.voice import PiperVoice
from vosk import KaldiRecognizer, Model

logger = logging.getLogger(__name__)

# Robust path resolution
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model/vosk-model-small-fr-0.22"


@dataclass
class SpeechToText:
    """Offline speech-to-text transcriber backed by Vosk.

    Captures raw audio from the default microphone at 16 kHz, feeds it to a
    :class:`vosk.KaldiRecognizer` and accumulates recognised words into
    paragraphs.  A paragraph is flushed to disk (JSON) after 3 seconds of
    silence.

    :param q: Thread-safe queue used to pass raw audio chunks from the
        sounddevice callback to the recognition loop.
    :type q: queue.Queue
    :param model: Loaded Vosk acoustic model.
    :type model: vosk.Model
    :param rec: Kaldi recogniser initialised at 16 000 Hz.
    :type rec: vosk.KaldiRecognizer
    :param accumulated_text: List of completed paragraphs collected during the
        current session.
    :type accumulated_text: list
    :param current_paragraph: Buffer for the paragraph being built from the
        most recent speech burst.
    :type current_paragraph: str
    :param last_speech_time: Unix timestamp of the last detected speech event,
        used to detect silence gaps.
    :type last_speech_time: float
    :param is_listening_msg_visible: Tracks whether the ``Listening...``
        indicator is currently printed on stdout so it can be erased cleanly.
    :type is_listening_msg_visible: bool
    :param json_path: Filesystem path where transcription paragraphs are
        persisted as a JSON array.
    :type json_path: pathlib.Path

    **Usage example**::

        stt = SpeechToText()
        # Blocks until KeyboardInterrupt (Ctrl-C)
        stt.start_listening()
        # Paragraphs are saved to BASE_DIR / "transcription.json"
    """

    q: queue.Queue = field(default_factory=queue.Queue)
    model: Model = field(default_factory=lambda: Model(str(MODEL_PATH)))
    rec: KaldiRecognizer = field(init=False)
    accumulated_text: list = field(default_factory=list)
    current_paragraph: str = ""
    last_speech_time: float = field(default_factory=time.time)
    is_listening_msg_visible: bool = False
    json_path: Path = field(default_factory=lambda: BASE_DIR / "transcription.json")

    def __post_init__(self):
        """Initialise the Kaldi recogniser after the dataclass fields are set.

        :class:`vosk.KaldiRecognizer` requires a fully loaded
        :class:`vosk.Model` instance, which is only available after
        ``__init__`` has run, hence the use of ``__post_init__``.
        """
        self.rec = KaldiRecognizer(self.model, 16000)

    def callback(self, indata, frames, time_info, status):
        """sounddevice stream callback — enqueues raw audio bytes.

        This method is called by sounddevice on a background thread each time a
        new audio block is available.  It simply converts the incoming buffer to
        bytes and pushes it onto :attr:`q` for consumption by the recognition
        loop in :meth:`start_listening`.

        :param indata: Raw audio samples provided by sounddevice.
        :param frames: Number of frames in the block (unused here).
        :param time_info: Timing information from PortAudio (unused here).
        :param status: Stream status flags (unused here).

        **Usage example**::

            # Registered automatically by start_listening(); not called directly.
            stream = sd.RawInputStream(callback=stt.callback, ...)
        """
        self.q.put(bytes(indata))

    def save_transcription(self, final=False):
        """Flush the current paragraph buffer and persist all paragraphs to disk.

        If :attr:`current_paragraph` contains non-whitespace text it is appended
        to :attr:`accumulated_text` and the buffer is cleared.  The full list is
        then serialised to :attr:`json_path` as a UTF-8 JSON array.

        :param final: Semantic flag indicating whether this is the end-of-session
            save.  Currently unused in the logic but reserved for future
            extension (e.g. adding a ``"final": true`` marker to the JSON).
        :type final: bool

        **Usage example**::

            stt = SpeechToText()
            stt.current_paragraph = "Hello world"
            stt.save_transcription()
            # BASE_DIR / "transcription.json" now contains ["Hello world"]
        """
        if self.current_paragraph.strip():
            self.accumulated_text.append(self.current_paragraph.strip())
            self.current_paragraph = ""

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.accumulated_text, f, ensure_ascii=False, indent=4)

    def show_listening(self):
        """Print the ``Listening...`` indicator on stdout.

        Uses a carriage-return trick so the indicator occupies the current line
        without scrolling the terminal.  Sets :attr:`is_listening_msg_visible`
        to ``True`` so :meth:`hide_listening` knows the line must be erased.

        **Usage example**::

            stt = SpeechToText()
            stt.show_listening()   # prints "Listening..." on the current line
        """
        print("\rListening...        ", end="\rListening...", flush=True)
        self.is_listening_msg_visible = True

    def hide_listening(self):
        """Erase the ``Listening...`` indicator from stdout.

        Only acts when :attr:`is_listening_msg_visible` is ``True`` to avoid
        unnecessary writes.  Overwrites the current line with spaces and resets
        the flag.

        **Usage example**::

            stt = SpeechToText()
            stt.show_listening()
            stt.hide_listening()   # erases the "Listening..." line
        """
        if self.is_listening_msg_visible:
            print("\r" + " " * 20 + "\r", end="", flush=True)
            self.is_listening_msg_visible = False

    def start_listening(self):
        """Start the blocking microphone capture and transcription loop.

        Opens a 16 kHz mono :class:`sounddevice.RawInputStream` and processes
        audio chunks in a tight loop:

        1. If the recogniser accepts a complete waveform, the recognised text is
           appended to :attr:`current_paragraph` and printed inline.
        2. If only a partial result is available the silence timer is reset so
           the paragraph is not prematurely flushed.
        3. After 3 consecutive seconds without any speech the current paragraph
           is saved via :meth:`save_transcription` and the indicator is shown
           again.

        The loop exits cleanly on ``KeyboardInterrupt`` (Ctrl-C), performing a
        final save before returning.

        :raises KeyboardInterrupt: Caught internally; triggers a final
            :meth:`save_transcription` call before the method returns.

        **Usage example**::

            stt = SpeechToText()
            stt.start_listening()  # blocks; press Ctrl-C to stop
        """
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


@dataclass
class TextToSpeech:
    """Offline text-to-speech synthesiser backed by Piper.

    Loads a Piper ONNX voice model and exposes a single :meth:`speak` method
    that synthesises audio into an in-memory WAV buffer, then plays it through
    the default audio output via sounddevice.  Words are printed to stdout in
    sync with the audio playback.

    :cvar MODEL_PATH: Relative path (from this file's directory) to the Piper
        ONNX voice model.
    :type MODEL_PATH: str
    :param voice: Loaded Piper voice used for synthesis.
    :type voice: piper.voice.PiperVoice

    **Usage example**::

        tts = TextToSpeech()
        tts.speak("Hello, I am your voice assistant.")
    """

    MODEL_PATH: ClassVar[str] = "model/piper/fr_FR-upmc-medium.onnx"
    voice: PiperVoice = field(default_factory=lambda: PiperVoice.load(TextToSpeech.MODEL_PATH))

    def speak(self, text: str):
        """Synthesise *text* and play the result through the audio output.

        The method:

        1. Synthesises the full utterance into an in-memory WAV buffer using
           :attr:`voice`.
        2. Reads the WAV data with *soundfile* to obtain a float32 NumPy array.
        3. Starts a daemon thread that prints each word to stdout at a pace
           proportional to the audio duration.
        4. Plays the audio synchronously with ``sd.play`` / ``sd.wait``.

        :param text: The text to synthesise and speak.  Must not be empty.
        :type text: str

        **Usage example**::

            tts = TextToSpeech()
            tts.speak("The quick brown fox jumps over the lazy dog.")
            # Words are printed progressively while audio plays.
        """
        logger.debug(f'VOICE TYPE - {type(self.voice)}')

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")

        words = text.split()
        duration = len(data) / sr
        delay = duration / len(words)

        def stream_words():
            for word in words:
                print(word, end=" ", flush=True)
                time.sleep(delay)
            print()

        threading.Thread(target=stream_words, daemon=True).start()
        sd.play(data, sr)
        sd.wait()


class SpeechToTextToSpeech(SpeechToText, TextToSpeech):
    """Combined speech-to-text and text-to-speech pipeline.

    Inherits :class:`SpeechToText` for microphone capture / transcription and
    :class:`TextToSpeech` for voice synthesis, making all capabilities
    available on a single object.

    Python's MRO ensures that :meth:`~SpeechToText.__post_init__` from
    :class:`SpeechToText` is called correctly when the dataclass is
    instantiated.

    **Usage example**::

        agent = SpeechToTextToSpeech()

        # Transcribe microphone input
        agent.start_listening()

        # Speak a response
        agent.speak("Transcription complete.")
    """

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
        "Monsieur, j'ai analysé 14 372 scénarios possibles concernant votre décision actuelle. Dans 92,4 % des cas, votre plan aboutit à un succès remarquable. Dans les 7,6 % restants, il entraîne une explosion, un incident diplomatique international ou une conversation particulièrement désagréable avec Mademoiselle Bamporiki. Je me permets donc de recommander une approche légèrement plus prudente. Cela étant dit, l'expérience m'a démontré qu'ignorer mes recommandations constitue l'une de vos compétences les plus constantes. J'ai donc préparé les protocoles d'urgence, alerté les systèmes de secours, renforcé les défenses et commandé du café. Si vous tenez absolument à défier les lois de la physique, de la logique et du bon sens simultanément, je serai naturellement à vos côtés pour documenter l'événement et tenter d'en limiter les conséquences. Après tout, Monsieur, mon rôle n'est pas de vous empêcher de faire l'impossible, mais de m'assurer que vous surviviez suffisamment longtemps pour vous en attribuer le mérite." # noqa
    )
