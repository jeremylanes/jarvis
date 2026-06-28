
if __name__ == "__main__":
    from speech2text2speech.main import SpeechToTextToSpeech
    stt2tts = SpeechToTextToSpeech()

    print('starting....')
    stt2tts.speak("Hello, world!")
    print('done')
