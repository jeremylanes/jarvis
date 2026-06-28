
if __name__ == "__main__":
    from speech2text2speech.main import SpeechToTextToSpeech
    stt2tts = SpeechToTextToSpeech()

    print('starting....')
    stt2tts.speak("Génère un script Bash rapide pour vérifier l'espace disque et m'alerter si un dossier dépasse 80%.")
    print('done')
