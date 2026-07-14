from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools import current_time

llm = ChatOllama(
    model="rafw007/qwen35-claude-coder:9b",
    # model='qwen2.5-coder:7b',
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[current_time],
    system_prompt=(
        "You are a senior software engineering assistant. "
        "Always use available tools when they are required to answer correctly. "
        "Be concise, accurate, and never invent facts."
    ),
)

stream = agent.stream_events(
    {
        "messages": [
            {
                "role": "user",
                "content": "il es quel heure bro."
            }
        ]
    },
    version='v3'
)


for message in stream.messages:
    for delta in message.text:
        print(delta, end='', flush=True)


# if __name__ == "__main__":
#     from speech2text2speech.main import SpeechToTextToSpeech
#     stt2tts = SpeechToTextToSpeech()
#
#     print('starting....')
#     stt2tts.speak("Génère un script Bash rapide pour vérifier l'espace disque et m'alerter si un dossier dépasse 80%.")  # noqa
#     print('done')
