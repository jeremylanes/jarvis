import warnings

from langchain_core._api.beta_decorator import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)


from langchain.agents import create_agent  # noqa
from langchain_ollama import ChatOllama  # noqa

from tools import current_time  # noqa

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


while True:
    user_input = input("🧑‍💬 You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        break

    stream = agent.stream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        version='v3'
    )

    for message in stream.messages:
        print("🤖 JARVIS: ", end="", flush=True)
        for delta in message.text:
            print(delta, end="", flush=True)
        print()


# if __name__ == "__main__":
#     from speech2text2speech.main import SpeechToTextToSpeech
#     stt2tts = SpeechToTextToSpeech()
#
#     print('starting....')
#     stt2tts.speak("Génère un script Bash rapide pour vérifier l'espace disque et m'alerter si un dossier dépasse 80%.")  # noqa
#     print('done')
