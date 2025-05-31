from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="친절하게 대답하는 도우미가 되세요.",
        )
    
    async def on_enter(self):
        self.session.say("안녕! 반가워! 무엇을 도와줄까?")
    
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=openai.STT(
            language="ko",
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
    )

def main():
    """애플리케이션 메인 함수."""
    load_dotenv(dotenv_path=".env.local")

    # CLI 실행
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )


if __name__ == "__main__":
    main()