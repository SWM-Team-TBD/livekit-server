from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.plugins import openai, silero, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from .basic_agent import UserData
from .agents import MyAgent

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = MyAgent()

    session = AgentSession[UserData](
        stt=openai.STT(
            language="ko",
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),
        tts=elevenlabs.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        userdata=UserData(
            agents={
                "my": agent,
            },
            prev_agent=None,
            user_id="soma123",
        ),
    )

    await session.start(
        agent=agent,
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