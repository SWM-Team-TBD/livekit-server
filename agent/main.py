from dotenv import load_dotenv
from livekit.agents import RunContext, JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from basic_agent import BaseAgent, UserData
from agents import MyAgent, SalesAgent

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = MyAgent()
    sales_agent = SalesAgent()

    session = AgentSession[UserData](
        stt=openai.STT(
            language="ko",
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        userdata=UserData(
            agents={
                "my": agent,
                "sales": sales_agent,
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