from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from .config import load_environment, create_tts, create_session_components, get_voice_id
from .types.user_data import UserData
from .agents.my_agent import MyAgent
from .agents.feedback_agent import FeedbackAgent
import asyncio

async def entrypoint(ctx: JobContext):
    """애플리케이션의 진입점"""
    await ctx.connect()
    
    # 환경 변수 로드 및 TTS 생성
    voice_id = get_voice_id()
    print("Using voice_id:", voice_id)  # 실제 사용되는 voice_id 확인
    
    tts = create_tts(voice_id, stability=1.0, similarity_boost=0.8, speed=1.0, style=0.2, use_speaker_boost=True)
    
    # 세션 컴포넌트 생성
    components = create_session_components(tts)
    
    # AgentSession 생성
    session = AgentSession[UserData](
        **components,
        userdata=UserData(
            agents={},
            prev_agent=None,
            user_id="soma123",
        ),
    )

    # MyAgent 생성 및 설정 (친구 역할 - TTS 사용)
    my_agent = MyAgent(tts=tts)
    session.userdata.agents["my"] = my_agent
    
    # FeedbackAgent 생성 및 설정 (선생님 역할 - TTS 없음, JSON 출력)
    feedback_agent = FeedbackAgent(tts=None)
    session.userdata.agents["feedback"] = feedback_agent
    
    # 기본 에이전트로 MyAgent 시작 (TTS 응답)
    await session.start(
        agent=my_agent,
        room=ctx.room,
    )

def main():
    """애플리케이션 메인 함수"""
    load_environment()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="kanata_agent",
        ),
    )

if __name__ == "__main__":
    main() 