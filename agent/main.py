from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from .config import load_environment, create_tts, create_session_components, get_voice_id
from .types.user_data import UserData
from .agents.my_agent import MyAgent
import json

def extract_user_info(ctx: JobContext) -> dict:
    # 본인(Agent)의 정보를 ctx.agent에서 추출
    agent = getattr(ctx, "agent", None)
    user_info = {
        "user_id": "unknown",
        "user_name": "사용자",
        "japanese_level": "beginner",
        "preferences": {}
    }
    if agent:
        user_info["user_id"] = getattr(agent, "identity", "unknown")
        user_info["user_name"] = getattr(agent, "name", "사용자")
        meta = getattr(agent, "metadata", None)
        if meta:
            try:
                meta_dict = json.loads(meta)
                user_info.update(meta_dict)
            except Exception:
                pass
    return user_info

async def entrypoint(ctx: JobContext):
    """애플리케이션의 진입점"""
    print(f"[entrypoint] MyAgent 활성화 (대화 + 피드백 통합)")
    await ctx.connect()
    
    # 사용자 정보 추출
    user_info = extract_user_info(ctx)
    print(f"[entrypoint] 사용자 정보: {user_info}")
    
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
            user_id=user_info["user_id"],
            user_name=user_info["user_name"],
            japanese_level=user_info.get("japanese_level", "beginner"),
            preferences=user_info.get("preferences", {}),
        ),
    )

    # MyAgent 생성 및 설정 (친구 역할 + 피드백 기능 통합)
    my_agent = MyAgent(tts=tts)
    session.userdata.agents["my"] = my_agent

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
            # agent_name="kanata_agent",
        ),
    )

if __name__ == "__main__":
    main() 