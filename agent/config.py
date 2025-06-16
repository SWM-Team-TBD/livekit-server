import os
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from livekit.plugins import openai, silero, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.elevenlabs.tts import VoiceSettings

def load_environment():
    """환경 변수를 로드합니다."""
    load_dotenv(dotenv_path=".env.local")
    return

def get_voice_id():
    """voice_id를 가져옵니다."""
    voice_id = os.getenv("ELEVEN_VOICE_ID")
    if not voice_id:
        raise ValueError("ELEVEN_VOICE_ID가 설정되지 않았습니다.")
    return voice_id

def create_tts(
    voice_id: str,
    stability: float = 0.5,      # 0.0 ~ 1.0 (낮을수록 더 감정적이고 변동성이 큼)
    similarity_boost: float = 0.75,  # 0.0 ~ 1.0 (목소리 유사도)
    style: float = 0.0,         # 0.0 ~ 1.0 (스타일 강도)
    speed: float = 1.0,         # 0.8 ~ 1.2 (말하기 속도)
    use_speaker_boost: bool = True  # 화자 부스트 사용 여부
):
    """
    TTS 객체를 생성합니다.
    
    Args:
        voice_id: 음성 ID
        stability: 안정성 (0.0 ~ 1.0)
            - 낮을수록 더 감정적이고 변동성이 큼
            - 높을수록 더 안정적이고 일관된 목소리
        similarity_boost: 목소리 유사도 (0.0 ~ 1.0)
            - 높을수록 원본 목소리와 더 유사
        style: 스타일 강도 (0.0 ~ 1.0)
            - 높을수록 더 강한 스타일 적용
        speed: 말하기 속도 (0.8 ~ 1.2)
            - 1.0이 기본 속도
            - 0.8은 20% 느리게
            - 1.2는 20% 빠르게
        use_speaker_boost: 화자 부스트 사용 여부
            - True: 화자 부스트 활성화 (더 선명한 목소리)
            - False: 화자 부스트 비활성화
    """
    # TTS 객체 생성 (기본 설정 사용)
    tts = elevenlabs.TTS(
        voice_id=voice_id,
        model="eleven_multilingual_v2"
    )
    
    print("Initial TTS _opts:", tts._opts)
    print("Initial TTS _opts.voice_settings:", tts._opts.voice_settings)
    
    # VoiceSettings 인스턴스 생성
    voice_settings = VoiceSettings(
        stability=float(stability),
        similarity_boost=float(similarity_boost),
        style=float(style),
        speed=float(speed),
        use_speaker_boost=bool(use_speaker_boost)
    )
    
    print("Created VoiceSettings:", voice_settings)
    
    # voice_id와 voice_settings를 한 번에 업데이트
    tts.update_options(
        voice_id=voice_id,
        voice_settings=voice_settings
    )
    
    print("Updated TTS _opts:", tts._opts)
    print("Updated TTS _opts.voice_settings:", tts._opts.voice_settings)
    return tts

def create_session_components(tts):
    """AgentSession에 필요한 컴포넌트들을 생성합니다."""
    return {
        "stt": openai.STT(language="ko"),
        "llm": openai.LLM(model="gpt-4o-mini"),
        "tts": tts,
        "vad": silero.VAD.load(),
        "turn_detection": MultilingualModel(),
    }