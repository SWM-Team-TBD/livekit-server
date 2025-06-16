from livekit.agents import RunContext, tts, NOT_GIVEN
from livekit.agents.voice import Agent
from ..types.user_data import UserData

RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    """기본 에이전트 클래스"""
    
    def __init__(
        self, 
        instructions: str,
        tts: tts.TTS | None = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=instructions,
            tts=tts,
        )
    
    def on_enter(self) -> None:
        """에이전트가 시작될 때 호출되는 메서드"""
        print('BaseAgent on_enter') 