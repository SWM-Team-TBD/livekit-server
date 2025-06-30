from livekit.agents import RunContext, tts, llm
from livekit.agents.voice import Agent
from ..types.user_data import UserData
from typing import override

RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    """기본 에이전트 클래스"""
    
    def __init__(
        self, 
        instructions: str,
        tts: tts.TTS | None = None,
    ) -> None:
        super().__init__(
            instructions=instructions,
            tts=tts,
        )
    
    def on_enter(self) -> None:
        """에이전트가 시작될 때 호출되는 메서드"""
        print('BaseAgent on_enter')

    @override
    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """사용자가 말을 마쳤을 때 호출되는 메서드 - 기본 메모리 처리를 수행합니다"""
        print(f"BaseAgent: on_user_turn_completed 호출됨 - '{new_message.text_content}'")
        
        # 사용자 메시지인지 확인 (role이 "user"이고 type이 "message"인 경우만)
        if new_message.role == "user" and new_message.type == "message" and new_message.text_content:
            print(f"BaseAgent: 사용자 메시지 확인됨 - '{new_message.text_content}'")
            # 자식 클래스에서 구현할 메서드 호출
            await self.handle_user_message(new_message.text_content)
        else:
            print(f"BaseAgent: 시스템 메시지 또는 기타 메시지로 처리 건너뜀: role={new_message.role}, type={new_message.type}")
        
        # 부모 클래스의 메서드 호출
        await super().on_user_turn_completed(turn_ctx, new_message)

    async def handle_user_message(self, user_message: str):
        """사용자 메시지를 처리하는 메서드 - 자식 클래스에서 오버라이드"""
        print(f"BaseAgent: 기본 사용자 메시지 처리 - '{user_message}'")
        pass 