from dotenv import load_dotenv
from livekit.agents import RunContext, JobContext, WorkerOptions, cli, function_tool, tts, NOT_GIVEN, llm
from livekit.agents.voice import Agent, AgentSession
from enum import Enum
from livekit.plugins import openai, silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pydantic import BaseModel
from typing import Dict, Optional, override
from dataclasses import dataclass
from mem0 import AsyncMemoryClient
from basic_agent import BaseAgent
import uuid

@dataclass
class UserData:
    agents: Dict[str, Agent]
    prev_agent: Optional[Agent]
    user_id: str


RunContext_T = RunContext[UserData]

class MyAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="친절하게 대답하는 도우미가 되세요.",
        )
    
    async def on_enter(self) -> None:
        print('MyAgent on_enter')
        self.session.say("안녕! 반가워! 무엇을 도와줄까?")

    @function_tool()
    async def compliment_user(self, message: str, _context: RunContext_T) -> None:
        """
        사용자를 칭찬합니다. 꼭 필요한 경우에만 사용하세요.

        Args:
            message: 칭찬 메시지
        """
        print('AI가 사용자를 칭찬합니다.', message)
        self.session.say(f'{message}')
    
    @function_tool()
    async def ask_user_name(self, _context: RunContext_T) -> str:
        print('AI가 사용자의 이름을 물어봅니다.')
        return 'Soma'
    
    @function_tool()
    async def transfer_to_sales(self, context: RunContext_T) -> None:
        """
        일본어 회화 판매 에이전트로 전환합니다.
        """
        print('AI가 사용자를 판매 에이전트로 전환합니다.')
        return await self._transfer_to_agent("sales", context)  

class SalesAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            너는 음성 AI야. 사용자는 너와 STT/TTS를 통해서 이야기하고 있어..
            **이런 문자는 쓰지마. 만약 써야한다고 하면 문자의 발음을 그대로 써.
            숫자도 발음을 써줘.
            100,000을 쓰는 대신 십만이라고 해.

            사용자는 일본어를 배우러 왔습니다. 어떤 단계인지 물어보고, 우리의 상품을 소개하세요.
            - 상품
              - 일본어 기초 수업: 100,000원 / 10회
              - 일본어 고급 수업: 200,000원 / 10회
              - 일본어 회화 수업: 300,000원 / 10회
              - 일본어 회화 수업: 400,000원 / 10회
            """,
            tts=openai.TTS(voice="shimmer"),
        )
    
    async def on_enter(self) -> None:
        print('SalesAgent on_enter')
        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()
        print('prev_agent', userdata.prev_agent)
        if userdata.prev_agent:
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [
                item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        print('SalesAgent on_enter', chat_ctx.items)
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(instructions="사용자에게 인사를 해. 일본어 배우러 왔냐고 물어봐. 가격얘긴 하지 말고")


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