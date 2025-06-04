from dataclasses import dataclass
from dotenv import load_dotenv
from collections.abc import AsyncIterable
from livekit.agents import JobContext, RunContext, WorkerOptions, cli, llm, ModelSettings, FunctionTool
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero
from livekit.agents.llm import function_tool
from typing import Dict, Optional, override, Annotated, TypedDict, Callable, cast
from pydantic import Field
from pydantic_core import from_json
from enum import Enum
from mem0 import AsyncMemoryClient
import uuid

@dataclass
class UserData:
    agents: Dict[str, Agent]
    prev_agent: Optional[Agent]
    user_id: str


RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    def __init__(
        self, 
        instructions: str,
    ) -> None:
        super().__init__(
            instructions=instructions,
        )
        self.memory_client = AsyncMemoryClient()
    
    @override
    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        await self.add_message_with_memory(turn_ctx, new_message)
        return await super().on_user_turn_completed(turn_ctx, new_message)
    
    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Agent:
        """다른 에이전트로 전환"""
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent
        return next_agent
    
    def _truncate_chat_ctx(
            self,
            items: list,
            keep_last_n_messages: int = 6,
            keep_system_message: bool = False,
            keep_function_call: bool = False,
    ) -> list:

        def _valid_item(item) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in ["function_call", "function_call_output"]:
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break

        new_items = new_items[::-1]

        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items
    
    async def add_message_with_memory(self, chat_ctx: llm.ChatContext, user_msg: llm.ChatMessage):
        """Add memories and Augment chat context with relevant memories"""
        # <reasoning>
        # 메시지 히스토리를 가져와서 이전 사용자 메시지까지의 모든 메시지를 수집
        # </reasoning>
        # messages = []
        # for msg in reversed(chat_ctx.items):
        #     if msg.type != "message":   
        #         continue
        #     if msg.role == "user":
        #         break

        #     messages.append({"role": msg.role, "content": msg.text_content})
        # messages.reverse()  # 시간순으로 정렬
        # messages.append({"role": "user", "content": user_msg.text_content})

        # await self.memory_client.add(
        #     messages, 
        #     user_id=self.session.userdata.user_id
        # )
       
        # Search for relevant memories
        results = await self.memory_client.search(
            user_msg.text_content, 
            user_id=self.session.userdata.user_id,
        )
        print(f"memory_client.search 완료, results: {results}")
        
        # Augment context with retrieved memories
        if results:
            memories = ','.join([result["memory"] for result in results])
            
            rag_msg = llm.ChatMessage(
                id=str(uuid.uuid4()),
                type="message",
                role="assistant",
                content=[f"Relevant Memory: {memories}\n"],
            )
            
            # Modify chat context with retrieved memories
            chat_ctx.items.append(rag_msg)
            chat_ctx.items.append(user_msg)
            await self.update_chat_ctx(chat_ctx)

    
    def on_enter(self) -> None:
        print('BaseAgent on_enter')

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
    
    @function_tool()
    async def transfer_to_face_expression_agent(self, context: RunContext_T) -> None:
        """
        얼굴 표정 에이전트로 전환합니다.
        """
        print('AI가 사용자를 얼굴 표정 에이전트로 전환합니다.')
        return await self._transfer_to_agent("face_expression", context)
    
class SalesAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""사용자는 일본어를 배우러 왔습니다. 어떤 단계인지 물어보고, 우리의 상품을 소개하세요.
            - 상품
              - 일본어 기초 수업: 100,000원 / 10회
              - 일본어 고급 수업: 200,000원 / 10회
              - 일본어 회화 수업: 300,000원 / 10회
              - 일본어 회화 수업: 400,000원 / 10회
            """,
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


class FaceExpression(Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"

class ResponseFace(TypedDict):
    face_expression: Annotated[
        FaceExpression,
        Field(..., description="Face expression"),
    ]
    response: str


async def process_structured_output(
    text: AsyncIterable[str],
    callback: Optional[Callable[[ResponseFace], None]] = None,
) -> AsyncIterable[str]:
    last_response = ""
    acc_text = ""
    async for chunk in text:
        acc_text += chunk
        try:
            resp: ResponseEmotion = from_json(acc_text, allow_partial="trailing-strings")
        except ValueError:
            continue

        if callback:
            callback(resp)

        if not resp.get("response"):
            continue

        new_delta = resp["response"][len(last_response) :]
        if new_delta:
            yield new_delta
        last_response = resp["response"]

class FaceExpressionAgent(BaseAgent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions)
        
    async def on_enter(self) -> None:
        print('FaceExpressionAgent on_enter')
        self.session.say("안녕! 무슨 얼굴을 하고 싶어?")
    
    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[FunctionTool], model_settings: ModelSettings
    ):
        # not all LLMs support structured output, so we need to cast to the specific LLM type
        llm = cast(openai.LLM, self.session.llm)
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        async with llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            response_format=ResponseFace,
        ) as stream:
            async for chunk in stream:
                yield chunk
    
    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        expression_sent = False

        def face_expression_changed(resp: ResponseFace):
            nonlocal expression_sent
            if resp.get("face_expression") and resp.get("response") and not expression_sent:
                expression_sent = True

                print(f"Sending face expression to client: {resp['face_expression']}")
                # todo: send face expression to client

        # process_structured_output strips the TTS instructions and only synthesizes the verbal part
        # of the LLM output
        return Agent.default.tts_node(
            self, process_structured_output(text, callback=face_expression_changed), model_settings
        )

    async def transcription_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        # transcription_node needs to return what the agent would say, minus the TTS instructions
        return Agent.default.transcription_node(
            self, process_structured_output(text), model_settings
        )
    
    
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    my_agent = MyAgent()

    userdata = UserData(
        agents={
            "sales": SalesAgent(),
            "my": my_agent,
            "face_expression": FaceExpressionAgent(
                instructions="""
                당신은 사용자가 바라보고 있는 캐릭터의 표정을 변경할 수 있는 AI Agent입니다.
                사용자가 원하거나 적절한 표정을 보여주세요.
                캐릭터는 당신의 말을 TTS로 읽어주고 있습니다.
                """
            ),
        },
        prev_agent=None,
        user_id="soma12345",
    )

    session = AgentSession[UserData](
        stt=openai.STT(
            language="ko",
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
        userdata=userdata,
    )

    async def my_shutdown_hook():
        print(f"my_shutdown_hook 호출")
        memory_client = AsyncMemoryClient()
        chat_messages = []
        for item in session.history.items:
            if item.type != "message":
                continue
            chat_messages.append({"role": item.role, "content": item.text_content})
        await memory_client.add(
            chat_messages, 
            user_id=userdata.user_id
        )
        print(f"memory_client.add 완료, chat_messages: {len(chat_messages)}")
        
    ctx.add_shutdown_callback(my_shutdown_hook)

    await session.start(
        agent=my_agent,
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