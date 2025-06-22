from livekit.agents import RunContext, tts, NOT_GIVEN, cli, llm
from livekit.agents.voice import Agent
from ..types.user_data import UserData
import uuid
from mem0 import AsyncMemoryClient

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
        self.memory_client = AsyncMemoryClient()
    
    def on_enter(self) -> None:
        """에이전트가 시작될 때 호출되는 메서드"""
        print('BaseAgent on_enter') 

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