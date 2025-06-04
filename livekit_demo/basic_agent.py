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

class BaseAgent(Agent):
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

    @override
    async def on_user_turn_completed(self, turn_ctx, new_message):
        await self.add_message_with_memory(turn_ctx, new_message)
   
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
       
    def on_enter(self) -> None:
        print('BaseAgent on_enter')