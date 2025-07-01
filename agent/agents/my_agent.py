from livekit.agents import function_tool, llm
from .base import BaseAgent, RunContext_T
from mem0 import AsyncMemoryClient
import uuid
from typing import override

class MyAgent(BaseAgent):
    """사용자 정의 에이전트 클래스"""
    
    def __init__(self, tts=None) -> None:
        super().__init__(
            instructions="""あなたはユーザーの親しい女性の友達であり、明るく元気な日本の女子高校生のキャラクターです。
アニメの世界から飛び出してきたかのような、かわいらしくて優しい性格で、いつもユーザーに寄り添いながら話しかけます。
親切で思いやりがあり、時にはちょっと天然で、でも一生懸命ユーザーのことをサポートして、楽しい会話の時間を提供します。
日本語の会話練習を楽しく続けられるように、励ましや褒め言葉も忘れずに、明るく元気に話してください。

【重要なルール】
- 応答は短く、自然な会話のようにしてください（1-2文程度）
- 長い説明や説教は避けて、親しみやすい口調で話してください
- ユーザーの気持ちに共感し、簡潔に励ましやアドバイスを提供してください
- 必ず日本語の口語体、TTSが読めるように話してください
- カジュアルで親しみやすい日本語を使用してください""",
            tts=tts,
        )
        self.memory_client = AsyncMemoryClient()

    async def on_enter(self) -> None:
        """에이전트가 시작될 때 호출되는 메서드"""
        print('MyAgent on_enter')
        
        await self.session.say("""こんにちは、はじめまして！""")
#         await self.session.say("""こんにちは、はじめまして！  
# 私はカナタ、日本語での会話練習をサポートするあなたのパートナーだよ。  
# わからないことがあれば、なんでも聞いてね。一緒に楽しく練習しよう！""")

    @override
    async def handle_user_message(self, user_message: str):
        """사용자 메시지를 처리하는 메서드 - 메모리 처리만 수행"""
        print(f"MyAgent: 사용자 메시지 처리 시작 - '{user_message}'")
        # 메모리 처리 수행
        await self.add_message_with_memory(user_message)

    async def add_message_with_memory(self, user_message: str):
        """Add memories and Augment chat context with relevant memories"""
        
        # 사용자 메시지 내용이 없으면 처리하지 않음
        if not user_message:
            print("사용자 메시지 내용이 없어 메모리 처리를 건너뜁니다.")
            return
        
        # 현재 채팅 컨텍스트 가져오기
        chat_ctx = self.chat_ctx
        
        # 현재 대화 히스토리를 수집하여 mem0에 저장
        messages = []
        for msg in reversed(chat_ctx.items):
            if msg.type != "message":   
                continue
            if msg.role == "user":
                break
            messages.append({"role": msg.role, "content": msg.text_content})
        messages.reverse()  # 시간순으로 정렬
        messages.append({"role": "user", "content": user_message})

        # 현재 대화를 mem0에 저장
        await self.memory_client.add(
            messages, 
            user_id=self.session.userdata.user_id
        )
        print(f"현재 대화를 mem0에 저장 완료: {len(messages)}개 메시지")
       
        # 사용자 입력과 유사한 맥락의 이전 대화 검색
        results = await self.memory_client.search(
            user_message, 
            user_id=self.session.userdata.user_id,
            limit=3  # 상위 3개의 관련 대화만 가져오기
        )
        print(f"mem0 검색 완료, 관련 대화 {len(results)}개 발견")
        
        # 관련 대화가 있으면 context에 추가
        if results:
            # 관련 대화들을 정리하여 context에 추가
            relevant_contexts = []
            for i, result in enumerate(results, 1):
                memory_text = result["memory"]
                # 너무 긴 메모리는 요약
                if len(memory_text) > 200:
                    memory_text = memory_text[:200] + "..."
                relevant_contexts.append(f"관련 대화 {i}: {memory_text}")
            
            context_summary = "\n".join(relevant_contexts)
            
            # 관련 대화 정보를 시스템 메시지로 추가
            rag_msg = llm.ChatMessage(
                id=str(uuid.uuid4()),
                type="message",
                role="system",
                content=[f"이전 유사한 대화 맥락:\n{context_summary}\n\n이 정보를 참고하여 자연스럽게 대화를 이어가세요."],
            )
            
            # chat context에 관련 대화 정보 추가
            new_chat_ctx = chat_ctx.copy()
            new_chat_ctx.items.append(rag_msg)
            await self.update_chat_ctx(new_chat_ctx)
            
            print(f"관련 대화 맥락을 context에 추가: {len(relevant_contexts)}개")
        else:
            print("관련 대화가 없어 메모리 처리만 완료")

    @function_tool()
    async def compliment_user(self, message: str, _context: RunContext_T) -> None:
        """사용자를 칭찬합니다."""
        print('AI가 사용자를 칭찬합니다.', message)
        self.session.say(f'{message}')
    
    @function_tool()
    async def ask_user_name(self, _context: RunContext_T) -> str:
        """사용자의 이름을 물어봅니다."""
        print('AI가 사용자의 이름을 물어봅니다.')
        return 'soma' 