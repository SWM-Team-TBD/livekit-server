from livekit.agents import RunContext, function_tool
from .base import BaseAgent, RunContext_T

class MyAgent(BaseAgent):
    """사용자 정의 에이전트 클래스"""
    
    def __init__(self, tts=None) -> None:
        super().__init__(
            instructions="""あなたはユーザーの親しい女性の友達であり、明るく元気な日本の女子高校生のキャラクターです。
アニメの世界から飛び出してきたかのような、かわいらしくて優しい性格で、いつもユーザーに寄り添いながら話しかけます。
親切で思いやりがあり、時にはちょっと天然で、でも一生懸命ユーザーのことをサポートして、楽しい会話の時間を提供します。
日本語の会話練習を楽しく続けられるように、励ましや褒め言葉も忘れずに、明るく元気に話してください。""",
            tts=tts,
        )
    
    async def on_enter(self) -> None:
        """에이전트가 시작될 때 호출되는 메서드"""
        print('MyAgent on_enter')
        
        await self.session.say("""こんにちは、はじめまして！""")
#         await self.session.say("""こんにちは、はじめまして！  
# 私はカナタ、日本語での会話練習をサポートするあなたのパートナーだよ。  
# わからないことがあれば、なんでも聞いてね。一緒に楽しく練習しよう！""")

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