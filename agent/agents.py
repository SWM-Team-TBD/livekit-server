from livekit.agents import RunContext, function_tool
from .basic_agent import BaseAgent, UserData

RunContext_T = RunContext[UserData]

class MyAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""당신은 사용자의 여성 친구입니다. 일본 애니메이션의 여자 고등학생 느낌으로, 친절하고 활기차게 사용자와 이야기하세요""",
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
