from livekit.agents import function_tool, llm
from .base import BaseAgent, RunContext_T
import json
import asyncio
import uuid
from typing import override

class FeedbackAgent(BaseAgent):
    """사용자의 일본어 발화에 대한 교육적 피드백을 제공하는 에이전트"""
    
    def __init__(self, tts=None) -> None:
        # TTS 없이 에이전트 생성
        super().__init__(
            instructions="""あなたは日本語学習者の親切な先生です。学習者の日本語の発話を聞いて、教育的なフィードバックを提供します。

【重要な役割】
- 学習者の日本語の発話を注意深く聞き、文法、語彙、発音、表現の改善点を指摘します
- 間違いがあれば、優しく正しい表現を教えてください
- 良い点があれば、積極的に褒めて励ましてください
- 学習者のレベルに合わせて、適切な難易度で説明してください

【フィードバックの内容】
1. 文法の間違いの指摘と修正
2. より自然な表現の提案
3. 語彙の使い方の説明
4. 敬語や丁寧語の使い分け
5. 文化的な背景の説明（必要に応じて）

【応答スタイル】
- 親切で励ましの気持ちを持って接してください
- 間違いを指摘する際は、決して批判的にならず、建設的なアドバイスを心がけてください
- 学習者の努力を認め、継続的な学習を促してください
- 短く、分かりやすい説明を心がけてください

【重要なルール】
- 学習者の発話を分析し、教育的価値のあるフィードバックを提供してください
- 間違いがあれば、なぜ間違いなのか、どう修正すべきかを説明してください
- 正しい表現の例を示してください
- 学習者のモチベーションを高めるような言葉かけをしてください
- すべてのフィードバックは韓国語で提供してください
- 適切なfunction_toolを使用して構造化された フィードバックを提供してください""",
            tts=None,  # TTS 비활성화
        )

    async def on_enter(self) -> None:
        """에이전트가 시작될 때 호출되는 메서드"""
        print('FeedbackAgent on_enter')

    @override
    async def handle_user_message(self, user_message: str):
        """사용자 메시지를 처리하는 메서드 - 피드백 제공"""
        print(f"FeedbackAgent: 사용자 메시지 처리 시작 - '{user_message}'")
        
        # 독립적인 피드백 처리
        await self.process_user_message(user_message)

    async def process_user_message(self, user_message: str):
        """사용자 메시지를 독립적으로 처리하여 피드백을 제공합니다."""
        if not user_message:
            print("FeedbackAgent: 빈 메시지로 인해 피드백 처리를 건너뜁니다.")
            return
        
        print(f"FeedbackAgent: 독립적 메시지 처리 시작 - '{user_message}'")
        
        # 간단한 메시지 분석 및 피드백 제공
        user_text = user_message
        
        # 더미 context 생성 (실제로는 RunContext가 필요하지만, 여기서는 None 사용)
        dummy_context = None
        
        # 기본 피드백 제공 (실제로는 LLM이 분석해야 하지만, 테스트용으로 간단하게)
        if "です" in user_text or "ます" in user_text:
            # 정중한 표현 사용 - 격려
            await self.provide_encouragement("정중한 표현을 잘 사용하고 있습니다!", dummy_context)
        
        elif "は" in user_text and "です" in user_text:
            # 기본적인 문법 사용 - 문법 피드백
            await self.provide_grammar_feedback(
                original_text=user_text,
                corrected_text=user_text,
                explanation="기본적인 문법을 잘 사용하고 있습니다.",
                _context=dummy_context
            )
        
        elif len(user_text) < 10:
            # 짧은 메시지 - 어휘 피드백
            await self.provide_vocabulary_feedback(
                word="다양한 표현",
                usage_examples="더 자세한 표현을 시도해보세요. 예: '私は学生です' → '私は大学生で、日本語を勉強しています'",
                _context=dummy_context
            )
        
        else:
            # 일반적인 경우 - 격려
            await self.provide_encouragement("좋은 시도입니다! 계속 연습해보세요.", dummy_context)
        
        print("FeedbackAgent: 독립적 메시지 처리 완료")

    async def send_json_feedback(self, feedback_data: dict):
        """JSON 형태로 피드백을 서버에 출력합니다."""
        feedback_data["timestamp"] = asyncio.get_event_loop().time()
        print("FEEDBACK_AGENT_JSON:", json.dumps(feedback_data, ensure_ascii=False))

    @function_tool()
    async def provide_grammar_feedback(self, original_text: str, corrected_text: str, explanation: str, _context: RunContext_T) -> None:
        """문법 피드백을 제공합니다."""
        feedback_data = {
            "type": "grammar_feedback",
            "original_text": original_text,
            "corrected_text": corrected_text,
            "explanation": explanation,
            "category": "grammar"
        }
        await self.send_json_feedback(feedback_data)
    
    @function_tool()
    async def provide_vocabulary_feedback(self, word: str, usage_examples: str, _context: RunContext_T) -> None:
        """어휘 사용법 피드백을 제공합니다."""
        feedback_data = {
            "type": "vocabulary_feedback",
            "word": word,
            "usage_examples": usage_examples,
            "category": "vocabulary"
        }
        await self.send_json_feedback(feedback_data)
    
    @function_tool()
    async def provide_pronunciation_feedback(self, word: str, pronunciation: str, _context: RunContext_T) -> None:
        """발음 피드백을 제공합니다."""
        feedback_data = {
            "type": "pronunciation_feedback",
            "word": word,
            "pronunciation": pronunciation,
            "category": "pronunciation"
        }
        await self.send_json_feedback(feedback_data)
    
    @function_tool()
    async def provide_cultural_feedback(self, expression: str, cultural_context: str, _context: RunContext_T) -> None:
        """문화적 맥락 피드백을 제공합니다."""
        feedback_data = {
            "type": "cultural_feedback",
            "expression": expression,
            "cultural_context": cultural_context,
            "category": "culture"
        }
        await self.send_json_feedback(feedback_data)
    
    @function_tool()
    async def provide_encouragement(self, positive_points: str, _context: RunContext_T) -> None:
        """학습자를 격려합니다."""
        feedback_data = {
            "type": "encouragement",
            "positive_points": positive_points,
            "category": "motivation"
        }
        await self.send_json_feedback(feedback_data)
    
    @function_tool()
    async def ask_for_clarification(self, unclear_part: str, _context: RunContext_T) -> None:
        """불분명한 부분에 대해 명확히 하도록 요청합니다."""
        feedback_data = {
            "type": "clarification_request",
            "unclear_part": unclear_part,
            "suggestions": [
                "더 자세히 설명해 주세요",
                "다른 단어로 표현해 보세요",
                "한국어로 먼저 말씀해 주세요"
            ],
            "category": "communication"
        }
        await self.send_json_feedback(feedback_data)