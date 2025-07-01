from livekit.agents import function_tool, llm
from .base import BaseAgent, RunContext_T
import json
import asyncio
import uuid
from typing import override

class FeedbackAgent(BaseAgent):
    """
    사용자의 일본어 발화에 대한 교육적 피드백을 제공하는 에이전트
    - 독립적으로 동작하며, 사용자 메시지를 직접 받아 피드백만 제공
    - TTS 없이 JSON 형태로 피드백 출력
    """
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
- 適切なfunction_toolを使用して構造化されたフィードバックを提供してください

【分析手順】
1. まず文法の正確性をチェックしてください
2. 語彙の適切性と豊富さを評価してください
3. 敬語や丁寧語の使い分けを確認してください
4. 文化的な背景やニュアンスを説明してください
5. 改善点と良い点をバランスよく指摘してください
6. 必ずprovide_analysis_toolを呼び出して詳細な分析を提供してください

【分析のポイント】
- 初級者: 基本的な文法と語彙の正確性を重視
- 中級者: 自然な表現と敬語の使い分けを重視
- 上級者: 文化的なニュアンスと高度な表現を重視""",
            tts=None,  # TTS 비활성화
        )

    async def on_enter(self) -> None:
        print('FeedbackAgent on_enter')

    @override
    async def handle_user_message(self, user_message: str):
        """사용자 메시지를 처리하는 메서드 - 독립적으로 피드백만 제공"""
        print(f"FeedbackAgent: 사용자 메시지 처리 시작 - '{user_message}'")
        await self.process_user_message(user_message)

    async def process_user_message(self, user_message: str):
        """사용자 메시지를 독립적으로 처리하여 피드백을 제공합니다."""
        if not user_message:
            print("FeedbackAgent: 빈 메시지로 인해 피드백 처리를 건너뜁니다.")
            return
        
        print(f"FeedbackAgent: 독립적 메시지 처리 시작 - '{user_message}'")
        dummy_context = None
        await self.provide_analysis_tool(user_message, dummy_context)
        print("FeedbackAgent: 독립적 메시지 처리 완료")

    async def send_json_feedback(self, feedback_data: dict):
        """JSON 형태로 피드백을 서버에 출력합니다."""
        feedback_data["timestamp"] = asyncio.get_event_loop().time()
        print("FEEDBACK_AGENT_JSON:", json.dumps(feedback_data, ensure_ascii=False))

    @function_tool()
    async def provide_analysis_tool(self, user_text: str, _context: RunContext_T) -> None:
        """사용자의 일본어 발화를 분석하여 종합적인 피드백을 제공합니다."""
        
        # LLM이 분석한 결과를 바탕으로 구조화된 피드백 생성
        feedback_data = {
            "type": "comprehensive_analysis",
            "original_text": user_text,
            "analysis": {
                "grammar_accuracy": {
                    "score": "문법 정확도 점수 (1-10)",
                    "issues": "발견된 문법 문제들",
                    "corrections": "수정 제안사항",
                    "examples": "올바른 표현 예시"
                },
                "vocabulary_usage": {
                    "score": "어휘 사용 점수 (1-10)",
                    "strengths": "잘 사용된 어휘",
                    "suggestions": "개선 가능한 어휘",
                    "alternatives": "대안 표현들"
                },
                "politeness_level": {
                    "score": "정중함 수준 (1-10)",
                    "assessment": "정중함 수준 평가",
                    "improvements": "정중함 개선 방안",
                    "context_appropriateness": "상황 적절성"
                },
                "cultural_aspects": {
                    "notes": "문화적 맥락 설명",
                    "nuances": "미묘한 뉘앙스",
                    "recommendations": "문화적 이해 개선 방안"
                },
                "overall_assessment": {
                    "level": "전체적인 수준 평가",
                    "strengths": "강점들",
                    "areas_for_improvement": "개선 영역",
                    "encouragement": "격려 메시지"
                }
            },
            "recommendations": {
                "immediate_focus": "즉시 개선할 점",
                "long_term_goals": "장기 학습 목표",
                "practice_suggestions": "연습 방법 제안"
            },
            "category": "comprehensive"
        }
        await self.send_json_feedback(feedback_data)

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