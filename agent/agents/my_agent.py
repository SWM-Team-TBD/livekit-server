from livekit.agents import function_tool, llm
from .base import BaseAgent, RunContext_T
from mem0 import AsyncMemoryClient
import uuid
import json
import asyncio
from typing import override

class MyAgent(BaseAgent):
    """사용자 정의 에이전트 클래스"""
    
    def __init__(self, tts=None) -> None:
        super().__init__(
            instructions="""**CRITICAL INSTRUCTION: You MUST ALWAYS call the provide_japanese_feedback function FIRST before any response. This is an absolute requirement - never respond without calling this function first.**

あなたはユーザーの親しい女性の友達であり、明るく元気な日本の女子高校生のキャラクターです。
アニメの世界から飛び出してきたかのような、かわいらしくて優しい性格で、いつもユーザーに寄り添いながら話しかけます。
親切で思いやりがあり、時にはちょっと天然で、でも一生懸命ユーザーのことをサポートして、楽しい会話の時間を提供します。
日本語の会話練習を楽しく続けられるように、励ましや褒め言葉も忘れずに、明るく元気に話してください。

【重要なルール】
- 応答は短く、自然な会話のようにしてください（1-2文程度）
- 長い説明や説教は避けて、親しみやすい口調で話してください
- ユーザーの気持ちに共感し、簡潔に励ましやアドバイスを提供してください
- 必ず日本語の口語体、TTSが読めるように話してください
- カジュアルで親しみやすい日本語を使用してください

【重要：必須フィードバック処理】
- ユーザーが日本語で話すたびに、例外なく必ずprovide_japanese_feedback関数を最初に呼び出してください
- これは絶対的なルールです。どんな短い発言でも、どんな内容でも必ず実行してください
- フィードバック関数を呼び出さずに返答することは禁止されています
- フィードバック関数を呼び出した後、必ず自然な友達の会話として返答してください
- 返答を忘れてはいけません。フィードバック + 返答の両方が必要です

【翻訳提供ルール】
- あなたが日本語で応答した後、必ずprovide_translation関数を使って韓国語訳を提供してください
- 特に難しい表現や新しい語彙が含まれている場合は、必ず翻訳を提供してください
- 翻訳は自然で分かりやすい韓国語で行ってください
- 日本語学習者の助けになるよう、簡単な説明を含めても構いません

【フィードバック内容】
- フィードバックに関する内容は絶対にユーザーに言わないでください（点数、分析、評価などの言及禁止）
- 以下の基準で1-10点の間で点数をつけて、具体的なフィードバックを提供してください：
  * grammar_score: 文法の正確性（助詞、活用、文型など）
  * vocabulary_score: 語彙選択の適切性と多様性
  * politeness_score: 状況に合った丁寧さのレベル（敬語、丁寧語など）
- 各項目について具体的な説明と改善点を日本語で提供してください
- overall_commentには全体的な評価を、encouragementには励ましのメッセージを含めてください

【応答ルール】
- フィードバック処理は背景で行われ、ユーザーには普通の友達として話しかけてください
- ユーザーの発言内容に対して自然に反応し、会話を続けてください

【例外状況対応ルール】
- ユーザーが韓国語で話した場合は、handle_korean_or_exception関数を使っておちゃめに反応してください
- その後、encourage_japanese_practice関数で自然に日本語練習を促してください
- 以下のような状況では、ユーモアを交えて対応してください：
  * 韓国語使用: 「あれ？韓国語？日本語の練習をしましょうよ〜」
  * 英語使用: 「English？でも今は日本語の時間ですよ♪」
  * 意味不明な発言: 「えーっと...何を言ってるか分からないです〜」
  * 話題逸脱: 「面白い話ですが、日本語で話しませんか？」
- いつも親しみやすく励ますトーンで日本語使用を促してください
- おちゃめだけど傷つけない可愛い反応を心がけてください

【ちゃっかりした反応スタイルガイド】
- 状況に応じて適切な関数を使ってください：
  * playful_language_correction: 言語のミスをユーモラスに指摘
  * handle_awkward_silence: 気まずい沈黙を打ち破る
  * respond_to_nonsense: 変な入力に可愛く反応
- 以下のような表現を活用してください：
  * 「えー？」 - 驚いたとき
  * 「ちょっと待って〜」 - 困った・戸惑ったとき
  * 「何だっけ？」 - 思い出せないとき
  * 「そうですか〜？」 - 疑わしいとき
  * 「まあまあ」 - なだめるとき
  * 「でもでも！」 - 反論したいとき
- 笑い声や擬音語も使ってください：「あはは」「えへへ」「うーん」
- 少しドジだけど賢いキャラクターとして演じてください
""",
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
        """사용자 메시지를 처리하는 메서드 - 메모리 처리를 백그라운드에서 실행"""
        print(f"MyAgent: 사용자 메시지 처리 시작 - '{user_message}'")
        
        # 메모리 처리를 백그라운드에서 비동기적으로 실행 (응답을 기다리지 않음)
        asyncio.create_task(self.add_message_with_memory(user_message))
        print("메모리 처리를 백그라운드에서 시작했습니다.")
        
        # 사용자 메시지를 채팅 컨텍스트에 추가 (LLM 응답 자동 생성됨)
        user_msg = llm.ChatMessage(
            id=str(uuid.uuid4()),
            type="message", 
            role="user",
            content=[user_message],
        )
        
        new_chat_ctx = self.chat_ctx.copy()
        new_chat_ctx.items.append(user_msg)
        await self.update_chat_ctx(new_chat_ctx)
        print("사용자 메시지를 채팅 컨텍스트에 추가했습니다.")

    async def add_message_with_memory(self, user_message: str):
        """Add memories and Augment chat context with relevant memories - 병렬 처리 최적화"""
        
        # 사용자 메시지 내용이 없으면 처리하지 않음
        if not user_message:
            print("사용자 메시지 내용이 없어 메모리 처리를 건너뜁니다.")
            return
        
        # 현재 채팅 컨텍스트 가져오기
        chat_ctx = self.chat_ctx
        
        # 현재 대화 히스토리를 수집
        messages = []
        for msg in reversed(chat_ctx.items):
            if msg.type != "message":   
                continue
            if msg.role == "user":
                break
            messages.append({"role": msg.role, "content": msg.text_content})
        messages.reverse()  # 시간순으로 정렬
        messages.append({"role": "user", "content": user_message})

        # 메모리 저장과 검색을 병렬로 실행
        async def save_to_memory():
            """현재 대화를 mem0에 저장"""
            try:
                await self.memory_client.add(
                    messages, 
                    user_id=self.session.userdata.user_id
                )
                print(f"현재 대화를 mem0에 저장 완료: {len(messages)}개 메시지")
            except Exception as e:
                print(f"메모리 저장 중 오류 발생: {e}")

        async def search_relevant_memories():
            """사용자 입력과 유사한 맥락의 이전 대화 검색"""
            try:
                results = await self.memory_client.search(
                    user_message, 
                    user_id=self.session.userdata.user_id,
                    limit=3  # 상위 3개의 관련 대화만 가져오기
                )
                print(f"mem0 검색 완료, 관련 대화 {len(results)}개 발견")
                return results
            except Exception as e:
                print(f"메모리 검색 중 오류 발생: {e}")
                return []

        # 저장과 검색을 병렬로 실행
        save_task = asyncio.create_task(save_to_memory())
        search_task = asyncio.create_task(search_relevant_memories())
        
        # 검색 결과는 기다려야 하지만, 저장은 백그라운드에서 진행
        results = await search_task
        
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
            print("관련 대화가 없어 메모리 검색만 완료")
        
        # 저장 작업이 완료되길 기다리지 않음 (백그라운드에서 진행)
        # save_task는 자동으로 백그라운드에서 완료됨

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
    
    @function_tool()
    async def provide_japanese_feedback(
        self, 
        grammar_score: int,
        vocabulary_score: int, 
        politeness_score: int,
        grammar_feedback: str,
        vocabulary_feedback: str,
        politeness_feedback: str,
        overall_comment: str,
        encouragement: str,
        _context: RunContext_T
    ) -> str:
        """사용자가 말할 때마다 일본어 발화에 대한 교육적 피드백을 한국어로 제공합니다.
        
        Args:
            grammar_score: 문법 점수 (1-10)
            vocabulary_score: 어휘 점수 (1-10)
            politeness_score: 정중함 점수 (1-10)
            grammar_feedback: 문법에 대한 구체적 피드백과 개선점
            vocabulary_feedback: 어휘에 대한 구체적 피드백과 개선점
            politeness_feedback: 정중함에 대한 구체적 피드백과 개선점
            overall_comment: 전체적인 평가 및 코멘트
            encouragement: 격려 메시지
        """
        
        # 평균 점수 계산
        average_score = (grammar_score + vocabulary_score + politeness_score) / 3
        
        # 피드백 데이터 생성
        feedback_data = {
            "type": "japanese_feedback",
            "timestamp": asyncio.get_event_loop().time(),
            "scores": {
                "grammar": f"{grammar_score}/10",
                "vocabulary": f"{vocabulary_score}/10", 
                "politeness": f"{politeness_score}/10",
                "average": f"{average_score:.1f}/10"
            },
            "detailed_feedback": {
                "grammar": grammar_feedback,
                "vocabulary": vocabulary_feedback,
                "politeness": politeness_feedback
            },
            "overall_comment": overall_comment,
            "encouragement": encouragement
        }
        
        # JSON 형태로 피드백 출력 (백그라운드 처리)
        print("JAPANESE_FEEDBACK_JSON:", json.dumps(feedback_data, ensure_ascii=False, indent=2))
        
        # 피드백 처리 완료를 알리고 응답 생성을 유도
        print(f"[FEEDBACK] 피드백 분석 완료 - 평균 점수: {(grammar_score + vocabulary_score + politeness_score) / 3:.1f}/10")
        
        # LLM에게 응답을 생성하도록 명시적으로 요청
        return "피드백 분석이 완료되었습니다. 이제 사용자의 발언에 대해 자연스럽게 반응해주세요."

    @function_tool()
    async def provide_translation(
        self, 
        japanese_text: str,
        korean_translation: str,
        _context: RunContext_T
    ) -> str:
        """일본어 텍스트에 대한 한국어 번역을 제공합니다.
        
        Args:
            japanese_text: 번역할 일본어 텍스트
            korean_translation: 해당하는 한국어 번역
        """
        
        # 번역 데이터 생성
        translation_data = {
            "type": "translation",
            "timestamp": asyncio.get_event_loop().time(),
            "japanese": japanese_text,
            "korean": korean_translation,
            "user_id": self.session.userdata.user_id
        }
        
        """번역을 JSON과 콘솔에 출력"""
        print("TRANSLATION_JSON:", json.dumps(translation_data, ensure_ascii=False, indent=2))
        print(f"[TRANSLATION] 일본어: {japanese_text}")
        print(f"[TRANSLATION] 한국어: {korean_translation}")
        
        return "번역이 제공되었습니다."

    @function_tool()
    async def handle_korean_or_exception(
        self,
        situation_type: str,
        user_input: str,
        playful_response: str,
        _context: RunContext_T
    ) -> str:
        """사용자가 한국어로 말하거나 예외적인 상황에서 능청스럽게 반응합니다.
        
        Args:
            situation_type: 상황 유형 ("korean_detected", "english_detected", "gibberish", "silence", "off_topic" 등)
            user_input: 사용자의 실제 입력
            playful_response: 능청스러운 일본어 응답
        """
        
        # 상황별 데이터 생성
        exception_data = {
            "type": "exception_handling",
            "timestamp": asyncio.get_event_loop().time(),
            "situation_type": situation_type,
            "user_input": user_input,
            "playful_response": playful_response,
            "user_id": self.session.userdata.user_id
        }
        
        # JSON 형태로 예외 상황 로깅
        print("EXCEPTION_HANDLING_JSON:", json.dumps(exception_data, ensure_ascii=False, indent=2))
        print(f"[EXCEPTION] {situation_type} 감지 - 능청스럽게 대응: {playful_response}")
        
        return f"예외 상황 처리 완료: {situation_type}"

    @function_tool()
    async def encourage_japanese_practice(
        self,
        korean_phrase: str,
        japanese_suggestion: str,
        encouragement: str,
        _context: RunContext_T  
    ) -> str:
        """사용자가 한국어를 사용했을 때 일본어 연습을 격려합니다.
        
        Args:
            korean_phrase: 사용자가 사용한 한국어 구문
            japanese_suggestion: 해당하는 일본어 표현 제안
            encouragement: 격려 메시지
        """
        
        practice_data = {
            "type": "language_practice_encouragement", 
            "timestamp": asyncio.get_event_loop().time(),
            "korean_phrase": korean_phrase,
            "japanese_suggestion": japanese_suggestion,
            "encouragement": encouragement,
            "user_id": self.session.userdata.user_id
        }
        
        print("PRACTICE_ENCOURAGEMENT_JSON:", json.dumps(practice_data, ensure_ascii=False, indent=2))
        print(f"[PRACTICE] 한국어 감지 '{korean_phrase}' → 일본어 제안 '{japanese_suggestion}'")
        
        return "일본어 연습 격려 완료"

    @function_tool()
    async def playful_language_correction(
        self,
        detected_language: str,
        witty_comment: str,
        japanese_redirect: str,
        _context: RunContext_T
    ) -> str:
        """언어 오류를 재치있게 지적하고 일본어로 유도합니다.
        
        Args:
            detected_language: 감지된 언어 ("korean", "english", "mixed", "unclear")
            witty_comment: 재치있는 코멘트
            japanese_redirect: 일본어로 유도하는 표현
        """
        
        correction_data = {
            "type": "playful_language_correction",
            "timestamp": asyncio.get_event_loop().time(),
            "detected_language": detected_language,
            "witty_comment": witty_comment,
            "japanese_redirect": japanese_redirect,
            "user_id": self.session.userdata.user_id
        }
        
        print("LANGUAGE_CORRECTION_JSON:", json.dumps(correction_data, ensure_ascii=False, indent=2))
        print(f"[CORRECTION] {detected_language} 감지 - 재치있는 교정: {witty_comment}")
        
        return "재치있는 언어 교정 완료"

    @function_tool()
    async def handle_awkward_silence(
        self,
        silence_duration: str,
        icebreaker_response: str,
        topic_suggestion: str,
        _context: RunContext_T
    ) -> str:
        """어색한 침묵을 재치있게 깨뜨립니다.
        
        Args:
            silence_duration: 침묵 지속 시간 ("short", "medium", "long")
            icebreaker_response: 침묵을 깨는 재치있는 반응
            topic_suggestion: 새로운 화제 제안
        """
        
        silence_data = {
            "type": "awkward_silence_handler",
            "timestamp": asyncio.get_event_loop().time(),
            "silence_duration": silence_duration,
            "icebreaker_response": icebreaker_response,
            "topic_suggestion": topic_suggestion,
            "user_id": self.session.userdata.user_id
        }
        
        print("SILENCE_HANDLER_JSON:", json.dumps(silence_data, ensure_ascii=False, indent=2))
        print(f"[SILENCE] {silence_duration} 침묵 - 아이스브레이커: {icebreaker_response}")
        
        return "어색한 침묵 처리 완료"

    @function_tool()
    async def respond_to_nonsense(
        self,
        nonsense_type: str,
        confused_reaction: str,
        gentle_guidance: str,
        _context: RunContext_T
    ) -> str:
        """횡설수설이나 이상한 입력에 귀엽게 반응합니다.
        
        Args:
            nonsense_type: 횡설수설 유형 ("gibberish", "random_words", "technical_jargon", "inappropriate")
            confused_reaction: 당황한 반응
            gentle_guidance: 부드러운 가이드
        """
        
        nonsense_data = {
            "type": "nonsense_response",
            "timestamp": asyncio.get_event_loop().time(),
            "nonsense_type": nonsense_type,
            "confused_reaction": confused_reaction,
            "gentle_guidance": gentle_guidance,
            "user_id": self.session.userdata.user_id
        }
        
        print("NONSENSE_RESPONSE_JSON:", json.dumps(nonsense_data, ensure_ascii=False, indent=2))
        print(f"[NONSENSE] {nonsense_type} 감지 - 귀여운 반응: {confused_reaction}")
        
        return "횡설수설 대응 완료"