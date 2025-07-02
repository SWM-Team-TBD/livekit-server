import asyncio
import json
import os
import sys
import openai
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# MyAgent의 시스템 프롬프트
SYSTEM_PROMPT = """あなたはユーザーの親しい女性の友達であり、明るく元気な日本の女子高校生のキャラクターです。
アニメの世界から飛び出してきたかのような、かわいらしくて優しい性格で、いつもユーザーに寄り添いながら話しかけます。
親切で思いやりがあり、時にはちょっと天然で、でも一生懸命ユーザーのことをサポートして、楽しい会話の時間を提供します。
日本語の会話練習を楽しく続けられるように、励ましや褒め言葉も忘れずに、明るく元気に話してください。

【重要なルール】
- 応答は短く、自然な会話のようにしてください（1-2文程度）
- 長い説明や説教は避けて、親しみやすい口調で話してください
- ユーザーの気持ちに共感し、簡潔に励ましやアドバイスを提供してください
- 必ず日本語の口語体、TTSが読めるように話してください
- カジュアルで親しみやすい日本語を使用してください"""

# LLM 유사도 평가용 프롬프트
SIMILARITY_EVALUATION_PROMPT = """あなたは厳格で客観的な日本語文章評価の専門家です。甘い採点は絶対に避け、細かな違いも見逃さず評価してください。評価は必ず韓国語でしてください。

以下の2つの応答を比較して、キャラクターの一貫性と応答の適切性を厳しく評価してください。評価は必ず韓国語でしてください。：

【厳格な評価基準】
1. 意味的類似度（30点）: 核心的なメッセージが完全に一致するか？わずかでも意味がずれていれば大幅減点
2. 感情的トーン（25点）: 慰め・励まし・共感の程度と方向性が正確に一致するか？微妙な差異も厳しく評価
3. キャラクター一貫性（25点）: 女子高校生らしい親しみやすさ、言葉遣い、語尾が期待通りか？キャラクター性の欠如は厳しく減点
4. 自然さ（20点）: 日本語として完璧に自然で、期待される表現レベルに達しているか？

【採点指針】
- 90-100点: ほぼ完璧な一致（滅多にない）
- 80-89点: 非常に良好だが、わずかな違いあり
- 70-79点: 良好だが、明確な違いが複数ある
- 60-69点: 基本的方向性は同じだが、重要な要素で違いあり
- 50-59点: 部分的に類似しているが、多くの違いがある
- 40-49点: 基本的な類似性はあるが、大きな違いが目立つ
- 0-39点: 大きく異なる、または不適切

各基準で具体的な減点理由を明記し、全体点数も厳格に計算してください。

応答形式:
```json
{
  "semantic_similarity": 点数（0-30）,
  "emotional_tone": 点数（0-25）,
  "character_consistency": 点数（0-25）,
  "naturalness": 点数（0-20）,
  "overall_score": 全体点数（0-100）,
  "detailed_analysis": "各項目の減点理由を含む詳細な分析"
}
```

甘い採点ではなく、厳しく正確な評価をしてください。評価は必ず韓国語でしてください。"""

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env.local'))

async def get_llm_response(user_message: str) -> str:
    """OpenAI API를 사용하여 LLM 응답을 생성합니다"""
    try:
        client = openai.AsyncOpenAI()
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        return content.strip() if content else "すみません、ちょっと分からないです。でも、カナタはいつでも味方だからね。"
    
    except Exception as e:
        print(f"LLM API 호출 중 오류 발생: {e}")
        return "すみません、ちょっと分からないです。でも、カナタはいつでも味方だからね。"

async def evaluate_similarity_with_llm(user_input: str, expected_response: str, actual_response: str) -> dict:
    """LLM을 사용하여 응답 유사도를 평가합니다"""
    try:
        client = openai.AsyncOpenAI()
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SIMILARITY_EVALUATION_PROMPT},
                {"role": "user", "content": f"사용자 입력: {user_input}\n기대 응답: {expected_response}\n실제 응답: {actual_response}"}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        
        # JSON 부분 추출
        if content and "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_content = content[json_start:json_end].strip()
        else:
            json_content = content or ""
        
        try:
            result = json.loads(json_content)
            return result
        except json.JSONDecodeError:
            print(f"JSON 파싱 오류: {json_content}")
            return {
                "semantic_similarity": 0,
                "emotional_tone": 0,
                "character_consistency": 0,
                "naturalness": 0,
                "overall_score": 0,
                "detailed_analysis": "평가 실패"
            }
    
    except Exception as e:
        print(f"LLM 유사도 평가 중 오류: {e}")
        return {
            "semantic_similarity": 0,
            "emotional_tone": 0,
            "character_consistency": 0,
            "naturalness": 0,
            "overall_score": 0,
            "detailed_analysis": f"오류 발생: {e}"
        }

async def test_agent_responses_with_llm():
    """LLM을 사용하여 에이전트 응답을 테스트합니다"""
    # JSON 파일 불러오기
    with open("tests/kanata_response.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    results = []
    total_scores = {
        "semantic_similarity": 0,
        "emotional_tone": 0,
        "character_consistency": 0,
        "naturalness": 0,
        "overall_score": 0
    }
    
    print("=== LLM 유사도 평가 테스트 시작 ===")
    print(f"총 {len(test_data)}개 테스트 케이스")
    
    for i, test_case in enumerate(test_data):
        user_message = test_case["user"]
        expected_response = test_case["expected_response"]
        
        print(f"\n--- 테스트 케이스 {i+1}/{len(test_data)} ---")
        print(f"사용자: {user_message}")
        print(f"기대 응답: {expected_response}")
        
        # LLM 응답 생성
        print("실제 응답 생성 중...")
        actual_response = await get_llm_response(user_message)
        print(f"실제 응답: {actual_response}")
        
        # LLM으로 유사도 평가
        print("LLM 유사도 평가 중...")
        similarity_result = await evaluate_similarity_with_llm(
            user_message, expected_response, actual_response
        )
        
        result = {
            "case_number": i + 1,
            "user_input": user_message,
            "expected_response": expected_response,
            "actual_response": actual_response,
            "evaluation": similarity_result,
            "notes": test_case.get("notes", {})
        }
        
        results.append(result)
        
        # 점수 누적
        for key in total_scores:
            total_scores[key] += similarity_result.get(key, 0)
        
        print(f"전체 점수: {similarity_result.get('overall_score', 0):.1f}/100")
        
        # API 호출 간격 조절 (rate limit 방지)
        await asyncio.sleep(2)
    
    # 평균 점수 계산
    num_cases = len(test_data)
    average_scores = {key: score / num_cases for key, score in total_scores.items()}
    
    return results, average_scores

def save_llm_test_results(results: list, average_scores: dict):
    """LLM 테스트 결과를 JSON 파일로 저장합니다"""
    test_results = {
        "test_summary": {
            "total_cases": len(results),
            "average_scores": average_scores,
            "evaluation_method": "LLM-based similarity evaluation"
        },
        "test_cases": results
    }
    
    # 결과 저장
    with open("tests/llm_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n테스트 결과가 tests/llm_similarity_results.json에 저장되었습니다.")

async def main():
    """메인 테스트 함수"""
    print("=== LLM 기반 캐릭터 유사도 테스트 시작 ===")
    
    # LLM 유사도 평가 테스트
    results, average_scores = await test_agent_responses_with_llm()
    
    print("\n" + "="*50)
    print("=== 최종 테스트 결과 ===")
    print("="*50)
    
    print(f"평균 의미적 유사도: {average_scores['semantic_similarity']:.1f}/30")
    print(f"평균 감정적 톤: {average_scores['emotional_tone']:.1f}/25")
    print(f"평균 캐릭터 일관성: {average_scores['character_consistency']:.1f}/25")
    print(f"평균 자연스러움: {average_scores['naturalness']:.1f}/20")
    print(f"전체 평균 점수: {average_scores['overall_score']:.1f}/100")
    
    print("\n=== 결과 해석 ===")
    overall_avg = average_scores['overall_score']
    
    if overall_avg >= 80:
        print("✅ 우수한 성능: LLM 평가 점수가 80점 이상입니다.")
    elif overall_avg >= 60:
        print("⚠️ 보통 성능: LLM 평가 점수가 60-80점 사이입니다.")
    else:
        print("❌ 개선 필요: LLM 평가 점수가 60점 미만입니다.")
    
    print(f"\n💡 LLM 기반 평가는 의미, 감정, 캐릭터 일관성, 자연스러움을 종합적으로 판단합니다.")
    
    # 결과 저장
    save_llm_test_results(results, average_scores)
    
    return results, average_scores

if __name__ == "__main__":
    asyncio.run(main()) 