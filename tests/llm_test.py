import asyncio
import json
import os
import sys
import openai
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
- カジュアルで親しみやすい日本語を使用してください"""

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env.local'))

def compute_sentence_embedding_similarity(expected_responses, actual_responses):
    """문장 임베딩 기반 코사인 유사도를 계산합니다"""
    try:
        # 일본어 문장 임베딩 모델 로드 (무료 모델)
        # 'intfloat/multilingual-e5-large' 또는 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        print(f"문장 임베딩 모델 로딩 중: {model_name}")
        
        model = SentenceTransformer(model_name)
        
        # 모든 응답을 하나의 리스트로 합치기
        all_responses = expected_responses + actual_responses
        
        # 문장 임베딩 생성
        print("문장 임베딩 생성 중...")
        embeddings = model.encode(all_responses, convert_to_tensor=True)
        
        # 기대 응답과 실제 응답 분리
        expected_embeddings = embeddings[:len(expected_responses)]
        actual_embeddings = embeddings[len(expected_responses):]
        
        similarities = []
        for i in range(len(expected_responses)):
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                expected_embeddings[i].cpu().numpy().reshape(1, -1),
                actual_embeddings[i].cpu().numpy().reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        # 디버깅 정보 출력
        print(f"\n=== 문장 임베딩 디버깅 정보 ===")
        print(f"모델: {model_name}")
        print(f"임베딩 차원: {embeddings.shape}")
        print(f"응답 수: {len(all_responses)}")
        
        return {
            "average_sentence_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "individual_similarities": similarities
        }
    
    except Exception as e:
        print(f"문장 임베딩 유사도 계산 중 오류: {e}")
        # 오류 발생 시 0으로 채움
        similarities = [0.0] * len(expected_responses)
        return {
            "average_sentence_similarity": 0.0,
            "individual_similarities": similarities
        }

async def get_llm_response(user_message: str) -> str:
    """OpenAI API를 사용하여 LLM 응답을 생성합니다"""
    try:
        # OpenAI 클라이언트 설정 (환경변수에서 API 키 가져오기)
        client = openai.AsyncOpenAI()
        
        response = await client.chat.completions.create(
            model="gpt-4",  # 또는 "gpt-3.5-turbo"
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
        # API 호출 실패 시 기본 응답 반환
        return "すみません、ちょっと分からないです。でも、カナタはいつでも味方だからね。"

async def test_agent_responses():
    """에이전트 응답을 테스트합니다"""
    # 1. JSON 파일 불러오기
    with open("tests/kanata_response.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # 2. 각 테스트 케이스에 대해 LLM 응답 생성
    actual_responses = []
    expected_responses = []
    user_inputs = []
    
    print("=== MyAgent 시스템 프롬프트로 LLM 응답 테스트 시작 ===")
    print(f"시스템 프롬프트: {SYSTEM_PROMPT[:100]}...")
    
    for i, test_case in enumerate(test_data):
        user_message = test_case["user"]
        expected_response = test_case["expected_response"]
        
        print(f"\n--- 테스트 케이스 {i+1} ---")
        print(f"사용자: {user_message}")
        print(f"기대 응답: {expected_response}")
        
        # LLM API 호출
        print("LLM 응답 생성 중...")
        actual_response = await get_llm_response(user_message)
        
        actual_responses.append(actual_response)
        expected_responses.append(expected_response)
        user_inputs.append(user_message)
        
        print(f"실제 응답: {actual_response}")
        
        # API 호출 간격 조절 (rate limit 방지)
        await asyncio.sleep(1)
    
    return actual_responses, expected_responses, user_inputs

def save_test_results(actual_responses, expected_responses, user_inputs, results):
    """테스트 결과를 JSON 파일로 저장합니다"""
    test_results = {
        "test_cases": [],
        "summary": {
            "average_sentence_similarity": float(results["sentence_results"]["average_sentence_similarity"])
        }
    }
    
    for i, (actual, expected, user_input) in enumerate(zip(actual_responses, expected_responses, user_inputs)):
        sentence_sim = float(results["sentence_results"]["individual_similarities"][i]) if i < len(results["sentence_results"]["individual_similarities"]) else 0.0
        
        test_results["test_cases"].append({
            "case_number": i + 1,
            "user_input": user_input,
            "expected_response": expected,
            "actual_response": actual,
            "sentence_similarity": sentence_sim
        })
    
    # 결과 저장
    with open("tests/test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n테스트 결과가 tests/test_results.json에 저장되었습니다.")

async def main():
    """메인 테스트 함수"""
    print("=== MyAgent 시스템 프롬프트 LLM 테스트 시작 ===")
    
    # 에이전트 응답 테스트
    actual_responses, expected_responses, user_inputs = await test_agent_responses()
    
    # 문장 임베딩 유사도 계산
    sentence_results = compute_sentence_embedding_similarity(expected_responses, actual_responses)
    
    print("\n" + "="*50)
    print("=== 최종 테스트 결과 ===")
    print("="*50)
    print(f"평균 문장 임베딩 유사도: {sentence_results['average_sentence_similarity']:.4f}")
    
    print("\n=== 개별 케이스 상세 결과 ===")
    for i, sentence_sim in enumerate(sentence_results['individual_similarities']):
        print(f"케이스 {i+1}: 문장 임베딩 유사도 = {sentence_sim:.4f}")
    
    # 결과 요약
    print("\n=== 결과 해석 ===")
    avg_sentence = sentence_results['average_sentence_similarity']
    
    if avg_sentence > 0.7:
        print("✅ 우수한 성능: 문장 임베딩 유사도가 0.7 이상입니다.")
    elif avg_sentence > 0.5:
        print("⚠️ 보통 성능: 문장 임베딩 유사도가 0.5-0.7 사이입니다.")
    else:
        print("❌ 개선 필요: 문장 임베딩 유사도가 0.5 미만입니다.")
    
    print(f"\n💡 문장 임베딩은 의미적 유사도를 정확하게 측정합니다.")
    
    # 결과 저장
    results = {
        "sentence_results": sentence_results
    }
    save_test_results(actual_responses, expected_responses, user_inputs, results)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
