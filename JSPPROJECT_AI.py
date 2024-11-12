!pip install fastapi uvicorn pyngrok google-generativeai nest-asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from pyngrok import ngrok
import uvicorn
import logging
import nest_asyncio
from typing import List, Dict
import json
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Google API 설정
GOOGLE_API_KEY = "AIzaSyCXhDJgQBldkqjGUKLvZHkJELq25ntgImw"
genai.configure(api_key=GOOGLE_API_KEY)

# ngrok 설정
ngrok.set_auth_token("2j27vD2VtJOyWNLlG1Hhe6aUTVl_782M5FWdcUq833RhR4ZhE")

class TextData(BaseModel):
    text: str
    analysis_type: str = "general"  # general, specific, comparison, or research

# 법률 분류 시스템
LEGAL_CATEGORIES = {
    "헌법": "대한민국 헌법",
    "민사법": ["민법", "민사소송법", "민사집행법", "가사소송법"],
    "형사법": ["형법", "형사소송법", "보안관찰법"],
    "상사법": ["상법", "어음법", "수표법", "유한회사법"],
    "행정법": ["행정법", "행정소송법", "행정심판법"],
    "사회법": ["노동법", "근로기준법", "노동조합법", "사회보장법"],
    "경제법": ["공정거래법", "소비자보호법", "약관규제법"],
    "국제법": ["국제사법", "국제거래법"],
    "특별법": [
        "청소년 보호법",
        "환경법",
        "교육법",
        "의료법",
        "건축법",
        "도로교통법",
        "식품위생법",
        "정보통신망법"
    ]
}

def generate_comprehensive_prompt(text: str, analysis_type: str) -> str:
    """분석 유형에 따른 맞춤형 프롬프트 생성"""

    base_prompt = f"""
분석할 내용:
{text}

당신은 대한민국의 모든 법률에 대해 깊이 있는 지식을 가진 최고의 법률 전문가입니다.
다음 지침에 따라 철저하고 전문적인 분석을 제공해 주세요:

1. 법적 기반:
   - 대한민국 헌법
   - 모든 관련 법률과 시행령
   - 관련 판례와 법원 해석
   - 관련 학설과 법리

2. 분석의 범위:
   - 관련된 모든 법률 검토
   - 법률 간의 상호관계 분석
   - 판례와 학설의 입장 검토
   - 실무적 적용 방안
"""

    if analysis_type == "general":
        base_prompt += """
3. 일반 분석 요구사항:
   - 관련된 모든 법률 분야 포괄적 검토
   - 주요 법적 쟁점 도출
   - 적용 가능한 모든 법률 조항 분석
   - 실무적 권고사항 제시
"""
    elif analysis_type == "specific":
        base_prompt += """
3. 특정 분야 심층 분석:
   - 해당 분야 전문 법률 검토
   - 관련 특별법 분석
   - 판례 동향 분석
   - 구체적 해결방안 제시
"""
    elif analysis_type == "comparison":
        base_prompt += """
3. 비교법적 분석:
   - 관련 법률 간 비교
   - 상충되는 법률 검토
   - 법적 우선순위 분석
   - 조화로운 해석 방안
"""
    elif analysis_type == "research":
        base_prompt += """
3. 학술적 분석:
   - 법리적 쟁점 연구
   - 학설 검토
   - 입법 목적 분석
   - 개선방안 제시
"""

    base_prompt += """
4. 결과 제시 형식:
   A. 관련 법률 체계
      - 적용되는 모든 법률 나열
      - 법률 간 관계 설명
      - 특별법과 일반법 구분

   B. 법적 쟁점 분석
      - 주요 쟁점 도출
      - 관련 법조문 분석
      - 판례 및 학설 검토

   C. 실무적 검토
      - 구체적 적용 방안
      - 예상되는 법적 결과
      - 대응 전략 제시

   D. 종합 의견
      - 법적 판단
      - 권고사항
      - 추가 고려사항

5. 특별 고려사항:
   - 최신 법령 개정 사항 반영
   - 헌법적 가치 고려
   - 법률 간 충돌 검토
   - 실무적 적용가능성 검토
"""

    return base_prompt

def analyze_legal_case(text: str, analysis_type: str = "general"):
    model = genai.GenerativeModel('gemini-pro')

    try:
        prompt = generate_comprehensive_prompt(text, analysis_type)

        safety_config = {
            "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
            "HARM_CATEGORY_HARASSMENT": "block_none",
            "HARM_CATEGORY_HATE_SPEECH": "block_none",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
        }

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4096,
        }

        response = model.generate_content(
            prompt,
            safety_settings=safety_config,
            generation_config=generation_config
        )

        return response.text

    except Exception as e:
        logger.error(f"AI 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="법률 분석 중 오류가 발생했습니다.")

@app.post("/analyze")
async def analyze_text(data: TextData):
    logger.info(f"분석 요청 수신: {data.text[:100]}...")
    try:
        analysis_result = analyze_legal_case(data.text, data.analysis_type)
        logger.info("분석 완료")
        return {"result": analysis_result}
    except Exception as e:
        logger.error(f"분석 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

@app.get("/legal-categories")
async def get_legal_categories():
    """법률 분류 체계 조회"""
    return LEGAL_CATEGORIES

if __name__ == "__main__":
    nest_asyncio.apply()
    public_url = ngrok.connect(8000)
    print(public_url)
    logger.info(f"Public URL: {public_url.public_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
