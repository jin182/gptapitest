!pip install pyngrok fastapi uvicorn torch diffusers transformers google-generativeai pillow nest-asyncio xformers deep-translator

import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import torch
from PIL import Image
import io
import google.generativeai as genai
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from typing import Optional
import logging
import asyncio
from deep_translator import GoogleTranslator
import boto3
from botocore.exceptions import ClientError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Smart Farm Diary API",
    description="농업 일지 생성 및 이미지 생성을 위한 API",
    version="1.0.0"
)

# 키 설정 (테스트용 하드코딩)
GOOGLE_API_KEY = ""
NGROK_AUTH_TOKEN = ""
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = ""
S3_BUCKET_NAME = ""
S3_BUCKET_PATH = "

# NGrok 설정
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Gemini API 설정
genai.configure(api_key=GOOGLE_API_KEY)

# AWS S3 클라이언트 설정
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Stable Diffusion 모델 (지연 초기화)
pipe = None

# 번역기 설정
translator = GoogleTranslator(source='ko', target='en')

class DiaryEntry(BaseModel):
    crop: str
    weather: str
    temperature: float
    humidity: float
    issues: str
    work: str

class DiaryResponse(BaseModel):
    diary_content: str
    image_url: Optional[str]
    created_at: str

def initialize_stable_diffusion():
    global pipe
    if pipe is None:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "dreamlike-art/dreamlike-photoreal-2.0",
                torch_dtype=torch.float16,
                safety_checker=None
            )
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                pipe.enable_attention_slicing()
                if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                    pipe.enable_xformers_memory_efficient_attention()
            else:
                logger.warning("CUDA is not available. Using CPU for inference.")
                pipe = pipe.to("cpu")
        except Exception as e:
            logger.error(f"Failed to initialize Stable Diffusion: {str(e)}")
            raise
    return pipe

def generate_enhanced_prompt(entry: DiaryEntry) -> str:
    # 작물 이름 사전
    crop_dict = {
        "사과": "apple",
        "샤인머스캣": "Shine Muscat grape",
        "토마토": "tomato",
        "열무": "young summer radish",
        "오이": "cucumber",
        "배추": "Brassica rapa",
        "고추": "chili pepper"
    }
    
    try:
        # 번역 수행
        translated_crop = crop_dict.get(entry.crop, translator.translate(entry.crop))
        translated_weather = translator.translate(entry.weather)
        translated_issues = translator.translate(entry.issues)
        translated_work = translator.translate(entry.work)

        # CLIP 토큰 제한(77)을 고려한 간결한 프롬프트
        prompt = f"""Professional {translated_crop} farm photo, {translated_weather}, \
        showing {translated_work}. {translated_issues} visible. \
        Photorealistic, high detail, perfect exposure."""

        return prompt

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Translation failed")

def generate_diary_content(entry: DiaryEntry) -> str:
    prompt = f"""
    전문 농업 컨설턴트의 관점에서 다음 정보를 바탕으로 상세한 농가 일지를 작성해주세요:

    작성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시')}

    기본 정보:
    - 작물: {entry.crop}
    - 기상 상황: {entry.weather}
    - 온도: {entry.temperature}°C
    - 습도: {entry.humidity}%

    주요 사항:
    - 특이사항: {entry.issues}
    - 작업내용: {entry.work}

    다음 항목들을 포함해 전문적으로 작성해주세요:
    1. 작물 생육 상태 분석
    2. 환경 요인 영향 평가
    3. 수행된 작업의 적절성 평가
    4. 향후 1주일간의 관리 권장사항
    5. 예상되는 문제점과 대응 방안
    6. 수확량 예측 및 품질 전망
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"일지 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="일지 생성 실패")

async def upload_to_s3(image: Image.Image, filename: str) -> str:
    try:
        # PIL Image를 바이트로 변환
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # S3에 업로드
        s3_key = f"{S3_BUCKET_PATH}{filename}"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=img_byte_arr,
            ContentType='image/png'
        )

        # URL 생성
        url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return url

    except ClientError as e:
        logger.error(f"S3 upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload image to S3")

@app.post("/generate_diary/", response_model=DiaryResponse)
async def create_diary(entry: DiaryEntry):
    try:
        # 일지 내용 생성
        diary_content = generate_diary_content(entry)
        
        # 이미지 생성
        pipe = initialize_stable_diffusion()
        prompt = generate_enhanced_prompt(entry)
        
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]

        # 이미지 저장 및 URL 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"farm_image_{timestamp}.png"
        image_url = await upload_to_s3(image, image_filename)

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return DiaryResponse(
            diary_content=diary_content,
            image_url=image_url,
            created_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in diary creation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_server():
    try:
        # NGrok 터널 생성
        public_url = ngrok.connect(8000)
        print(f'NGrok 터널이 생성되었습니다. Public URL: {public_url.public_url}')

        # FastAPI 서버 설정
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)

        # 서버 실행
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Server shutdown initiated")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        # NGrok 터널 제거
        try:
            ngrok.disconnect(public_url)
        except:
            pass

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(run_server())
