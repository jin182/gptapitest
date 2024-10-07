!pip install fastapi[all] uvicorn[standard] pyngrok googletrans==4.0.0-rc1 google-generativeai diffusers torch pillow

import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import torch
from PIL import Image
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
from typing import Optional
import logging
import asyncio
from googletrans import Translator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Smart Farm Diary API",
    description="An API for generating farm diaries and images",
    version="1.0.0"
)

# Set environment variables
GOOGLE_API_KEY = "AIzaSyCXhDJgQBldkqjGUKLvZHkJELq25ntgImw"  
NGROK_AUTH_TOKEN = "2j27vD2VtJOyWNLlG1Hhe6aUTVl_782M5FWdcUq833RhR4ZhE"
SAVE_PATH = "./farm_diary_outputs"
os.makedirs(SAVE_PATH, exist_ok=True)

# NGrok configuration
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Gemini API configuration
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize translation and diffusion models
pipe = None
translator = Translator()

def initialize_dreamlike_photoreal():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "dreamlike-art/dreamlike-photoreal-2.0",
            torch_dtype=torch.float16
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            pipe.enable_xformers_memory_efficient_attention()
    return pipe

# Define the Pydantic models
class DiaryEntry(BaseModel):
    crop: str
    weather: str
    temperature: float
    humidity: float
    issues: str
    work: str

class DiaryResponse(BaseModel):
    diary_content: str
    image_path: Optional[str]
    created_at: str

# Custom translation function
def custom_translate(text, dest='en'):
    if text == "사과":
        return "apple"
    return translator.translate(text, dest=dest).text

# Define the FastAPI endpoint
@app.post("/generate_diary/", response_model=DiaryResponse)
async def create_diary(entry: DiaryEntry):
    try:
        # Simulate long task
        await asyncio.sleep(5)

        # Translate Korean to English with custom translation
        translated_crop = custom_translate(entry.crop, dest='en')
        translated_weather = custom_translate(entry.weather, dest='en')
        translated_issues = custom_translate(entry.issues, dest='en')
        translated_work = custom_translate(entry.work, dest='en')

        # Generate diary content
        diary_content = f"""
        전문 농업 컨설턴트의 관점에서 다음 정보를 바탕으로 상세한 농가 일지를 작성합니다:
        
        작성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시')}
        
        기본 정보:
        - 작물: {entry.crop}
        - 기상 상황: {entry.weather}
        - 온도: {entry.temperature}°C
        - 습도: {entry.humidity}%
        
        주요 사항:
        - 특이사항: {entry.issues}
        - 작업내용: {entry.work}
        
        다음 항목들을 포함해 전문적으로 작성합니다:
        1. 작물 생육 상태 분석
        2. 환경 요인 영향 평가
        3. 수행된 작업의 적절성 평가
        4. 향후 1주일간의 관리 권장사항
        5. 예상되는 문제점과 대응 방안
        6. 수확량 예측 및 품질 전망
        """

        # Generate diary content using Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(diary_content)
        generated_diary = response.text

        # Initialize image generation
        pipe = initialize_dreamlike_photoreal()
        prompt = f"""
        A professional agricultural photograph of a {translated_crop} farm.
        {translated_weather} weather conditions with {entry.temperature}°C temperature.
        
        Photorealistic style with the following specific details:
        - Crystal clear 8K resolution
        - Professional agricultural photography
        - Golden hour lighting
        - Perfect exposure and composition
        - Vibrant natural colors
        - Shallow depth of field where appropriate
        
        Must show:
        - Detailed view of {translated_crop} plants
        - Evidence of {translated_work} activities
        - {translated_weather} conditions in the sky
        - Subtle signs of {translated_issues}
        - Modern farming equipment
        
        Additional specifications:
        - Hyperrealistic detail
        - Professional agricultural setting
        - Environmental context
        """

        # Generate image
        with torch.cuda.amp.autocast():
            image = pipe(prompt, num_inference_steps=75, guidance_scale=10.0, width=768, height=768).images[0]

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"farm_image_{timestamp}.png"
        diary_filename = f"farm_diary_{timestamp}.txt"
        image_path = os.path.join(SAVE_PATH, image_filename)
        diary_path = os.path.join(SAVE_PATH, diary_filename)

        # Save image
        image.save(image_path)

        # Save diary content
        with open(diary_path, 'w', encoding='utf-8') as f:
            f.write(generated_diary)

        return DiaryResponse(
            diary_content=generated_diary,
            image_path=image_path,
            created_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Processing error")

# Run the server
def run_server():
    # Create NGrok tunnel
    public_url = ngrok.connect(8000)
    print(f'NGrok tunnel created. Public URL: {public_url.public_url}')

    # Configure and run FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Apply nest_asyncio
    nest_asyncio.apply()
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        ngrok.disconnect(public_url.public_url)

# Execute the server when run as main
if __name__ == "__main__":
    run_server()
