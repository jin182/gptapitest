!pip install fastapi uvicorn pyngrok google-generativeai nest-asyncio


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from pyngrok import ngrok
import uvicorn
import logging
import nest_asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up Google API
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Set up ngrok
ngrok.set_auth_token("")

class TextData(BaseModel):
    text: str

def analyze_legal_case(text):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""분석해야 할 법률 사건:
    {text}

    이 사건에 대해 다음 정보를 제공해주세요:
    1. 적용되는 법률:
        - 이 사건과 관련된 모든 법률 조항들을 간단히 나열하고, 각각이 어떻게 적용되는지 설명해주세요.
        - 법률 조항의 번호와 제목을 명시하고, 해당 조항이 사건에 어떤 영향을 미치는지 구체적으로 설명해주세요.
    2. 주요 법적 쟁점:
        - 이 사건에서 다투어지는 핵심 법적 쟁점들을 요약하고, 각각의 쟁점이 법률적으로 어떤 의미를 가지는지 설명해주세요.
        - 각 쟁점에 대한 당사자의 입장과 이에 대한 법적 논거를 명확하게 제시해주세요.
    3. 잠재적 위험:
        - 사건 당사자들이 겪을 수 있는 잠재적 법적 위험이나 책임을 설명하고, 가능한 시나리오를 제시해주세요.
        - 각 시나리오에 대해 발생 가능한 법적 결과와 그에 따른 영향을 설명해주세요.
    4. 유사 판례 (있다면):
        - 이 사건과 유사한 판례들을 예시로 들어주시고, 해당 판례들이 이 사건과 어떻게 연관될 수 있는지 설명해주세요.
        - 판례의 이름, 번호, 결정 내용 등을 포함하여 설명하고, 이 판례가 현재 사건에 주는 시사점을 제시해주세요.
    5. 기타 관련 법률 사항 (해당되는 경우):
        - 이 사건과 관련된 모든 법률(청소년 기본법 포함)에 대한 사항을 설명해주세요.
        - 각 법률이 사건에 미치는 영향과 적용 여부를 설명해주세요.
    6. 사건의 예상 결과 및 권고 사항:
        - 이 사건의 예상 결과를 간략히 제시하고, 이를 바탕으로 당사자에게 줄 수 있는 법적 권고 사항을 설명해주세요.
        - 권고 사항은 법적 대응 방안, 잠재적 합의 가능성, 추가적인 법적 조치 등을 포함해주세요.
    
    각 항목에 대해 명확하고 간결한 형식으로 설명해 주세요:
    - 적용되는 법률: (여기에 설명)
    - 주요 법적 쟁점: (여기에 설명)
    - 잠재적 위험: (여기에 설명)
    - 유사 판례: (여기에 설명)
    - 기타 관련 법률 사항: (여기에 설명)
    - 사건의 예상 결과 및 권고 사항: (여기에 설명)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Generative AI 응답 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="AI 모델을 통해 분석 중 오류가 발생했습니다.")

@app.post("/analyze")
async def analyze_text(data: TextData):
    logger.info(f"Received request with text: {data.text}")
    try:
        analysis_result = analyze_legal_case(data.text)
        logger.info("Analysis completed successfully")
        return {"result": analysis_result}
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    # Apply nest_asyncio to allow running uvicorn in a notebook environment
    nest_asyncio.apply()

    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print( public_url)
    logger.info(f"Public URL: {public_url.public_url}")  # 수정된 부분: .public_url 추가

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)

