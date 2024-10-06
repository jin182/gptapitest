from openai import OpenAI
from datetime import datetime

# OpenAI 클라이언트 초기화
client = OpenAI(api_key="")

def generate_diary_content(crop, weather, temperature, humidity, issues, work):
    prompt = f"""
    농업 전문가의 관점에서 다음 정보를 바탕으로 상세한 농가 일지를 작성해주세요:
    
    날짜: {datetime.now().strftime('%Y년 %m월 %d일')}
    작물: {crop}
    날씨: {weather}
    온도: {temperature}°C
    습도: {humidity}%
    특이사항: {issues}
    작업내용: {work}
    
    다음 항목들을 포함해주세요:
    1. 작물 상태 전문적 분석
    2. 환경 조건이 작물에 미치는 영향
    3. 수행한 작업의 적절성 평가
    4. 향후 관리 권장사항
    5. 예상되는 문제점과 예방책
    
    전문적이고 구체적으로 작성해주세요.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"일지 생성 중 오류 발생: {str(e)}")
        return None

def generate_farm_image(crop, weather, issues):
    prompt = f"Professional agricultural photograph showing {crop} plants in {weather} weather conditions. Close-up shot focusing on plant leaves showing {issues}. Style: hyper-realistic, documentary photography, natural lighting, 4K quality. The image should clearly show the agricultural setting and plant details."

    try:
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {str(e)}")
        return None

def create_farm_diary_entry(crop, weather, temperature, humidity, issues, work):
    print("\n==== 새로운 일지 시작 ====\n")
    
    # 일지 내용 생성
    diary_content = generate_diary_content(crop, weather, temperature, humidity, issues, work)
    if diary_content:
        print("=== 생성된 농가 일지 ===")
        print(diary_content)
    
    # 이미지 생성
    image_url = generate_farm_image(crop, weather, issues)
    if image_url:
        print("\n=== 생성된 이미지 URL ===")
        print(image_url)

# 테스트 케이스
test_cases = [
    ("토마토", "맑음", 25.5, 70.0, "잎이 조금 황변됨", "살균제 살포"),
    ("상추", "흐림", 22.0, 75.0, "벌레 피해 발견", "유기농 해충제 처리"),
    ("고추", "비", 23.5, 85.0, "곰팡이 징후", "환기 작업")
]

# 테스트 실행
for case in test_cases:
    create_farm_diary_entry(*case)
