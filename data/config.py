from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 조리식품레시피 API 키 설정
RECIPE_API = os.getenv("RECIPE_API")

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(OPENAI_API_KEY)