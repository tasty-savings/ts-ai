import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from utils import get_recipe_info, get_user_info, get_system_prompt
from logger import logger

# 모델 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=1000,
    timeout=10,
    api_key=OPENAI_API_KEY
)
logger.info("LLM 초기화 완료.")

# 출력 파서 초기화
output_parser = JsonOutputParser()
logger.info("json 출력 파서 초기화 완료.")

# 프롬프트 템플릿 생성 (1: 냉장고 파먹기, 2: 레시피 단순화, 3: 사용자 영양 맞춤형 레시피)
prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt(1)),
    ("user", "{user_info}, {recipe_info}")
])
logger.info("프롬프트 템플릿 생성 완료.")

chain = prompt | llm | output_parser

logger.info("LLM 레시피 생성중... : 냉장고 파먹기")
result = chain.invoke({"user_info" : get_user_info(), "recipe_info" : get_recipe_info()})
logger.info("LLM 레시피 생성 완료 : 냉장고 파먹기")

# 결과 출력
pretty_recipe_info_json = json.dumps(get_recipe_info(), indent=4, ensure_ascii=False)
logger.debug("==========원래 레시피 정보==========\n%s", pretty_recipe_info_json)

pretty_user_info_json = json.dumps(get_user_info(), indent=4, ensure_ascii=False)
logger.debug("==========유저 정보==========\n%s", pretty_user_info_json)

pretty_result_json = json.dumps(result[0], indent=4, ensure_ascii=False)
logger.debug("==========LLM 제안 레시피==========\n%s",pretty_result_json)