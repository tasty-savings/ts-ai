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
import pandas as pd
from logger import logger

def get_recipe_info(recipe_info_index):
    """
        *** recipe_info_index 따라 레시피 정보를 선택합니다. (추후 레시피 DB 추가 시 수정 필요)
        레시피 정보를 가져오는 함수
        레시피 메뉴명, 재료, 인분, 조리시간, 타입, 만드는 방법, 조리 팁을 포함한 레시피 정보를 반환합니다.
        return: dict
    """
    df = pd.read_csv('data/result/10000recipe_all.csv')
    recipe_info = {}
    recipe_info["recipe_menu_name"] = df['title'][recipe_info_index]              # 레시피 메뉴명
    recipe_info["recipe_ingredients"] = df['ingredient'][recipe_info_index]       # 레시피 재료
    recipe_info["recipe_serving"] = df['servings'][recipe_info_index]             # 레시피 인분
    recipe_info["recipe_cooking_time"] = df['cooking_time'][recipe_info_index]    # 레시피 조리 시간
    recipe_info["recipe_type"] = df['type_key'][recipe_info_index]                # 레시피 타입(밑반찬, 차/음료/술 ...)
    recipe_info["recipe_cooking_order"] = df['cooking_order'][recipe_info_index]  # 레시피 만드는 방법
    recipe_info["recipe_tips"] = df['tips'][recipe_info_index]                    # 레시피 조리 팁

    logger.info("레시피 정보 읽기 완료: %s", recipe_info)
    return recipe_info

def get_system_prompt(recipe_change_type):
    """
        LLM에 넣을 시스템 프롬프트를 가져오는 함수
        각 프롬프트 파일은 src/prompt 폴더에 저장되어 있습니다.
        
        Args:
            recipe_change_type (int): 레시피 변환 기능에서 프롬프트를 바꾸기 위한 인덱스 
                (0: 기본값, 1: 냉장고 파먹기, 2: 레시피 단순화, 3: 사용자 영양 맞춤형 레시피)
        
        Returns:
            str: 시스템 프롬프트
    """
    default_folder_path = 'src/prompt/'
    if recipe_change_type == 1:
        prompt_file_path = 'fridge_recipe_transform_system_prompt.txt'
        logger.info("냉장고 파먹기 프롬프트 선택")
    elif recipe_change_type == 2:
        prompt_file_path = 'fridge_recipe_transform_user_prompt.txt'
        logger.info("레시피 단순화 프롬프트 선택")
    elif recipe_change_type == 3:
        prompt_file_path = 'fridge_recipe_transform_output_prompt.txt'
        logger.info("사용자 영양 맞춤형 레시피 프롬프트 선택")
    else:
        logger.error("recipe_change_type이 올바르지 않습니다. 1, 2, 3 중 하나를 입력해주세요.")

    try:
        with open(default_folder_path + prompt_file_path, 'r') as file:
            system_prompt = file.read()
            logger.info("System Prompt 읽기 완료")
    except FileNotFoundError:
        logger.error(f"{prompt_file_path}을 찾을 수 없습니다.")
        system_prompt = ""
        
    return system_prompt

def generate_recipe(recipe_info_index, user_info, recipe_change_type):
    """
    레시피를 생성하는 함수
    
    Args:
        recipe_info_index (int): 레시피 정보 인덱스
        user_info (dict): 사용자 정보
        recipe_change_type (int): 레시피 변환 기능에서 프롬프트를 바꾸기 위한 인덱스 
            (0: 기본값, 1: 냉장고 파먹기, 2: 레시피 단순화, 3: 사용자 영양 맞춤형 레시피)
    
    Returns:
        dict: 생성된 레시피 정보
    """
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

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(recipe_change_type)),
        ("user", "- user_info: {user_info}\n\n- recipe_info: {recipe_info}")
    ])
    logger.info("프롬프트 템플릿 생성 완료.")

    chain = prompt | llm | output_parser

    recipe_info = get_recipe_info(recipe_info_index)
    
    logger.info("LLM 레시피 생성중...")
    result = chain.invoke({"user_info": user_info, "recipe_info": recipe_info})
    logger.info("LLM 레시피 생성 완료")

    # log로 결과 출력
    pretty_recipe_info_json = json.dumps(recipe_info, indent=4, ensure_ascii=False)
    logger.debug("==========원래 레시피 정보==========\n%s", pretty_recipe_info_json)

    pretty_user_info_json = json.dumps(user_info, indent=4, ensure_ascii=False)
    logger.debug("==========유저 정보==========\n%s", pretty_user_info_json)

    pretty_result_json = json.dumps(result[0], indent=4, ensure_ascii=False)
    logger.debug("==========LLM 제안 레시피==========\n%s",pretty_result_json)
    
    return result[0]