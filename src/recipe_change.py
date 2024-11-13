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
from logger import logger_recipe

def get_recipe_info(recipe_info_index):
    """
        *** recipe_info_index 따라 레시피 정보를 선택합니다. (추후 레시피 DB 추가 시 수정 필요)
        레시피 정보를 가져오는 함수
        
        Args:
            recipe_info_index (int): 레시피 정보 인덱스
            
        Returns:
            dict: 레시피 정보
    """
    df = pd.read_csv('data/result/recipe_analysis_sample.csv')
    recipe_info = {}
    recipe_info["recipe_menu_name"] = df['title'][recipe_info_index]              # 레시피 메뉴명
    recipe_info["recipe_ingredients"] = string_to_list(df['ingredient'][recipe_info_index])       # 레시피 재료
    recipe_info["recipe_serving"] = df['servings'][recipe_info_index]             # 레시피 인분
    recipe_info["recipe_cooking_time"] = df['cooking_time'][recipe_info_index]    # 레시피 조리 시간
    recipe_info["recipe_type"] = df['type_key'][recipe_info_index]                # 레시피 타입(밑반찬, 차/음료/술 ...)
    recipe_info["recipe_cooking_order"] = string_to_list(df['cooking_order'][recipe_info_index])  # 레시피 만드는 방법
    recipe_info["recipe_tips"] = df['tips'][recipe_info_index]                    # 레시피 조리 팁


    logger_recipe.info(f"{recipe_info_index}번 레시피 정보 검색 : {recipe_info["recipe_menu_name"]}")
    return recipe_info

def get_user_info(recipe_change_type, data):
    """
    요청 데이터에서 user_info를 추출하는 함수
    
    Args:
        recipe_change_type (int): 레시피 변환 기능에서 프롬프트를 바꾸기 위한 인덱스 
                (0: 기본값, 1: 냉장고 파먹기, 2: 레시피 단순화, 3: 사용자 영양 맞춤형 레시피)
        data (dict): 클라이언트 요청 데이터
        
    Returns:
        dict: 사용자 정보
    """
    if recipe_change_type == 1:
        user_info = {
            "user_allergy_ingredients": data.get('user_allergy_ingredients', []),
            "user_dislike_ingredients": data.get('user_dislike_ingredients', []),
            "user_spicy_level": data.get('user_spicy_level'),
            "user_cooking_level": data.get('user_cooking_level'),
            "user_owned_ingredients": data.get('user_owned_ingredients', []),
            "user_basic_seasoning": data.get('user_basic_seasoning', []),
            "must_use_ingredients": data.get('must_use_ingredients', [])
        }
    elif recipe_change_type == 2:
        user_info = {
            "user_allergy_ingredients": data.get('user_allergy_ingredients', []),
            "user_cooking_level": data.get('user_cooking_level')
        }
    logger_recipe.info("사용자 정보 추출 완료 : %s", user_info)
    return user_info

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
        prompt_file_path = '1_fridge_recipe_transform_system_prompt.txt'
    elif recipe_change_type == 2:
        prompt_file_path = '2_simple_recipe_transform_system_prompt.txt'
    elif recipe_change_type == 3:
        prompt_file_path = 'fridge_recipe_transform_output_prompt.txt'
    else:
        logger_recipe.error("recipe_change_type이 올바르지 않습니다. 1, 2, 3 중 하나를 입력해주세요.")

    try:
        with open(default_folder_path + prompt_file_path, 'r') as file:
            system_prompt = file.read()
            logger_recipe.info(f"system prompt : {prompt_file_path} 읽기 완료")
    except FileNotFoundError:
        logger_recipe.error(f"{prompt_file_path}을 찾을 수 없습니다.")
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
        timeout=20,
        api_key=OPENAI_API_KEY
    )
    logger_recipe.info("LLM 초기화 완료.")

    # 출력 파서 초기화
    output_parser = JsonOutputParser()
    logger_recipe.info("json 출력 파서 초기화 완료.")

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(recipe_change_type)),
        ("user", "- user_info: {user_info}\n\n- recipe_info: {recipe_info}")
    ])
    logger_recipe.debug("==========프롬프트==========\n%s",prompt)
    logger_recipe.info("프롬프트 템플릿 생성 완료.")
    
    chain = prompt | llm | output_parser

    recipe_info = get_recipe_info(recipe_info_index)
    
    logger_recipe.info("LLM 레시피 생성중...")
    result = chain.invoke({"user_info": user_info, "recipe_info": recipe_info})
    logger_recipe.info("LLM 레시피 생성 완료")
    logger_recipe.debug("LLM 레시피 생성 결과 : %s", result)
    
    # 결과가 리스트인 경우 첫 번째 항목을 사용, 아닌 경우 그대로 사용
    recipe_result = result[0] if isinstance(result, list) else result
    
    # recipe_ingredients, recipe_cooking_order을 list형으로 형변환
    recipe_result["recipe_ingredients"] = string_to_list(recipe_result["recipe_ingredients"])
    recipe_result["recipe_cooking_order"] = string_to_list(recipe_result["recipe_cooking_order"])
    
    # log로 결과 출력
    pretty_recipe_info_json = json.dumps(recipe_info, indent=4, ensure_ascii=False)
    logger_recipe.debug("==========원래 레시피 정보==========\n%s", pretty_recipe_info_json)

    pretty_user_info_json = json.dumps(user_info, indent=4, ensure_ascii=False)
    logger_recipe.debug("==========유저 정보==========\n%s", pretty_user_info_json)

    pretty_result_json = json.dumps(recipe_result, indent=4, ensure_ascii=False)
    logger_recipe.debug("==========LLM 제안 레시피==========\n%s", pretty_result_json)
    
    return recipe_result

def string_to_list(text):
    """
    레시피 결과값의 필드 타입을 list로 변환하는 함수
    
    Args:
        result(text): LLM이 생성한 레시피 결과값
        
    Returns:
        list: 타입이 변환된 레시피 결과값(result)
    """
    # 문자열이 이미 리스트인 경우 그대로 반환
    if isinstance(text, list):
        logger_recipe.info(f"레시피 변환 중... {text}는 List 형이므로 그대로 반환")
        return text
    
    # 문자열을 리스트로 변환
    result = eval(text)
    if isinstance(result, list):
        logger_recipe.info(f"레시피 변환 중... {text}를 List 형으로 변환")
        return result
    # 변환 실패시 log로 error를 찍고, 빈 리스트 반환
    else:
        logger_recipe.error(f"레시피 변환 중, {text}을 List형으로 바꾸는데 에러 발생. 빈 리스트 반환")
        return []