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
from db import MongoDB
from bson import ObjectId

def get_recipe_data(recipe_info_index):
    """
        recipe_info_index 따라 레시피 정보를 가져오는 함수
        
        Args:
            int: 레시피 정보 인덱스
            
        Returns:
            dict: 레시피 정보
    """
    with MongoDB() as mongo_db:
        try:
            collection_name = "recipe"  # 컬렉션 이름 (필요에 따라 수정)
            query = { "_id": ObjectId(recipe_info_index) }

            projection = {"title": 1, "type_key": 1, "method_key":1, "servings" : 1, "cooking_time" : 1, "ingredients" : 1, "cooking_order": 1, "difficulty":1, "tips" : 1}

            recipe = mongo_db.find_one(collection_name, query, projection)
            if recipe:
                # ObjectId를 문자열로 변환
                recipe['_id'] = str(recipe['_id'])
                return recipe
            else:
                logger_recipe.error(f"{recipe_info_index} ID의 레시피를 찾을 수 없습니다.")
                return None
        except Exception as e:
            logger_recipe.error(f"get_recipe_data() 오류 발생: {e}")
            return None

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

    recipe_info = get_recipe_data(recipe_info_index)

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
        text: LLM이 생성한 레시피 결과값
        
    Returns:
        list: 타입이 변환된 레시피 결과값
    """
    # None이나 빈 값 체크
    if not text:
        logger_recipe.warning("빈 값이 입력되어 빈 리스트를 반환합니다.")
        return []
    
    # 이미 리스트인 경우 그대로 반환
    if isinstance(text, list):
        logger_recipe.info(f"레시피 변환 중... {text}는 List 형이므로 그대로 반환")
        return text
        
    # 딕셔너리인 경우 리스트로 감싸서 반환
    if isinstance(text, dict):
        logger_recipe.info(f"레시피 변환 중... 딕셔너리를 리스트로 변환")
        return [text]
    
    try:
        # 문자열이 아닌 경우 문자열로 변환
        if not isinstance(text, str):
            text = str(text)
            
        # 문자열의 시작과 끝의 대괄호 확인
        if not (text.startswith('[') and text.endswith(']')):
            text = f"[{text}]"
            
        # 작은따옴표를 큰따옴표로 변환
        text = text.replace("'", '"')
            
        # JSON 파싱을 통한 리스트 변환
        result = json.loads(text)
        if isinstance(result, list):
            logger_recipe.info("레시피 변환 중... 텍스트를 List 형으로 변환 성공")
            return result
        else:
            logger_recipe.warning("변환된 결과가 리스트가 아닙니다. 입력값을 리스트로 감싸서 반환")
            return [result]
            
    except json.JSONDecodeError as e:
        logger_recipe.error(f"JSON 파싱 중 에러 발생: {str(e)}")
        try:
            result = [item.strip() for item in text.strip('[]').split(',')]
            logger_recipe.info("쉼표로 구분된 문자열을 리스트로 변환 성공")
            return result
        except Exception as e:
            logger_recipe.error(f"문자열 분리 중 에러 발생: {str(e)}")
            return []
    except Exception as e:
        logger_recipe.error(f"레시피 변환 중 예상치 못한 에러 발생: {str(e)}")
        return []