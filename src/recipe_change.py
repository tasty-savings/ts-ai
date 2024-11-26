from config import OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from logger import logger_recipe
from db import MongoDB
from bson import ObjectId
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, Field

def langfuse_tracking():
    """
        langchain 콜백 시스템을 사용한 langchain 실행 추적

        Returns:
            CallbackHandler: LangFuse의 실행 추적을 위한 CallbackHandler 객체.
    """
    langfuse_handler = CallbackHandler(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )
    return langfuse_handler

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

            projection = {"title": 1, "type_key": 1, "method_key":1, "servings" : 1, "cooking_time" : 1, "ingredients" : 1, "cooking_steps": 1, "difficulty":1, "tips" : 1}

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
        data (dict): 사용자 데이터
        
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
        각 프롬프트 파일은 langfuse에 저장되어 있고, tracking 가능
        
        Args:
            recipe_change_type (int): 레시피 변환 기능에서 프롬프트를 바꾸기 위한 인덱스 
                (0: 기본값, 1: 냉장고 파먹기, 2: 레시피 단순화, 3: 사용자 영양 맞춤형 레시피)
        
        Returns:
            str: 시스템 프롬프트
    """
    langfuse = Langfuse()

    if recipe_change_type==1:
        langfuse_text_prompt = langfuse.get_prompt("fridge_recipe_transform")
    elif recipe_change_type==2:
        langfuse_text_prompt = langfuse.get_prompt("simple_recipe_transform")
    elif recipe_change_type==3:
        langfuse_text_prompt = langfuse.get_prompt("fridge_recipe_transform")
    else:
        logger_recipe.error("Langfuse Prompt Get Error")
        raise ValueError(f"지원하지 않는 recipe_change_type: {recipe_change_type}")

    langchain_text_prompt = PromptTemplate.from_template(
        langfuse_text_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_text_prompt},
    )

    logger_recipe.info("프롬프트 템플릿 생성 완료.")
    return langchain_text_prompt

# 원하는 데이터 구조를 정의합니다.
class ChangeRecipe(BaseModel):
    main_changes_from_original_recipe: str = Field(description="기본 레시피와 새로운 레시피 사이의 주요 변경점")
    reason_for_changes: str = Field(description="레시피가 바뀐 이유")
    recipe_cooking_order: list = Field(description="조리 순서")
    recipe_cooking_time: str = Field(description="조리 시간")
    recipe_difficulty: str = Field(description="조리 난이도")
    recipe_ingredients: list = Field(description="조리에 사용되는 재료(양)")
    recipe_menu_name: str = Field(description="새로운 레시피의 이름")
    recipe_tips: str = Field(description="조리팁")
    recipe_type: str = Field(description="조리 타입")
    unchanged_parts_and_reasons: str = Field(description="기존 레시피에서 바뀌지 않은 부분과 바뀌지 않은 이유")


def generate_recipe(recipe_info, user_info, recipe_change_type):
    """
    레시피를 생성하는 함수
    
    Args:
        recipe_info, user_info (dict): 레시피 정보, 사용자 정보
        recipe_change_type (int): 레시피 변환 기능에서 프롬프트를 바꾸기 위한 인덱스 
            (0: 기본값, 1: 냉장고 파먹기, 2: 레시피 단순화, 3: 사용자 영양 맞춤형 레시피)
    
    Returns:
        dict: 생성된 레시피 정보
    """
    # langchain 콜백 시스템을 사용한 langchain 실행 추적.
    langfuse_handler = langfuse_tracking()
    
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
    output_parser = JsonOutputParser(pydantic_object=ChangeRecipe)
    logger_recipe.info("json 출력 파서 초기화 완료.")

    # 프롬프트 템플릿에 출력파서 지정
    prompt = get_system_prompt(recipe_change_type)
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())

    chain = prompt | llm | output_parser
    logger_recipe.info("LLM 레시피 생성 중...")
    result = chain.invoke({"user_info": user_info, "recipe_info": recipe_info}, config={"callbacks": [langfuse_handler]})
    logger_recipe.info("LLM 레시피 생성 완료")
    
    # 결과가 리스트인 경우 첫 번째 항목을 사용, 아닌 경우 그대로 사용
    recipe_result = result[0] if isinstance(result, list) else result
    
    return recipe_result
