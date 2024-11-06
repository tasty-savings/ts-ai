import pandas as pd
from logger import logger

def get_recipe_info():
    """
        레시피 정보를 가져오는 함수
        레시피 메뉴명, 재료, 인분, 조리시간, 타입, 만드는 방법, 조리 팁을 포함한 레시피 정보를 반환합니다.
        return: dict
    """
    idx = 0
    df = pd.read_csv('data/result/10000recipe_all.csv')
    recipe_info = {}
    recipe_info["recipe_menu_name"] = df['title'][idx]              # 레시피 메뉴명
    recipe_info["recipe_ingredients"] = df['ingredient'][idx]       # 레시피 재료
    recipe_info["recipe_serving"] = df['servings'][idx]             # 레시피 인분
    recipe_info["recipe_cooking_time"] = df['cooking_time'][idx]    # 레시피 조리 시간
    recipe_info["recipe_type"] = df['type_key'][idx]                # 레시피 타입(밑반찬, 차/음료/술 ...)
    recipe_info["recipe_cooking_order"] = df['cooking_order'][idx]  # 레시피 만드는 방법
    recipe_info["recipe_tips"] = df['tips'][idx]                    # 레시피 조리 팁

    logger.info("레시피 정보 읽기 완료: %s", recipe_info)
    return recipe_info

def get_user_info():
    """
        유저 정보를 가져오는 함수
        유저의 알레르기 있는 식재료, 싫어하는 식재료, 매운 맛을 좋아하는 정도, 요리 숙련도, 보유한 재료, 기본 조미료, 반드시 사용해야 하는 식재료를 포함한 유저 정보를 반환합니다.
        return: dict
    """
    user_info = {}
    user_info["user_allergy_ingredients"] = "복숭아, 닭고기"        # 유저의 알레르기 있는 식재료
    user_info["user_dislike_ingredients"] = "닭고기"                # 유저가 싫어하는 식재료
    user_info["user_spicy_level"] = "2단계"                     # 매운 맛을 좋아하는 정도
    user_info["user_cooking_level"] = "초급"                    # 유저의 요리 숙련도
    user_info["user_owned_ingredients"] = "오이, 양파, 당근"    # 보유한 재료
    user_info["user_basic_seasoning"] = "식초, 소금, 설탕"      # 유저가 가진 기본 조미료
    user_info["must_use_ingredients"] = "비행기"                  # 반드시 사용해야 하는 식재료
    
    logger.info("유저 정보 읽기 완료: %s", user_info)
    return user_info

def get_system_prompt(prompt_index):
    """
        LLM에 넣을 시스템 프롬프트를 가져오는 함수
        각 프롬프트 파일은 src/prompt 폴더에 저장되어 있습니다.
        prompt_index에 따라 프롬프트 파일을 선택합니다. (1: 냉장고 파먹기, 2: 레시피 단순화, 3: 사용자 영양 맞춤형 레시피)
        return: str
    """
    default_folder_path = 'src/prompt/'
    if prompt_index == 1:
        prompt_file_path = 'fridge_recipe_transform_system_prompt.txt'
    elif prompt_index == 2:
        prompt_file_path = 'fridge_recipe_transform_user_prompt.txt'
    elif prompt_index == 3:
        prompt_file_path = 'fridge_recipe_transform_output_prompt.txt'

    try:
        with open(default_folder_path + prompt_file_path, 'r') as file:
            system_prompt = file.read()
            logger.info("System Prompt 읽기 완료: %s", system_prompt)
    except FileNotFoundError:
        logger.error(f"{prompt_file_path}을 찾을 수 없습니다.")
        system_prompt = ""
        
    return system_prompt
