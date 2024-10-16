import os
from dotenv import load_dotenv
import requests
import pandas as pd
import time
import json
import re

load_dotenv()
food_safety_api_key = os.getenv('FOOD_SAFETY_API_KEY')
recipe_api_key = os.getenv('RECIPE_API_KEY')

def get_food_safety_recipes(api_key):
    all_recipes = []
    start = 1
    end = 1000  # API의 최대 요청 가능 레코드 수

    while True:
        url = f"http://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}/"
        response = requests.get(url)
        data = response.json()

        if 'COOKRCP01' not in data or 'row' not in data['COOKRCP01']:
            break

        recipes = data['COOKRCP01']['row']
        all_recipes.extend(recipes)

        if len(recipes) < end - start + 1:
            break

        start += 1000
        end += 1000

    return all_recipes

def process_food_safety_ingredients(ingredients_raw):
    ingredients = []
    servings = 1

    for line in ingredients_raw.split('\n'):
        line = line.strip()
        if not line:
            continue

        servings_match = re.search(r'(\d+)인분', line)
        if servings_match:
            servings = int(servings_match.group(1))
            continue

        ingredients.append(line)

    return ingredients, servings

def process_food_safety_tips(tips_raw):
    tips = tips_raw.replace('•', '').replace('<br>', ' ').replace('\n', ' ')
    tips = re.sub(r'\s+', ' ', tips).strip()
    return tips

def process_food_safety_recipe(recipe):
    food_name = recipe['RCP_NM']
    recipe_type = recipe['RCP_PAT2']

    ingredients, servings = process_food_safety_ingredients(recipe['RCP_PARTS_DTLS'])

    cooking_order = []
    for i in range(1, 21):
        step = recipe.get(f'MANUAL{i:02d}', '').strip()
        if step:
            step = re.sub(r'\w$', '', step).strip()
            step = step.replace('\n', ' ')
            cooking_order.append(step)

    views = 0
    tips = process_food_safety_tips(recipe.get('RCP_NA_TIP', ''))

    return {
        'food_name': food_name,
        'type': recipe_type,
        'servings': servings,
        'ingredients': json.dumps(ingredients, ensure_ascii=False),
        'cooking_order': json.dumps(cooking_order, ensure_ascii=False),
        'views': views,
        'tips': tips
    }

def get_recipe_list(api_key, start_row, end_row):
    url = f"http://211.237.50.150:7080/openapi/{api_key}/json/Grid_20150827000000000226_1/{start_row}/{end_row}"
    response = requests.get(url)
    return response.json()["Grid_20150827000000000226_1"]["row"]

def get_recipe_steps(api_key, recipe_id):
    url = f"http://211.237.50.150:7080/openapi/{api_key}/json/Grid_20150827000000000228_1/1/1000?RECIPE_ID={recipe_id}"
    response = requests.get(url)
    return response.json()["Grid_20150827000000000228_1"]["row"]

def get_recipe_ingredients(api_key, recipe_id):
    url = f"http://211.237.50.150:7080/openapi/{api_key}/json/Grid_20150827000000000227_1/1/1000?RECIPE_ID={recipe_id}"
    response = requests.get(url)
    return response.json()["Grid_20150827000000000227_1"]["row"]

def process_recipe_api_recipes(api_key):
    recipes = []
    total_recipes = 537  # 총 레시피 수
    batch_size = 20  # 한 번에 가져올 레시피 수

    for start_row in range(1, total_recipes + 1, batch_size):
        end_row = min(start_row + batch_size - 1, total_recipes)
        recipe_list = get_recipe_list(api_key, start_row, end_row)

        for recipe in recipe_list:
            recipe_id = recipe["RECIPE_ID"]
            food_name = recipe["RECIPE_NM_KO"]
            food_type = recipe["TY_NM"]
            servings = re.sub(r'[^0-9]', '', recipe["QNT"])

            steps = get_recipe_steps(api_key, recipe_id)
            ingredients = get_recipe_ingredients(api_key, recipe_id)

            cooking_order = [f"{i+1}. {step['COOKING_DC']}" for i, step in enumerate(steps)]
            ingredients_list = [f"{ing['IRDNT_NM']}({ing['IRDNT_CPCTY']})" for ing in ingredients]

            recipes.append({
                "food_name": food_name,
                "type": food_type,
                "servings": servings,
                "ingredients": json.dumps(ingredients_list, ensure_ascii=False),
                "cooking_order": json.dumps(cooking_order, ensure_ascii=False),
                "views": '0',
                "tips": ""
            })

        print(f"Processed recipes {start_row} to {end_row}")
        time.sleep(1)  # API 요청 간 1초 대기

    return recipes

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"{len(data)} 개의 레시피가 {filename} 파일로 저장되었습니다.")

def main():
    # 식품안전나라 API에서 레시피 가져오기
    food_safety_recipes = get_food_safety_recipes(food_safety_api_key)
    processed_food_safety_recipes = [process_food_safety_recipe(recipe) for recipe in food_safety_recipes]
    save_to_csv(processed_food_safety_recipes, 'food_safety_recipes.csv')

    # 레시피 API에서 레시피 가져오기
    recipe_api_recipes = process_recipe_api_recipes(recipe_api_key)
    save_to_csv(recipe_api_recipes, 'recipe_api_recipes.csv')

    # # 두 API에서 가져온 레시피 합치기 및 저장 (선택적)
    # all_recipes = processed_food_safety_recipes + recipe_api_recipes
    # save_to_csv(all_recipes, 'combined_recipes.csv')

if __name__ == "__main__":
    main()