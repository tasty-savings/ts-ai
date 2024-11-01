import os
from dotenv import load_dotenv
import requests
import pandas as pd
import json
import re

load_dotenv()
food_safety_api_key = os.getenv('FOOD_SAFETY_API_KEY')

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
    type_key = recipe['RCP_PAT2']
    method_key = recipe['RCP_WAY2']
    main_img = recipe['ATT_FILE_NO_MAIN']
    ingredients, servings = process_food_safety_ingredients(recipe['RCP_PARTS_DTLS'])

    hashtag = []
    if recipe['HASH_TAG']:
        hashtag.append(recipe['HASH_TAG'])

    cooking_order = []
    cooking_img = []
    for i in range(1, 21):
        order_step = recipe.get(f'MANUAL{i:02d}', '').strip()
        img_step = recipe.get(f'MANUAL_IMG{i:02d}', '').strip()
        if order_step:
            step = re.sub(r'\w$', '', order_step).strip()
            step = step.replace('\n', ' ')
            cooking_order.append(step)
            cooking_img.append(img_step)

    tips = []
    if process_food_safety_tips(recipe.get('RCP_NA_TIP', '')):
        tips.append(process_food_safety_tips(recipe.get('RCP_NA_TIP', '')))

    return {
        'title': food_name,
        'intro': '',
        'main_img': main_img,
        'type_key': type_key,
        'situ_key': '',
        'ing_key': '',
        'method_key': method_key,
        'servings': servings,
        'cooking_time': '',
        'difficulty': '',
        'ingredient': json.dumps(ingredients, ensure_ascii=False),
        'cooking_order': json.dumps(cooking_order, ensure_ascii=False),
        'cooking_img': json.dumps(cooking_img, ensure_ascii=False),
        'views': 0,
        'hashtag': hashtag,
        'tips': tips
    }

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"{len(data)} 개의 레시피가 {filename} 파일로 저장되었습니다.")

def main():
    # 식품안전나라 API에서 레시피 가져오기
    food_safety_recipes = get_food_safety_recipes(food_safety_api_key)
    processed_food_safety_recipes = [process_food_safety_recipe(recipe) for recipe in food_safety_recipes]
    save_to_csv(processed_food_safety_recipes, 'jori_recipe1.csv')

if __name__ == "__main__":
    main()