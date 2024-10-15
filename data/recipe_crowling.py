import re
import logging
import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from aiohttp_retry import RetryClient, ExponentialRetry

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
fh = logging.FileHandler(f'scraping_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

save_recipe_fname = './10000recipe.csv'

by_type = {'밑반찬': '63', '메인반찬': '56', '국/탕': '54', '찌개': '55', '디저트': '60', '면/만두': '53',
           '밥/죽/떡': '52', '퓨전': '61', '양념/잼/소스': '58', '양식': '65', '샐러드': '64', '스프': '68',
           '빵': '66', '과자': '69', '차/음료/술': '59'}  # cat4
by_situation = {'일상': '12', '초스피드': '18', '손님접대': '13', '술안주': '19', '다이어트': '21',
                '도시락': '15', '영양식': '43', '간식': '17', '야식': '45', '명절': '44'}  # cat2
by_ingredient = {'소고기': '70', '돼지고기': '71', '닭고기': '72', '육류': '23', '채소류': '28', '해물류': '24',
                 '달걀/유제품': '50', '쌀': '47', '밀가루': '32', '건어물류': '25', '버섯류': '31', '과일류': '48',
                 '곡류': '26'}  # cat3
by_method = {'볶음': '6', '끓이기': '1', '부침': '7', '조림': '36', '무침': '41', '비빔': '42',
             '찜': '8', '절임': '10', '튀김': '9', '삶기': '38', '굽기': '67', '회': '37'}  # cat1

recipe_idx = 1
list4df = []


def safe_select(soup, selector, default=None):
    try:
        return soup.select_one(selector).text.strip()
    except AttributeError:
        return default


async def fetch_with_retry(session, url):
    retry_options = ExponentialRetry(attempts=3)
    retry_client = RetryClient(client_session=session, retry_options=retry_options)
    async with retry_client.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
        return await response.text()


async def process_recipe(session, recipe_url, type_key, situ_key, ing_key, method_key):
    try:
        response_text = await fetch_with_retry(session, recipe_url)
        soup_r = BeautifulSoup(response_text, 'html.parser')

        title = safe_select(soup_r, 'div.view2_summary.st3 > h3')
        views_elem = soup_r.select('.view_cate_num .hit.font_num')
        if views_elem:
            views = views_elem[0].text.strip()
            views = ''.join(filter(str.isdigit, views))  # 숫자만 추출
        else:
            views = '0'
        servings = safe_select(soup_r, 'div.view2_summary.st3 > div.view2_summary_info > span.view2_summary_info1')
        if servings:
            servings = re.sub(r'[^0-9]', '', servings) + '인분'
        else:
            return None

        ingredient_div = soup_r.find('div', class_='ready_ingre3', id='divConfirmedMaterialArea')
        ingredient = []
        if ingredient_div:
            for li in ingredient_div.select('ul > li'):
                name_div = li.select_one('div.ingre_list_name')
                amount_span = li.select_one('span.ingre_list_ea')
                if name_div:
                    name = name_div.text.strip().split('      ')[0]
                    amount = amount_span.text.strip() if amount_span else ''
                    ingredient.append(f"{name}({amount})" if amount else name)

        if not ingredient:
            return None

        cooking_order = []
        for step_div in soup_r.select('div.view_step_cont'):
            media_body = step_div.select_one('div.media-body')
            if media_body:
                for element in media_body.select('.step_add.add_tool'):
                    element.decompose()
                step_text = media_body.get_text(strip=True)
                cooking_order.append(step_text)

        if cooking_order:
            return [title, type_key, servings, ingredient, cooking_order, views, '']
        else:
            return None

    except Exception as e:
        logger.error(f"Error processing recipe: {recipe_url}")
        logger.error(f"Error message: {str(e)}")
        return None


async def process_page(session, page_url, type_key, situ_key, ing_key, method_key):
    response_text = await fetch_with_retry(session, page_url)
    soup = BeautifulSoup(response_text, 'html.parser')

    sources = soup.select('#contents_area_full > ul > ul > li > div.common_sp_thumb > a')
    tasks = []
    for source in sources:
        recipe_url = 'https://www.10000recipe.com' + source['href']
        task = process_recipe(session, recipe_url, type_key, situ_key, ing_key, method_key)
        tasks.append(task)

    return await asyncio.gather(*tasks)


async def main():
    global recipe_idx, list4df

    async with aiohttp.ClientSession() as session:
        for type_key, type_value in by_type.items():
            for situ_key, situ_value in by_situation.items():
                for ing_key, ing_value in by_ingredient.items():
                    for method_key, method_value in by_method.items():
                        main_url = f'https://www.10000recipe.com/recipe/list.html?q=&query=&cat1={method_value}&cat2={situ_value}&cat3={ing_value}&cat4={type_value}&fct=&order=reco&lastcate=cat4&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource='

                        try:
                            response_text = await fetch_with_retry(session, main_url)
                            soup = BeautifulSoup(response_text, 'html.parser')
                            page_len = len(soup.select('#contents_area_full > ul > nav > ul > li'))

                            tasks = []
                            for page in range(1, page_len + 1):
                                page_url = main_url + f'&page={page}' if page != 1 else main_url
                                task = process_page(session, page_url, type_key, situ_key, ing_key, method_key)
                                tasks.append(task)

                            results = await asyncio.gather(*tasks)
                            for page_results in results:
                                for result in page_results:
                                    if result:
                                        list4df.append(result)
                                        recipe_idx += 1

                                        if recipe_idx % 50 == 0:
                                            recipe_df = pd.DataFrame(list4df,
                                                                     columns=['food_name', 'type', 'servings','ingredients',
                                                                              'cooking_order', 'views', 'tips'])
                                            recipe_df.to_csv(save_recipe_fname, encoding='utf-8', index=False)
                                            logger.info(f"Saved {recipe_idx} recipes to CSV")

                        except Exception as e:
                            logger.error(f"Error processing category: {type_key}, {situ_key}, {ing_key}, {method_key}")
                            logger.error(f"Error message: {str(e)}")
                            continue

    recipe_df = pd.DataFrame(list4df,
                             columns=['food_name', 'type', 'servings', 'ingredients', 'cooking_order', 'views', 'tips'])
    recipe_df.to_csv(save_recipe_fname, encoding='utf-8', index=False)
    logger.info("Scraping completed. Final results saved.")


if __name__ == "__main__":
    asyncio.run(main())