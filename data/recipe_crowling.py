import re
import logging
import asyncio
import pycurl
import json
from io import BytesIO
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import certifi
from typing import List, Dict, Optional
from urllib.parse import urljoin

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
fh = logging.FileHandler(f'scraping_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

save_recipe_fname = './10000recipe1_sample.csv'

by_type = {'밑반찬': '63', '메인반찬': '56', '국/탕': '54', '찌개': '55', '디저트': '60', '면/만두': '53',
           '밥/죽/떡': '52', '퓨전': '61', '양념/잼/소스': '58', '양식': '65', '샐러드': '64', '스프': '68',
           '빵': '66', '과자': '69', '차/음료/술': '59'}
by_situation = {'일상': '12', '초스피드': '18', '손님접대': '13', '술안주': '19', '다이어트': '21',
                '도시락': '15', '영양식': '43', '간식': '17', '야식': '45', '명절': '44'}
by_ingredient = {'소고기': '70', '돼지고기': '71', '닭고기': '72', '육류': '23', '채소류': '28', '해물류': '24',
                 '달걀/유제품': '50', '쌀': '47', '밀가루': '32', '건어물류': '25', '버섯류': '31', '과일류': '48',
                 '곡류': '26'}
by_method = {'볶음': '6', '끓이기': '1', '부침': '7', '조림': '36', '무침': '41', '비빔': '42',
             '찜': '8', '절임': '10', '튀김': '9', '삶기': '38', '굽기': '67', '회': '37'}

recipe_idx = 1
list4df = []


class RecipeCrawler:
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.headers = [
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept: text/html,application/xhtml+xml",
            "Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
        ]

    def safe_select(self, soup, selector, default=''):
        try:
            element = soup.select_one(selector)
            return element.text.strip() if element else default
        except AttributeError:
            return default

    async def fetch_url(self, url: str) -> Optional[str]:
        buffer = BytesIO()
        curl = None

        try:
            async with self.semaphore:
                curl = pycurl.Curl()
                curl.setopt(pycurl.URL, url)
                curl.setopt(pycurl.WRITEDATA, buffer)
                curl.setopt(pycurl.HTTPHEADER, self.headers)
                curl.setopt(pycurl.CAINFO, certifi.where())
                curl.setopt(pycurl.FOLLOWLOCATION, 1)
                curl.setopt(pycurl.MAXREDIRS, 5)
                curl.setopt(pycurl.TIMEOUT, 30)
                curl.setopt(pycurl.CONNECTTIMEOUT, 10)

                await asyncio.get_event_loop().run_in_executor(None, curl.perform)

                return buffer.getvalue().decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

        finally:
            if curl is not None:
                curl.close()
            if buffer is not None:
                buffer.close()

    async def process_recipe(self, recipe_url: str, type_key: str, situ_key: str,
                             ing_key: str, method_key: str) -> Optional[List]:
        try:
            html_content = await self.fetch_url(recipe_url)
            if not html_content:
                return None

            # 메인 이미지
            main_img = ''
            main_img_pattern = r'<div class="centeredcrop">\s*<img[^>]*src="([^"]+)"'
            main_img_match = re.search(main_img_pattern, html_content)
            if main_img_match:
                main_img = main_img_match.group(1)

            # ld+json에서 이미지 URL 추출
            json_pattern = r'<script type="application/ld\+json">(.*?)</script>'
            json_match = re.search(json_pattern, html_content, re.DOTALL)

            cooking_img = []
            if json_match:
                try:
                    recipe_data = json.loads(json_match.group(1))
                    instructions = recipe_data.get('recipeInstructions', [])
                    for instruction in instructions:
                        if isinstance(instruction, dict) and 'image' in instruction:
                            img_url = instruction['image']
                            if img_url and isinstance(img_url, str) and img_url.strip():
                                cooking_img.append(img_url)
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Failed to parse JSON from {recipe_url}: {str(e)}")

            # 이미지가 없으면 레시피 제외
            if not cooking_img:
                return None

            soup_r = BeautifulSoup(html_content, 'html.parser')

            # 필수 데이터 확인
            title_elem = soup_r.select_one('div.view2_summary.st3 > h3')
            if not title_elem:
                return None

            title = title_elem.text.strip()
            if not title:
                return None

            ingredient_div = soup_r.find('div', class_='ready_ingre3', id='divConfirmedMaterialArea')
            if not ingredient_div:
                return None

            # 기본 데이터
            intro = soup_r.select_one('#recipeIntro')
            intro = intro.text.strip() if intro else ''

            views_elem = soup_r.select_one('.view_cate_num .hit.font_num')
            views = ''.join(filter(str.isdigit, views_elem.text.strip())) if views_elem else '0'

            # 요리 정보
            summary_info = soup_r.select('div.view2_summary_info span')
            servings = ''
            cooking_time = ''
            difficulty = ''

            for info in summary_info:
                class_name = info.get('class', [])
                if 'view2_summary_info1' in class_name:
                    servings = re.sub(r'[^0-9]', '', info.text.strip()) + '인분'
                elif 'view2_summary_info2' in class_name:
                    cooking_time = info.text.strip()
                elif 'view2_summary_info3' in class_name:
                    difficulty = info.text.strip()

            # 재료
            ingredients = []
            for li in ingredient_div.select('ul > li'):
                name_div = li.select_one('div.ingre_list_name')
                amount_span = li.select_one('span.ingre_list_ea')
                if name_div:
                    name = name_div.text.strip().split('      ')[0]
                    amount = amount_span.text.strip() if amount_span else ''
                    ingredients.append(f"{name}({amount})" if amount else name)

            # 해시태그
            hashtags = []
            for tag in soup_r.select('div.view_step > div.view_tag > a'):
                tag_text = tag.text.strip()
                if tag_text.startswith('#'):
                    hashtags.append(tag_text[1:])

            # 팁
            tips = []
            for tip in soup_r.select('div.view_step > dl.view_step_tip > dd'):
                tip_text = re.sub(r'\s+', ' ', tip.text.strip())
                if tip_text:
                    tips.append(tip_text)

            # 조리과정
            cooking_steps = []
            for step_div in soup_r.select('div.view_step_cont'):
                media_body = step_div.select_one('div.media-body')
                if media_body:
                    # 불필요한 요소 제거
                    for element in media_body.select('.step_add.add_tool'):
                        element.decompose()
                    step_text = media_body.get_text(strip=True)
                    if step_text:
                        cooking_steps.append(step_text)

            if not cooking_steps:
                return None

            return [
                title, intro,main_img, type_key, situ_key, ing_key, method_key,
                servings, cooking_time, difficulty, ingredients,
                cooking_steps, cooking_img, views, hashtags, tips
            ]

        except Exception as e:
            logger.error(f"Error processing recipe: {recipe_url}")
            logger.error(f"Error message: {str(e)}")
            return None


    async def process_page(self, page_url: str, type_key: str, situ_key: str,
                           ing_key: str, method_key: str) -> List:
        html_content = await self.fetch_url(page_url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        sources = soup.select('#contents_area_full > ul > ul > li > div.common_sp_thumb > a')

        tasks = []
        for source in sources:
            recipe_url = urljoin('https://www.10000recipe.com', source['href'])
            task = self.process_recipe(recipe_url, type_key, situ_key, ing_key, method_key)
            tasks.append(task)

        return [result for result in await asyncio.gather(*tasks) if result]


async def main():
    global recipe_idx, list4df
    crawler = RecipeCrawler(max_concurrent=10)

    for type_key, type_value in by_type.items():
        for situ_key, situ_value in by_situation.items():
            for ing_key, ing_value in by_ingredient.items():
                for method_key, method_value in by_method.items():
                    main_url = (f'https://www.10000recipe.com/recipe/list.html?'
                                f'q=&query=&cat1={method_value}&cat2={situ_value}&'
                                f'cat3={ing_value}&cat4={type_value}&order=reco')

                    try:
                        html_content = await crawler.fetch_url(main_url)
                        if not html_content:
                            continue

                        soup = BeautifulSoup(html_content, 'html.parser')
                        page_len = len(soup.select('#contents_area_full > ul > nav > ul > li'))

                        tasks = []
                        for page in range(1, page_len + 1):
                            page_url = f"{main_url}&page={page}" if page != 1 else main_url
                            task = crawler.process_page(page_url, type_key, situ_key, ing_key, method_key)
                            tasks.append(task)

                        results = await asyncio.gather(*tasks)
                        for page_results in results:
                            for result in page_results:
                                list4df.append(result)
                                recipe_idx += 1

                                if recipe_idx % 50 == 0:
                                    recipe_df = pd.DataFrame(
                                        list4df,
                                        columns=['title', 'intro', 'main_img', 'type_key', 'situ_key', 'ing_key',
                                                 'method_key', 'servings', 'cooking_time', 'difficulty',
                                                 'ingredient', 'cooking_order', 'cooking_img', 'views',
                                                 'hashtag', 'tips']
                                    )
                                    recipe_df.to_csv(save_recipe_fname, encoding='utf-8', index=False)
                                    logger.info(f"Saved {recipe_idx} recipes to CSV")

                    except Exception as e:
                        logger.error(f"Error processing category: {type_key}, {situ_key}, {ing_key}, {method_key}")
                        logger.error(f"Error message: {str(e)}")
                        continue

    # 최종 결과 저장
    recipe_df = pd.DataFrame(
        list4df,
        columns=['title', 'intro', 'main_img', 'type_key', 'situ_key', 'ing_key', 'method_key',
                 'servings', 'cooking_time', 'difficulty', 'ingredient',
                 'cooking_order', 'cooking_img', 'views', 'hashtag', 'tips']
    )
    recipe_df.to_csv(save_recipe_fname, encoding='utf-8', index=False)
    logger.info("Scraping completed. Final results saved.")


if __name__ == "__main__":
    asyncio.run(main())