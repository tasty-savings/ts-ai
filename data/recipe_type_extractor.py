import os
import json
import asyncio
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from config import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime

# 환경 변수 로드
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, df: pd.DataFrame, current_index: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"recipe_analysis_checkpoint_{timestamp}.csv"
        )
        df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')

        # 체크포인트 정보 저장
        metadata_path = os.path.join(self.checkpoint_dir, "checkpoint_metadata.json")
        metadata = {
            "last_checkpoint": checkpoint_path,
            "current_index": current_index,
            "timestamp": timestamp
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_latest_checkpoint(self) -> tuple[pd.DataFrame, int]:
        metadata_path = os.path.join(self.checkpoint_dir, "checkpoint_metadata.json")
        if not os.path.exists(metadata_path):
            return None, 0

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        checkpoint_path = metadata["last_checkpoint"]
        if os.path.exists(checkpoint_path):
            df = pd.read_csv(checkpoint_path, encoding='utf-8-sig')
            return df, metadata["current_index"]
        return None, 0

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)

async def analyze_recipe_async(row: pd.Series) -> List[str]:
    try:
        prompt = f"""
        다음 레시피를 분석하여 정확하고 구체적인 특징을 추출해주세요.

        레시피 이름: {row['title']}
        인분: {row['servings']}
        재료: {row['ingredient']}
        조리 방법: {row['cooking_order']}

        다음 내용을 참고하여 JSON 형식으로 응답해주세요:

        **레시피 유형에 포함될 수 있는 항목들**
         건강식, 고단백, 한식, 주식, 채식, 저칼로리, 반찬, 해산물, 간편식, 구이, 전통식, 고섬유, 새콤달콤한맛, 볶음, 생식, 저당식, 무침,
         양식, 매운맛1단계, 매운맛2단계, 매운맛3단계, 매운맛4단계, 매운맛5단계, 저염식, 퓨전, 담백한맛, 국, 찜, 디저트, 샐러드, 저탄수화물,
         가벼운, 튀김, 비타민, 조림, 소화촉진, 음료, 장건강, 체중조절, 중식, 부침, 무기질, 삶기, 특별식, 죽, 글루텐프리, 전, 일식, 스프

        주의사항:
        ***0. *레시피 유형에 포함될 수 있는 항목들*만 포함해주세요***
        1. 재료의 양과 인분수를 고려하여 1인분 기준으로 판단해주세요
        2. 실제 레시피의 재료와 조리법을 기반으로 판단해주세요
        3. 각 항목은 재료의 종류와 양에 맞게 구체적인 건강 및 영양 관련 특성 위주로 포함해주세요
            - 다음 조건을 모두 만족하는 경우 '고단백' 분류
                -- 음식의 총 칼로리 중 단백질이 차지하는 비율이 20% 이상
                -- 해당 단백질 공급원의 양이 1인분당 최소:
                    --- 육류/생선/콩류: 100g 이상
                    --- 달걀: 2개 이상
            - *고칼로리 재료(기름,버터,고탄수화물,견과류 등)가 사용된 요리는 '저칼로리' 제외*
            - 인분과 재료의 양을 기반으로 1인분(150g) 기준 300kcal 이하는 '저칼로리' 포함 
        4. 매운맛은 다음을 기준으로 분류해주세요
            -  기준: 매운맛 강도를 스코빌 지수(SHU)로 조절
                -- 매운맛1단계: 640 SHU (진라면 순한맛)
                -- 매운맛2단계: 2000 SHU (진라면 매운맛)
                -- 매운맛3단계: 3400 SHU (신라면)
                -- 매운맛4단계: 4400 SHU (불닭볶음면)
                -- 매운맛5단계: 9416 SHU (틈새라면)
        4. 확실한 특징만 포함해주세요
        6. 한 레시피에 여러 항목이 해당되면 모두 포함해주세요
        7. 해산물을 주재료로 하는 요리는 '해산물' 추가해주세요
        8. 일반적이거나 모순되는 분류는 피해주세요
            - 파스타/스테이크/리조또/크림소스/치즈 등 서양 조리법과 재료가 주된 요리는 '양식' 포함
            - 장류, 젓갈 등 고나트륨 재료가 과다 사용된 경우 '저염식' 제외
            - **육류/생선이 들어간 요리만 '채식' 제외**
            - 밀가루가 들어간 요리는 '글루텐프리' 제외
            

        JSON 응답 형식:
        {{"recipe_type": ["유형1", "유형2", "유형3"]}}
        """

        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "당신은 요리와 영양에 대한 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        result_text = response.choices[0].message.content
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(result_text)
        return result['recipe_type']

    except Exception as e:
        print(f"레시피 분석 중 오류 발생 ({row['title']}): {str(e)}")
        return ["분석 실패"]


async def process_batch(batch: pd.DataFrame, semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
    async with semaphore:
        tasks = []
        for _, row in batch.iterrows():
            task = analyze_recipe_async(row)
            tasks.append(task)
            # 요청 간 0.5초 지연
            await asyncio.sleep(0.5)
        return await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    # 체크포인트 매니저 초기화
    checkpoint_mgr = CheckpointManager()

    # 이전 체크포인트 확인
    df, start_index = checkpoint_mgr.load_latest_checkpoint()
    if df is None:
        # 새로 시작하는 경우
        df = pd.read_csv('banchan_samples7.csv')
        start_index = 0
    else:
        print(f"체크포인트에서 재시작: 인덱스 {start_index}부터 시작")

    # 동시 실행 제한 설정
    BATCH_SIZE = 5  # 한 번에 처리할 레시피 수
    MAX_CONCURRENT_REQUESTS = 3  # 동시 API 요청 수 제한
    CHECKPOINT_INTERVAL = 50  # 50개 레시피마다 체크포인트 저장
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 남은 데이터 처리
    remaining_df = df.iloc[start_index:]
    batches = [remaining_df[i:i + BATCH_SIZE] for i in range(0, len(remaining_df), BATCH_SIZE)]

    all_results = list(df['recipe_type'][:start_index]) if 'recipe_type' in df.columns else []

    try:
        for batch_idx, batch in enumerate(tqdm(batches, desc="레시피 분석 중")):
            batch_results = await process_batch(batch, semaphore)

            # 에러 처리된 결과 필터링
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    processed_results.append(["분석 실패"])
                else:
                    processed_results.append(result)

            all_results.extend(processed_results)

            # 현재까지의 결과를 데이터프레임에 업데이트
            df['recipe_type'] = pd.Series(all_results + [None] * (len(df) - len(all_results)))

            # 체크포인트 저장
            current_index = start_index + (batch_idx + 1) * BATCH_SIZE
            if (batch_idx + 1) % (CHECKPOINT_INTERVAL // BATCH_SIZE) == 0:
                print(f"\n체크포인트 저장 중... (인덱스: {current_index})")
                checkpoint_mgr.save_checkpoint(df, current_index)

            await asyncio.sleep(1)  # 배치 처리 후 대기

    except KeyboardInterrupt:
        print("\n프로그램 중단 감지. 현재 상태를 저장합니다...")
        current_index = start_index + len(all_results)
        checkpoint_mgr.save_checkpoint(df, current_index)
        print(f"중단 시점까지의 진행상황이 저장되었습니다. (인덱스: {current_index})")
        return

    # 최종 결과 저장
    df['recipe_type'] = all_results
    df.to_csv('banchan_samples7.csv', index=False, encoding='utf-8-sig')
    print("\n분석이 완료되었습니다!")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())