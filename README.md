
# 맛있는 절약 AI 🥕


![Python](https://img.shields.io/badge/python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/mongodb-47A248.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![OpenAI](https://img.shields.io/badge/openai-412991.svg?style=for-the-badge&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/langchain-1C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/faiss-0467DF.svg?style=for-the-badge&logo=meta&logoColor=white)
![LangGraph](https://img.shields.io/badge/langgraph-1C3C3C.svg?style=for-the-badge)
![Langfuse](https://img.shields.io/badge/langfuse-DD1100.svg?style=for-the-badge)


## ⭐ Main Feature
### 레시피 추천
- 만개의레시피 크롤링 및 식품안전나라 API 레시피 데이터 수집
- LLM을 활용하여 1만8천여개 레시피 데이터 라벨링
- **Hybrid Search**( RAG + Exact Tag Match )를 활용한 레시피 추천 기능 제공

### LLM 레시피 변형
- 냉장고 파먹기 : 사용자 보유 식재료, 기피 식재료, 사용하고 싶은 식재료, 요리 숙련도, 선호 맵기를 반영한 변형
- 쉽게 만들기 : 레시피 재료 및 조리 단계 간소화
- 영양맞춤 : 사용자 연령, 활동성, 성별, 섭취 이력을 반영하여 보건복지부 지정 필수 영양소(식품구성자전거)에 맞춘 영양 균형 레시피 변형

## 🔧 Stack
- **Language**: Python
- **Framework** : FastAPI
- **Library** : OpenAI, LangChain + LangGraph
- **VectorDB** : FAISS

## :open_file_folder: Project Structure

```markdown
├── data
│   ├── analysis
│   │   ├── 10000_analysis.ipynb
│   │   ├── 10000_change.ipynb
│   │   └── jori_analysis.ipynb
│   ├── api_recipe_collecting.py
│   ├── combine_recipes.ipynb
│   ├── jori_recipe_api.py
│   ├── recipe_crowling.py
│   ├── recipe_index
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── recipe_type_extractor.py
│   ├── result
│   │   ├── 1124_땅콩호박타르트.json
│   │   ├── 1124_새우두부계란찜.json
│   │   └── recipe_analysis_sample.csv
│   └── type_embeddings.npy
│
├── src
│   ├── config.py
│   ├── db.py
│   ├── evaluation
│   │   ├── combined_dataset.json
│   │   ├── generate_dataset.ipynb
│   │   ├── recipeinfo_example_dataset.csv
│   │   └── userinfo_example_dataset.csv
│   ├── main.py
│   ├── recipe_change.py
│   ├── recipe_change_origin.py
│   └── recipe_recommend.py
```


## 👨‍👩‍👧‍👦 Developer
*  **김민지** ([minji0916](https://github.com/minji0916))
*  **탁하선** ([xkr2990s](https://github.com/xkr2990s))
