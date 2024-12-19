from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import datetime
from recipe_change_origin import generate_recipe, get_user_info, get_recipe_data, get_system_prompt
from logger import logger_main
from typing import Optional
from langfuse import Langfuse
from langchain_core.prompts import PromptTemplate
import asyncio

app = FastAPI()
langfuse = Langfuse()

def get_system_prompt(recipe_change_type):
    """LLM 프롬프트를 가져오는 함수 (각 프롬프트 파일은 langfuse에 저장되어 있고, tracking 가능)"""
    if recipe_change_type==1:
        langfuse_prompt_name = "fridge_recipe_transform"
    elif recipe_change_type==2:
        langfuse_prompt_name = "simple_recipe_transform"
    elif recipe_change_type==3:
        langfuse_prompt_name = "balance_nutrition"
    else:
        logger_main.error("Langfuse Prompt Get Error")
        raise ValueError(f"지원하지 않는 recipe_change_type: {recipe_change_type}")
    
    # langfuse_text_prompt = langfuse.get_prompt(langfuse_prompt_name)
    # 캐시 ttl, 재시도, timeout 지정
    langfuse_text_prompt = langfuse.get_prompt(
        langfuse_prompt_name,
        cache_ttl_seconds=300,  # 캐시 TTL 300초 (default=60)
        max_retries=3,          # 최대 재시도 횟수 3회 (default=2)
        fetch_timeout_seconds=3 # API 호출 타임아웃 3초 (default=20)
    )
    
    langchain_text_prompt = PromptTemplate.from_template(
        langfuse_text_prompt.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_text_prompt},
    )

    logger_main.info("langfuse prompt template 생성 완료")
    return langchain_text_prompt

@app.get("/ai/health-check")
async def health_check(request: Request):
    client_ip = request.client.host
    return {
        'message': 'Hello My FastAPI World!',
        'time': datetime.datetime.now(),
        'client_ip': client_ip
    }

@app.post("/ai/recipe")
async def transform_recipe(
    request: Request,
    recipe_change_type: Optional[int] = 0,
    recipe_info_index: Optional[str] = "0"
):
    logger_main.debug(f"recipe_chage_type : {recipe_change_type}, recipe_info_index : {recipe_info_index}")
    
    if not request.headers.get("content-type") == "application/json":
        logger_main.error("Content-Type 헤더가 'application/json'이 아닙니다.")
        raise HTTPException(status_code=400, detail="Content-Type 헤더가 'application/json'이 아닙니다.")

    try:
        data = await request.json()
        logger_main.debug("body 정보 추출 완료 : %s", data)
        
        # user, recipe 정보 비동기로 가져옴
        user_info, recipe_info = await asyncio.gather(
            get_user_info(recipe_change_type, data),
            get_recipe_data(recipe_info_index)
        )
        
        # user, recipe 정보를 가져오면 레시피 생성
        result = await generate_recipe(recipe_info, user_info, recipe_change_type)
        return JSONResponse(content=result, status_code=200)
    
    except Exception as e:
        logger_main.error(f"에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# reload=True : 코드 변경 시 자동 재시작
if __name__ == "__main__":
    import sys
    import os
    
    # 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(ROOT_DIR)
    
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=5555, reload=True)