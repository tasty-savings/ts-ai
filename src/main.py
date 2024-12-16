from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import datetime
from recipe_change_origin import generate_recipe, get_user_info, get_recipe_data
from logger import logger_main
from typing import Optional
import asyncio
import sys
import os
import uvicorn

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

app = FastAPI()

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
    uvicorn.run("src.main:app", host="0.0.0.0", port=5555, reload=True)