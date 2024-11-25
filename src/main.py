from flask import Flask, request, jsonify
from typing import Optional
import threading
import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from recipe_change import generate_recipe, get_user_info
from recipe_recommend import recommend_recipes
from logger import logger_main


class EmbeddingsManager:
    _instance: Optional['EmbeddingsManager'] = None
    _lock = threading.Lock()
    _embeddings: Optional[HuggingFaceEmbeddings] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2"
                    )
        return self._embeddings


app = Flask(__name__)
embeddings_manager = EmbeddingsManager()

def get_embeddings():
    return embeddings_manager.get_embeddings()

@app.route("/ai/health-check")
def hello():
    client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    return f'Hello My Flask World! <br><br> Now: {datetime.datetime.now()} <br><br> Your IP: {client_ip}'

@app.route("/ai/recipe", methods=['POST'])
def transform_recipe():
    recipe_change_type = request.args.get('recipe_change_type', default=0, type=int)
    recipe_info_index = request.args.get('recipe_info_index', default=0, type=str)
    logger_main.debug(f"recipe_chage_type : {recipe_change_type}, recipe_info_index : {recipe_info_index}")
    if not request.is_json:
        logger_main.error("Content-Type 헤더가 'application/json'이 아닙니다.")
        return jsonify({"error": "Content-Type 헤더가 'application/json'이 아닙니다."}), 400

    try:
        data = request.get_json()
        logger_main.debug("body 정보 추출 완료 : %s", data)
        user_info = get_user_info(recipe_change_type, data)
        result = generate_recipe(recipe_info_index, user_info, recipe_change_type)
        # dict를 json으로 변환하여 반환
        return jsonify(result), 200
    
    except Exception as e:
        logger_main.error(f"에러 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/ai/recommend", methods=['POST'])
def recommend_recipe():
    try:
        data = request.get_json()
        logger_main.debug("body 정보 추출 완료 : %s", data)
        result = recommend_recipes(data.get('search_types', []), get_embeddings())
        
        return jsonify(result), 200
    
    except Exception as e:
        logger_main.error(f"에러 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5555", debug=True)

    