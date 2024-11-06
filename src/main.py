from flask import Flask, request, jsonify
import datetime
from recipe_change import generate_recipe
from logger import logger

app = Flask(__name__)

@app.route("/health-check")
def hello():
    client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    return f'Hello My Flask World! <br><br> Now: {datetime.datetime.now()} <br><br> Your IP: {client_ip}'

@app.route("/recipe", methods=['POST'])
def transform_recipe():
    # 클라이언트 요청이 JSON 형식인지 확인 : Content-Type 헤더가 "application/json"일 경우 True를 반환
    if not request.is_json:
        logger.error("Content-Type 헤더가 'application/json'이 아닙니다.")
        return jsonify({"error": "Content-Type 헤더가 'application/json'이 아닙니다."}), 400
    
    # recipe_change_type, recipe_info_index를 query parameter로 받음
    recipe_change_type = request.args.get('recipe_change_type', default=0, type=int)
    recipe_info_index = request.args.get('recipe_info_index', default=0, type=int)
    
    try:
        # body에서 데이터 추출
        data = request.get_json()
        
        # user_info 데이터 추출
        user_info = {
            "user_allergy_ingredients": data.get('user_allergy_ingredients', []),
            "user_dislike_ingredients": data.get('user_dislike_ingredients', []),
            "user_spicy_level": data.get('user_spicy_level'),
            "user_cooking_level": data.get('user_cooking_level'),
            "user_owned_ingredients": data.get('user_owned_ingredients', []),
            "user_basic_seasoning": data.get('user_basic_seasoning', []),
            "must_use_ingredients": data.get('must_use_ingredients', [])
        }
        
        # 레시피 생성
        result = generate_recipe(recipe_info_index, user_info, recipe_change_type)
        
        # dict를 json으로 변환하여 반환
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"에러 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5555", debug=True)