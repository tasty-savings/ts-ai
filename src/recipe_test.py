import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.config import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(api_key=OPENAI_API_KEY)
prompt = PromptTemplate.from_template(""" 
냉장고의 재료를 최대한 활용하여 자신만의 레시피를 만들려고 합니다.
레시피를 생성할 때는 유저의 특성과 취향, 메뉴명, 유통기한 임박한 식재료, 보유한 식재료를 고려하여 식사 레시피를 만들어주세요.
메뉴명, 재료, 만드는 방법, 고려한 유저 특성을 포함하여 식사 레시피를 만들어주세요.
                                      
[고려사항]
 1. 유저의 특성과 취향
 2. 메뉴명
 3. 유통기한 임박한 식재료
 4. 보유한 식재료
 5. 사람이 먹을 수 없는 재료는 요리에 사용하지 마십시오. (ex. 비누, 세제, 치약 등)
 6. 레시피는 요리 초보자도 가능하게 각 과정을 최대한 자세히 설명해주세요.
 7. 레시피의 예상 비용을 생성하여 주세요. (보유한 식재료의 비용은 반드시 제외해야합니다.)
 8. 필요한 재료는 1인분 기준이라고 명시 후, 양을 표시해주세요.
 9. 기본 조미료(설탕, 소금, 올리브유, 다시다, 미원, 식용유, 참기름, 들기름, 후추, 간장, 고추장, 된장, 식초, 물엿) 는 집에 있다고 가정합니다.
 10. 유저 특성이 [가난, 무지출 챌린지, 돈 없음, 소비 절약]과 같이 경제적인 상황이 고려되어야 하는 경우, 레시피의 예상 비용을 최대한 저렴하게 설정해주세요.
 11. 모든 레시피는 유저 특성을 고려해서 왜 이 레시피를 만드는지 설명해주세요.
                                      

유저의 특성: "{user_character}"
메뉴명: "{menu_name}"
유통기한 임박한 재료: "{expired_ingredients}"
보유한 재료: "{owned_ingredients}"

""")

chain = prompt | llm

print(chain.invoke({"user_character":"가난", "menu_name":"소불고기 파스타", "expired_ingredients":"오이", "owned_ingredients":"김치, 양파, 배추, 상추"}))