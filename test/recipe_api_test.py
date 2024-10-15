from config import RECIPE_API
import requests
import json

menu = "떡볶이"
parts = "토마토"
# 기본 API 요청 URL (필요한 부분을 자신의 API 정보로 변경)
#url = f"http://openapi.foodsafetykorea.go.kr/api/{RECIPE_API}/COOKRCP01/json/1/5/RCP_NM={menu}&RCP_PARTS_DTLS={parts}"

url = f"http://openapi.foodsafetykorea.go.kr/api/{RECIPE_API}/COOKRCP01/json/1000/2000/"
# API 요청
response = requests.get(url)

# 응답 상태 확인
if response.status_code == 200:
    data = json.loads(response.text)
    print(data)
else:
    print(f"API 요청 실패, 상태 코드: {response.status_code}")

# 필요한 정보 추출
total_count = data['COOKRCP01']['total_count']
recipe_name = data['COOKRCP01']['row'][0]['RCP_NM']

# 파일 이름 생성 (total_count_RCP_NM.json 형식)
filename =  f"data/{total_count}_{recipe_name.replace(' ', '')}.json"

# JSON 파일로 저장
with open(filename, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"'{filename}' 파일이 저장되었습니다.")