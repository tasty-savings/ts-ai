# Step 1: 베이스 이미지로 Python 3.12.7 사용
FROM python:3.12.7-slim

# Step 2: 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# Step 3: pip 업그레이드
RUN pip install --upgrade pip

# Step 4: 로컬의 requirements.txt 파일을 컨테이너로 복사
COPY requirements.txt /app/

# Step 5: 필요한 Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: 애플리케이션의 모든 소스 파일을 컨테이너로 복사
COPY . /app

# Step 7: Flask 애플리케이션이 5000번 포트에서 리스닝하도록 설정
EXPOSE 5000

# Step 8: Flask 애플리케이션 실행 명령어
CMD ["python", "main.py"]