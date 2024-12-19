# Step 1: 베이스 이미지로 Python 3.12.7 사용
FROM python:3.12.7-slim AS builder

# Step 2: 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# Git과 Git LFS 설치
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Step 3: pip 업그레이드
RUN pip install --upgrade pip

# Step 4: 로컬의 requirements.txt 파일을 컨테이너로 복사
COPY requirements.txt .

# Step 5: 필요한 Python 패키지 설치
RUN pip install --no-cache-dir --use-feature=fast-deps -r requirements.txt

# Step 6: 실행 단계 (runtime image)
FROM python:3.12.7-slim AS runtime

# Git LFS 설치 (runtime 이미지에도 필요)
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Step 7: 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# Step 8: 빌드 단계에서 설치한 패키지를 실행 이미지로 복사
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Step 9: 애플리케이션의 모든 소스 파일을 컨테이너로 복사
COPY . .

# Git LFS 파일 pull
RUN git lfs pull

# Step 10: FastAPI 애플리케이션이 5555번 포트에서 리스닝하도록 설정
EXPOSE 5555

# Step 11: FastAPI 애플리케이션 실행 명령어
CMD ["python", "src/main.py"]
