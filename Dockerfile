# 1. 베이스 이미지 선택
# python:3.12-slim은 Debian 기반의 경량화된 Python 3.12 이미지
# 2025년 기준 Python 3.12가 안정적이고 성능이 우수함
FROM python:3.12-slim

# 2. 메타데이터 추가 (선택사항이지만 실무에서 권장)
LABEL maintainer="your-email@example.com"
LABEL version="1.0.0"
LABEL description="FastAPI User Authentication API"

# 3. 작업 디렉토리 설정
# 이후 모든 명령어는 /app 디렉토리에서 실행
WORKDIR /app

# 4. Python 환경변수 설정
# PYTHONDONTWRITEBYTECODE: .pyc 파일 생성 방지 (컨테이너에서 불필요)
# PYTHONUNBUFFERED: 로그를 버퍼링 없이 즉시 출력 (실시간 로그 확인 가능)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 5. 시스템 패키지 업데이트 및 필요한 도구 설치
# curl: 헬스체크에 사용
# --no-install-recommends: 불필요한 패키지 제외
# 캐시 삭제로 이미지 크기 최소화
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 6. 의존성 파일 먼저 복사 (레이어 캐싱 최적화)
# requirements.txt가 변경되지 않으면 이 레이어는 캐시 사용
COPY requirements.txt .

# 7. Python 패키지 설치
# --no-cache-dir: pip 캐시 저장 안함 (이미지 크기 감소)
# --upgrade pip: pip를 최신 버전으로 업그레이드
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. 애플리케이션 소스 코드 복사
# 소스는 자주 변경되므로 마지막에 복사
COPY main.py .

# 9. 컨테이너가 사용할 포트 명시 (문서화 목적)
# 실제 포트 바인딩은 docker run -p 옵션으로 수행
EXPOSE 8000

# 10. 헬스체크 설정
# 30초마다 헬스체크 수행, 타임아웃 10초, 3번 실패 시 unhealthy 상태로 전환
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 11. 컨테이너 시작 시 실행할 명령어
# uvicorn: ASGI 서버
# main:app: main.py 파일의 app 객체
# --host 0.0.0.0: 모든 네트워크 인터페이스에서 접속 허용
# --port 8000: 8000번 포트 사용
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]