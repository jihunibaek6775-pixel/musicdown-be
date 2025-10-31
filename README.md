# YouTube Audio Downloader & Lyrics Extractor

유튜브 URL을 입력하면 음악/영상을 다운로드하고 faster-whisper AI를 사용하여 가사를 자동으로 추출하는 FastAPI 프로젝트입니다.

## 기능

- 유튜브 영상/음악 다운로드 (MP3 형식)
- AI 기반 가사 자동 추출 (faster-whisper - 일반 Whisper보다 4-5배 빠름!)
- VAD(Voice Activity Detection)로 음성 구간만 처리
- 타임스탬프와 함께 가사 저장
- 다운로드한 파일 목록 조회

## 설치 방법

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 필요한 패키지 설치
pip install fastapi uvicorn yt-dlp faster-whisper
```

## 실행 방법

```bash
# FastAPI 서버 실행
uvicorn main:app --reload

# 또는
python main.py
```

서버가 실행되면 http://localhost:8000 에서 접속할 수 있습니다.

## API 사용법

### 1. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. 유튜브 영상 다운로드 및 가사 추출

```bash
curl -X POST "http://localhost:8000/download" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

또는 Swagger UI에서 직접 테스트 가능합니다.

### 3. 다운로드한 파일 목록 확인

```bash
curl http://localhost:8000/files
```

## 프로젝트 구조

```
1031mini/
├── main.py              # FastAPI 애플리케이션
├── downloads/           # 다운로드한 음악 파일 저장
├── lyrics/             # 추출한 가사 파일 저장
├── venv/               # 가상환경
└── README.md           # 프로젝트 설명
```

## 주의사항

- ffmpeg가 시스템에 설치되어 있어야 합니다
- faster-whisper 모델은 처음 실행 시 자동으로 다운로드됩니다
- GPU가 있으면 자동으로 감지하여 사용합니다 (10-20배 더 빠름)
- CPU 사용 시에도 일반 Whisper보다 4-5배 빠릅니다

## 성능

- **faster-whisper**: 일반 Whisper보다 4-5배 빠름
- **GPU 사용 시**: 추가로 10-20배 가속
- **VAD 필터**: 음성이 없는 구간 건너뛰어 더욱 빠름
- **메모리 효율**: 적은 메모리로 더 빠른 처리
