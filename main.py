from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import yt_dlp
from faster_whisper import WhisperModel
import os
from pathlib import Path
import logging
import re
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Audio Downloader & Lyrics Extractor")

# CORS 설정 (React 프론트엔드 연결)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 디렉토리 생성
DOWNLOAD_DIR = Path("downloads")
LYRICS_DIR = Path("lyrics")
DOWNLOAD_DIR.mkdir(exist_ok=True)
LYRICS_DIR.mkdir(exist_ok=True)

# faster-whisper 모델 로드 (4-5배 빠름!)
# 모델 크기: tiny < base < small < medium < large
# device="cpu": CPU 사용 (CUDA 오류 방지)
# compute_type="int8": CPU 최적화 (빠르고 메모리 효율적)
logger.info("Loading faster-whisper model (medium) on CPU...")
whisper_model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    download_root=None,  # 기본 캐시 디렉토리 사용
)
logger.info("faster-whisper model loaded successfully (CPU mode)")


class YouTubeURL(BaseModel):
    url: HttpUrl


def sanitize_filename(filename: str) -> str:
    """
    파일명에서 특수문자를 제거하고 안전한 형식으로 변환합니다.
    """
    # 특수문자 제거
    filename = re.sub(r'[<>:"/\\|?*\[\]\']', '', filename)
    # 연속된 공백을 하나로
    filename = re.sub(r'\s+', ' ', filename)
    # 앞뒤 공백 제거
    filename = filename.strip()
    # 파일명이 너무 길면 자르기 (Windows 경로 제한)
    if len(filename) > 100:
        filename = filename[:100]
    return filename


def download_audio(youtube_url: str) -> tuple[str, str, str]:
    """
    유튜브에서 오디오를 다운로드합니다.

    Returns:
        tuple: (audio_file_path, video_title, safe_filename)
    """
    # 고유한 ID 생성
    unique_id = str(uuid.uuid4())[:8]

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(DOWNLOAD_DIR / f'{unique_id}.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading audio from: {youtube_url}")
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info['title']

            # 다운로드된 파일 경로 (고유 ID 사용)
            audio_file = DOWNLOAD_DIR / f"{unique_id}.mp3"

            # 안전한 파일명 생성
            safe_filename = sanitize_filename(video_title)

            # 파일명 변경 (안전한 이름으로)
            new_audio_file = DOWNLOAD_DIR / f"{safe_filename}.mp3"

            # 같은 이름의 파일이 있으면 고유 ID 추가
            if new_audio_file.exists():
                new_audio_file = DOWNLOAD_DIR / f"{safe_filename}_{unique_id}.mp3"

            # 파일 이름 변경
            if audio_file.exists():
                audio_file.rename(new_audio_file)
                audio_file = new_audio_file

            logger.info(f"Audio downloaded: {audio_file}")
            return str(audio_file.absolute()), video_title, safe_filename
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"다운로드 실패: {str(e)}")


def format_timestamp(seconds: float) -> str:
    """초를 MM:SS 형식으로 변환"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def extract_lyrics(audio_file: str, video_title: str, safe_filename: str) -> str:
    """
    faster-whisper를 사용하여 오디오에서 가사를 추출합니다.

    Returns:
        str: lyrics_file_path
    """
    try:
        logger.info(f"Transcribing audio: {audio_file}")
        # 절대 경로 사용
        audio_path = Path(audio_file).absolute()

        # faster-whisper 옵션 (정확도 향상)
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="ko",  # 한국어 명시
            task="transcribe",  # 번역이 아닌 전사(transcribe)
            beam_size=5,  # 빔 서치 크기
            temperature=0.0,  # 더 결정적인 결과
            vad_filter=False,  # VAD 필터 비활성화 (모든 가사 추출)
        )

        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        # 세그먼트 리스트로 변환 (generator이므로)
        segments_list = list(segments)

        lyrics_file = LYRICS_DIR / f"{safe_filename}_lyrics.txt"

        # 전체 가사 텍스트 생성
        full_text_parts = []
        for segment in segments_list:
            text = segment.text.strip()
            full_text_parts.append(text)

        # 가사 파일 저장 (전체 가사만)
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(f"제목: {video_title}\n\n")
            f.write("=" * 50 + "\n\n")
            f.write(" ".join(full_text_parts))

        logger.info(f"Lyrics saved: {lyrics_file}")
        return str(lyrics_file.absolute())
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"가사 추출 실패: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "YouTube Audio Downloader & Lyrics Extractor API",
        "endpoints": {
            "/download": "POST - 유튜브 URL로 오디오 다운로드 및 가사 추출 (Whisper AI)",
            "/download/{filename}": "GET - 다운로드한 오디오 파일 다운로드",
            "/lyrics/{filename}": "GET - 추출한 가사 파일 다운로드",
            "/files": "GET - 다운로드된 파일 목록 조회"
        }
    }


@app.post("/download")
async def download_and_extract(data: YouTubeURL):
    """
    유튜브 URL을 받아서 오디오를 다운로드하고 Whisper AI로 가사를 추출합니다.
    """
    try:
        # 1. 오디오 다운로드
        audio_file, video_title, safe_filename = download_audio(str(data.url))

        # 2. Whisper로 가사 추출
        logger.info("Using Whisper AI for lyrics extraction...")
        lyrics_file = extract_lyrics(audio_file, video_title, safe_filename)

        return JSONResponse(content={
            "status": "success",
            "message": "다운로드 및 가사 추출 완료",
            "video_title": video_title,
            "audio_file": os.path.basename(audio_file),
            "lyrics_file": os.path.basename(lyrics_file),
            "download_links": {
                "audio": f"/download/{os.path.basename(audio_file)}",
                "lyrics": f"/lyrics/{os.path.basename(lyrics_file)}"
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    다운로드한 오디오 파일을 반환합니다.
    """
    file_path = DOWNLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")

    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename
    )


@app.get("/lyrics/{filename}")
async def get_lyrics(filename: str):
    """
    추출한 가사 파일을 반환합니다.
    """
    file_path = LYRICS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="가사 파일을 찾을 수 없습니다")

    return FileResponse(
        path=file_path,
        media_type="text/plain",
        filename=filename
    )


@app.get("/files")
async def list_files():
    """
    다운로드된 파일과 가사 파일 목록을 반환합니다.
    """
    audio_files = [f.name for f in DOWNLOAD_DIR.glob("*.mp3")]
    lyrics_files = [f.name for f in LYRICS_DIR.glob("*.txt")]

    return {
        "audio_files": audio_files,
        "lyrics_files": lyrics_files
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
