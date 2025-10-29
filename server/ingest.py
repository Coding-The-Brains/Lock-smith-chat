import os
import re
import json
import time
import pathlib
import argparse
import subprocess
import numpy as np
import yt_dlp
import faiss
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# ======================
# 📁 Paths
# ======================
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TMP_DIR = DATA_DIR / "tmp"
INDEX_DIR = DATA_DIR / "index"
META_PATH = INDEX_DIR / "meta.json"
INDEX_PATH = INDEX_DIR / "faiss.index"

MAX_FILE_MB = 24  # Whisper 25MB limit with safe margin
CHUNK_DURATION = 600  # 10 min segments

# ======================
# 🔑 Load API Key
# ======================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise SystemExit("❌ Missing GROQ_API_KEY in .env file or environment variable")

client = Groq(api_key=GROQ_API_KEY)

# ======================
# 🆔 Helper functions
# ======================
def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id
    m = re.search(r"v=([\w-]{11})", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([\w-]{11})", url_or_id)
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse video id from: {url_or_id}")

def _video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"

# ======================
# 🎧 Download MP3
# ======================
def download_audio_mp3(video_id: str) -> pathlib.Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(TMP_DIR / f"{video_id}.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "extractor_args": {"youtube": ["player_client=android"]},  # bypass SABR
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    url = _video_url(video_id)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    mp3_path = TMP_DIR / f"{video_id}.mp3"
    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        raise FileNotFoundError(f"❌ Audio file missing or empty for {video_id}")
    return mp3_path

# ======================
# ✂️ Split if too large
# ======================
def split_audio_if_needed(file_path: pathlib.Path) -> List[pathlib.Path]:
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb <= MAX_FILE_MB:
        return [file_path]

    print(f"⚠️ File {file_path.name} is {size_mb:.2f} MB, splitting into chunks...")
    chunk_paths = []
    output_template = TMP_DIR / f"{file_path.stem}_part_%03d.mp3"

    subprocess.run([
        "ffmpeg",
        "-i", str(file_path),
        "-f", "segment",
        "-segment_time", str(CHUNK_DURATION),
        "-c", "copy",
        str(output_template)
    ], check=True)

    for p in TMP_DIR.glob(f"{file_path.stem}_part_*.mp3"):
        if p.stat().st_size > 0:
            chunk_paths.append(p)

    file_path.unlink(missing_ok=True)
    return sorted(chunk_paths)

# ======================
# 🧠 Transcribe with Groq Whisper
# ======================
def transcribe_audio(file_path: pathlib.Path) -> str:
    """Transcribe audio using Groq Whisper-Large-v3"""
    print(f"🎙️ Transcribing: {file_path.name}")
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="json"
        )
    text = response.text.strip()
    return text

# ======================
# 🧠 Transcribe multiple chunks
# ======================
def transcribe_chunks(paths: List[pathlib.Path]) -> str:
    full_text = []
    for idx, p in enumerate(paths, 1):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"🧠 Transcribing chunk {idx}/{len(paths)} ({size_mb:.2f} MB)...")
        text = transcribe_audio(p)
        full_text.append(text)
        p.unlink(missing_ok=True)
    return " ".join(full_text)

# ======================
# ✂️ Chunk text for embedding
# ======================
def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

# ======================
# 🧮 Embeddings (local or API)
# ======================
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("❌ Missing OPENAI_API_KEY in .env file or environment variable")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    arr = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(arr)
    return arr

# ======================
# 🧠 FAISS Index
# ======================
def build_index(chunks: List[Dict[str, Any]]):
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index: faiss.Index, meta: List[Dict[str, Any]]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ======================
# 📝 Read URLs
# ======================
def read_videos_file(path: pathlib.Path) -> List[str]:
    ids = []
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                ids.append(extract_video_id(line))
            except Exception:
                pass
    return ids

# ======================
# 🏃 Main
# ======================
def main():
    parser = argparse.ArgumentParser(description="🎧 Ingest YouTube audio → Transcribe (Groq) → Embed → FAISS Index")
    parser.add_argument("--videos-file", type=str, default="data/videos.txt", help="Path to file with YouTube URLs/IDs")
    args = parser.parse_args()

    videos = read_videos_file(pathlib.Path(args.videos_file))
    if not videos:
        raise SystemExit("❌ No videos found in videos file.")

    meta_records = []
    chunk_records = []

    print(f"🎧 Processing {len(videos)} videos...\n")

    for i, vid in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] Downloading audio for {vid} ...")
        try:
            mp3_path = download_audio_mp3(vid)
        except Exception as e:
            print(f"❌ Failed to download audio for {vid}: {e}")
            continue

        chunks_paths = split_audio_if_needed(mp3_path)
        if not chunks_paths:
            print(f"⚠️ No valid audio chunks for {vid}, skipping.")
            continue

        transcript_text = transcribe_chunks(chunks_paths)
        if not transcript_text.strip():
            print(f"⚠️ Empty transcription for {vid}, skipping.")
            continue

        # Get video title
        try:
            r = requests.get(
                "https://www.youtube.com/oembed",
                params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"},
                timeout=20,
            )
            title = r.json().get("title", vid)
        except Exception:
            title = vid

        text_chunks = chunk_text(transcript_text)
        for c in text_chunks:
            chunk_records.append({
                "video_id": vid,
                "title": title,
                "text": c,
            })
        meta_records.append({
            "video_id": vid,
            "title": title,
            "total_chunks": len(text_chunks),
        })

        print(f"🧩 {len(text_chunks)} chunks created for {title}\n")
        time.sleep(0.5)

    print(f"🪄 Embedding and indexing {len(chunk_records)} chunks...")
    index = build_index(chunk_records)
    save_index(index, chunk_records)
    print(f"✅ Index saved at {INDEX_PATH}")
    print(f"✅ Metadata saved at {META_PATH}")
    print("🎉 Done!")


if __name__ == "__main__":
    main()
