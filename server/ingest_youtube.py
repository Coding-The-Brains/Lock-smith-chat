import os
import subprocess

VIDEO_URLS = [
    "https://youtu.be/eigYEYR0N_w",
    "https://youtu.be/PB0OvFtGmSg",
    "https://youtu.be/kK7l4vL_yKU",
    "https://youtu.be/pmEeR5MTjrg",
    "https://youtu.be/70r6r0SMhjg",
    "https://youtu.be/3kkVIpYahR0",
    "https://youtu.be/SwOgRbsJDhM",
    "https://youtu.be/FQkoEj9GLtU"
]

os.makedirs("transcripts", exist_ok=True)

for url in VIDEO_URLS:
    print(f"📥 Downloading transcript for {url} ...")
    try:
        subprocess.run([
            "yt-dlp",
            "--write-auto-subs",
            "--skip-download",
            "--sub-lang", "en",
            "--convert-subs", "srt",
            "-o", f"transcripts/%(id)s.%(ext)s",
            url
        ], check=True)
        print(f"✅ Done: {url}")
    except Exception as e:
        print(f"❌ Failed: {url} - {e}")
