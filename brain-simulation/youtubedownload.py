import os
from pathlib import Path
import yt_dlp

SAVE_DIR = "다운로드 위치"
MODE = "video"


def download_video(url, mode=MODE):
    ydl_opts = {
        "outtmpl": os.path.join(SAVE_DIR, "%(title)s.%(ext)s"),
        "noplaylist": True,
    }

    if mode == "video":
        ydl_opts["format"] = "bestvideo[height<=480][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=480][vcodec^=avc1]/bestvideo[height<=480]+bestaudio/best"
        ydl_opts["merge_output_format"] = "mp4"
    else:
        ydl_opts["format"] = "bestaudio/best"
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        info = ydl.process_ie_result(info, download=True)
        saved = Path(ydl.prepare_filename(info))

    saved = saved.with_suffix(".mp4" if mode == "video" else ".mp3")
    print(f"저장 완료: {saved}")
    return saved


if __name__ == "__main__":
    url = input("다운로드할 유튜브 URL을 입력하세요: ").strip()
    download_video(url)