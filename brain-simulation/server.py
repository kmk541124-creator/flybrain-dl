import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MAX_RESULTS = 20
MAX_WAITING = 10
JOB_TIMEOUT_SEC = 30 * 60
MAX_UPLOAD_MB = 50
UPLOAD_NAME = "upload.bin"
YOUTUBE_URL = re.compile(r"^https://(www\.|m\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)[A-Za-z0-9_-]{11}")

app = FastAPI()
jobs = {}
waiting = []
job_queue = queue.Queue()
lock = threading.Lock()


class ProcessRequest(BaseModel):
    source: str
    start: float = 0
    end: float | None = None


def check_range(start, end):
    if start < 0:
        raise HTTPException(status_code=400, detail="시작 시간은 0초 이상이어야 합니다.")
    if end is not None and end <= start:
        raise HTTPException(status_code=400, detail="끝 시간은 시작 시간보다 뒤여야 합니다.")


def check_queue():
    if len(waiting) >= MAX_WAITING:
        raise HTTPException(status_code=429, detail="대기 중인 요청이 너무 많습니다. 잠시 후 다시 시도하세요.")


def add_job(job_id, source, start, end):
    with lock:
        jobs[job_id] = {"state": "waiting", "error": None, "created": time.time()}
        waiting.append(job_id)
    job_queue.put((job_id, source, start, end))


def clean_old_results():
    folders = sorted((p for p in RESULTS_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for folder in folders[:-MAX_RESULTS]:
        shutil.rmtree(folder, ignore_errors=True)
        jobs.pop(folder.name, None)


def worker():
    while True:
        job_id, source, start, end = job_queue.get()
        with lock:
            waiting.remove(job_id)
            jobs[job_id]["state"] = "running"
        out_dir = RESULTS_DIR / job_id
        out_dir.mkdir(exist_ok=True)
        try:
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "9.py"), source, str(out_dir), str(start), "" if end is None else str(end)],
                capture_output=True, text=True, timeout=JOB_TIMEOUT_SEC,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                lines = result.stderr.strip().splitlines()
                raise RuntimeError(lines[-1] if lines else "알 수 없는 오류")
            if not (out_dir / "brain_meta.json").exists():
                raise RuntimeError("결과 파일이 만들어지지 않았습니다.")
            with lock:
                jobs[job_id]["state"] = "done"
        except subprocess.TimeoutExpired:
            with lock:
                jobs[job_id].update(state="error", error="처리 시간이 너무 오래 걸려 중단했습니다.")
        except Exception as err:
            message = str(err)
            if "CalledProcessError" in message and "ffmpeg" in message:
                message = "영상 파일을 읽지 못했습니다. mp4 또는 webm 영상인지 확인해 주세요."
            with lock:
                jobs[job_id].update(state="error", error=message)
        (out_dir / UPLOAD_NAME).unlink(missing_ok=True)
        with lock:
            clean_old_results()


threading.Thread(target=worker, daemon=True).start()


@app.post("/api/process")
def process(req: ProcessRequest):
    url = req.source.strip()
    if not YOUTUBE_URL.match(url):
        raise HTTPException(status_code=400, detail="유튜브 영상 링크만 넣을 수 있습니다.")
    check_range(req.start, req.end)
    with lock:
        check_queue()
    job_id = uuid.uuid4().hex[:12]
    add_job(job_id, url, req.start, req.end)
    return {"job": job_id}


@app.post("/api/upload")
async def upload(request: Request, start: float = 0, end: float | None = None):
    check_range(start, end)
    with lock:
        check_queue()
    job_id = uuid.uuid4().hex[:12]
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir()
    path = out_dir / UPLOAD_NAME
    size = 0
    try:
        with open(path, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"영상 파일은 {MAX_UPLOAD_MB}MB까지 올릴 수 있습니다.")
                f.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    add_job(job_id, str(path), start, end)
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            if (RESULTS_DIR / job_id / "brain_meta.json").exists():
                return {"state": "done"}
            raise HTTPException(status_code=404, detail="없는 작업입니다.")
        ahead = waiting.index(job_id) + 1 if job_id in waiting else 0
        return {"state": job["state"], "error": job["error"], "ahead": ahead}


app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/", StaticFiles(directory=BASE_DIR / "viewer", html=True), name="viewer")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
