"""유튜브 영상을 초파리의 눈과 귀에 넣고 뇌 반응을 기록하는 파일입니다 (메모리 절약판).
무거운 준비 작업은 prepare_light.py가 미리 끝내 두었기 때문에 여기서는 numpy와 scipy만 씁니다."""
import sys
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from youtubedownload import download_video

DATA_DIR = "data"
LIGHT_FILE = f"{DATA_DIR}/light_brain.npz"
VIEWER_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("viewer")

START_SEC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
END_SEC = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
DURATION_SEC = END_SEC - START_SEC if END_SEC is not None else None
CUT = ["-ss", str(START_SEC)] + (["-t", str(DURATION_SEC)] if DURATION_SEC is not None else [])
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 1024
FRAME_STEP = 2
FPS = SAMPLE_RATE / HOP_LENGTH / FRAME_STEP

SCALE = 0.05
HOPS = 4
FRAME_W = 64
FRAME_H = 48
BATCH = 32

source = sys.argv[1] if len(sys.argv) > 1 else input("유튜브 링크 또는 영상 파일 경로: ").strip().strip('"')

if source.startswith("http"):
    print("영상 다운로드 중...")
    video_path = download_video(source)
else:
    video_path = Path(source)

VIEWER_DIR.mkdir(parents=True, exist_ok=True)

print(f"소리 꺼내는 중... ({START_SEC}초부터 {f'{DURATION_SEC}초 동안' if DURATION_SEC is not None else '끝까지'})")
audio = subprocess.run([
    "ffmpeg", "-loglevel", "error",
    *CUT,
    "-i", str(video_path),
    "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
], capture_output=True, check=True).stdout
y = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
del audio

print("화면 꺼내는 중...")
video = subprocess.run([
    "ffmpeg", "-loglevel", "error",
    *CUT,
    "-i", str(video_path),
    "-vf", f"fps={FPS},scale={FRAME_W}:{FRAME_H},format=gray",
    "-f", "rawvideo", "-",
], capture_output=True, check=True).stdout
frames = np.frombuffer(video, dtype=np.uint8).reshape(-1, FRAME_H, FRAME_W)
print("화면 장 수:", len(frames))
if len(frames) == 0 or len(y) == 0:
    sys.exit("시작 시간이 영상 길이보다 깁니다.")

print("뇌 불러오기")
light = np.load(LIGHT_FILE)
num_neurons = int(light["num_neurons"])
W = csr_matrix((light["W_data"], light["W_indices"], light["W_indptr"]), shape=(num_neurons, num_neurons))
jo_idx = light["jo_idx"]
jo_band = light["jo_band"]
photo_idx = light["photo_idx"]
pixel_x = light["pixel_x"]
pixel_y = light["pixel_y"]
show_idx = light["show_idx"]
groups = light["groups"]
positions = light["positions"]
mel_filters = light["mel_filters"]
print("보여줄 뉴런:", len(show_idx))

window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
padded = np.pad(y, N_FFT // 2)
num_steps = 1 + len(y) // HOP_LENGTH
mel = np.zeros((len(mel_filters), num_steps), dtype=np.float32)
for start in range(0, num_steps, 256):
    end = min(start + 256, num_steps)
    chunk = np.stack([padded[t * HOP_LENGTH: t * HOP_LENGTH + N_FFT] for t in range(start, end)])
    power = np.abs(np.fft.rfft(chunk * window, axis=1)) ** 2
    mel[:, start:end] = mel_filters @ power.T
db = 10 * np.log10(np.maximum(mel, 1e-10)) - 10 * np.log10(max(mel.max(), 1e-10))
db = np.maximum(db, db.max() - 80)
bands = np.clip((db + 80) / 80, 0, 1).T.astype(np.float32)

steps = list(range(0, len(bands), FRAME_STEP))
num_frames = min(len(steps), len(frames))
record = np.zeros((num_frames, len(show_idx)), dtype=np.float16)

print("초파리 뇌에 보여 주고 들려주는 중...")
for start in range(0, num_frames, BATCH):
    end = min(start + BATCH, num_frames)
    sense_input = np.zeros((num_neurons, end - start), dtype=np.float32)
    sense_input[jo_idx] = bands[steps[start:end]][:, jo_band].T
    sense_input[photo_idx] = (frames[start:end, pixel_y, pixel_x] / 255.0).T

    activity = sense_input.copy()
    for hop in range(HOPS):
        activity = np.tanh(np.maximum(SCALE * (W @ activity) + sense_input, 0))
    record[start:end] = activity[show_idx].T
    print(f"  {end}/{num_frames}")
del W, sense_input, activity

low = record.min(axis=0).astype(np.float32)
high = record.max(axis=0).astype(np.float32)
alive = high - low >= 0.01
with open(VIEWER_DIR / "brain_data.bin", "wb") as f:
    for start in range(0, num_frames, 64):
        change = (record[start:start + 64].astype(np.float32) - low) / (high - low + 1e-6)
        change[:, ~alive] = 0
        (change * 255).round().astype(np.uint8).tofile(f)

probe = subprocess.run([
    "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "csv=p=0", str(video_path),
], capture_output=True, text=True).stdout
codecs = {}
for line in probe.split():
    parts = line.split(",")
    if len(parts) >= 2:
        codecs.setdefault(parts[1], parts[0])
playable = codecs.get("video") == "h264" and codecs.get("audio", "aac") == "aac"

if playable:
    video_start = START_SEC
    if source.startswith("http"):
        shutil.move(video_path, VIEWER_DIR / "video.mp4")
    else:
        shutil.copy(video_path, VIEWER_DIR / "video.mp4")
else:
    print(f"브라우저에서 재생되도록 영상 변환 중... ({codecs.get('video')} -> h264)")
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        *CUT,
        "-i", str(video_path),
        "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(VIEWER_DIR / "video.mp4"),
    ], check=True)
    video_start = 0
    if source.startswith("http"):
        Path(video_path).unlink(missing_ok=True)

meta = {
    "num_neurons": len(show_idx),
    "num_frames": num_frames,
    "fps": FPS,
    "start_sec": video_start,
    "video": "video.mp4",
    "positions": [round(v, 4) for v in positions.flatten().tolist()],
    "groups": groups.tolist(),
}
with open(VIEWER_DIR / "brain_meta.json", "w") as f:
    json.dump(meta, f)

print("완료! 반응하는 뉴런 비율:", f"{alive.mean():.1%}")
