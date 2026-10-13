import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from youtubedownload import download_video
import pandas as pd
import torch
import librosa
import cv2

DATA_DIR = "data"
ANNO_FILE = f"{DATA_DIR}/Supplemental_file1_neuron_annotations.tsv"
BRAIN_FILE = f"{DATA_DIR}/brain_weights.pt"
SONG_FILE = f"{DATA_DIR}/song_inputs.pt"
VIEWER_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("viewer")

START_SEC = 0
DURATION_SEC = 300
SAMPLE_RATE = 22050
HOP_LENGTH = 1024
FRAME_STEP = 2
FPS = SAMPLE_RATE / HOP_LENGTH / FRAME_STEP
FMIN = 100
FMAX = 1000

SCALE = 0.05
HOPS = 4
NUM_OTHERS = 20000
VOXEL_NM = np.array([4, 4, 40])

PHOTO_TYPES = ["R1-6", "R7", "R8"]
FRAME_W = 64
FRAME_H = 48
EYE_OVERLAP = 0.1




source = sys.argv[1] if len(sys.argv) > 1 else input("유튜브 링크 또는 영상 파일 경로: ").strip().strip('"')

if source.startswith("http"):
    print("영상 다운로드 중...")
    video_path = download_video(source)
else:
    video_path = Path(source)

VIEWER_DIR.mkdir(parents=True, exist_ok=True)

print("뇌 불러오기")
brain = torch.load(BRAIN_FILE, weights_only=False)
W = brain["W"].to_sparse_csr()
id_to_index = brain["id_to_index"]
jo_groups = torch.load(SONG_FILE, weights_only=False)["jo_groups"]
num_neuros = W.shape[0]

anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False)
anno = anno[anno["root_id"].isin(list(id_to_index))].copy()
anno["index"] = anno["root_id"].map(id_to_index)
anno = anno.set_index("index")

jo_idx = torch.cat(jo_groups)
kc_idx = torch.tensor(anno.index[anno["cell_class"] == "Kenyon_Cell"].values)

photo = anno[anno["cell_type"].isin(PHOTO_TYPES) & anno["side"].isin(["left", "right"])]
photo_idx = torch.tensor(photo.index.values)

rest = anno.drop(index=jo_idx.tolist() + kc_idx.tolist() + photo_idx.tolist())
other_idx = torch.tensor(rest.sample(n=NUM_OTHERS, random_state=0).index.values)

show_idx = torch.cat([jo_idx, kc_idx, photo_idx, other_idx])
groups = [0] * len(jo_idx) + [1] * len(kc_idx) + [2] * len(photo_idx) + [3] * len(other_idx)
print("소리 뉴런:", len(jo_idx), "| 케년 세포:", len(kc_idx), "| 광수용체:", len(photo_idx))

is_dan = anno["cell_class"] == "DAN"
cell_type = anno["cell_type"].fillna("")

pam_idx = torch.tensor(anno.index[is_dan & cell_type.str.startswith("PAM")].values)
ppl1_idx = torch.tensor(anno.index[is_dan & cell_type.str.startswith("PPL1")].values)
octo_idx = torch.tensor(anno.index[anno["top_nt"] == "octopamine"].values)
mbon_idx = torch.tensor(anno.index[anno["cell_class"] == "MBON"].values)

print("좋음(PAM):", len(pam_idx), "| 싫음(PPL1):", len(ppl1_idx), "| 흥분(옥토파민):", len(octo_idx), "| 판단(MBON):", len(mbon_idx))

def eye_map(eye_df):
    p = eye_df[["pos_x", "pos_y", "pos_z"]].values * VOXEL_NM
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    uv = p @ vt[:2].T
    uv = (uv - uv.min(axis=0)) / (uv.max(axis=0) - uv.min(axis=0))
    return uv


eye_x = np.zeros(len(photo))
eye_y = np.zeros(len(photo))
for side, x_from, x_to in [("left", 0.0, 0.5 + EYE_OVERLAP), ("right", 0.5 - EYE_OVERLAP, 1.0)]:
    is_side = (photo["side"] == side).values
    uv = eye_map(photo[is_side])
    eye_x[is_side] = x_from + uv[:, 0] * (x_to - x_from)
    eye_y[is_side] = uv[:, 1]

pixel_x = torch.tensor((eye_x * (FRAME_W - 1)).round().astype(int))
pixel_y = torch.tensor((eye_y * (FRAME_H - 1)).round().astype(int))

cap = cv2.VideoCapture(str(video_path))
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

DURATION_SEC = frame_count / video_fps
print(f"영상 길이: {DURATION_SEC:.1f}초")

print("소리 꺼내는 중...")
with tempfile.TemporaryDirectory() as tmp_dir:
    wav_path = Path(tmp_dir) / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(START_SEC), "-t", str(DURATION_SEC),
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(wav_path),
    ], check=True)
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=2048, hop_length=HOP_LENGTH,
    n_mels=len(jo_groups), fmin=FMIN, fmax=FMAX,
)
db = librosa.power_to_db(mel, ref=np.max)
bands = torch.tensor(np.clip((db + 80) / 80, 0, 1).T, dtype=torch.float32)

print("화면 꺼내는 중...")
result = subprocess.run([
    "ffmpeg", "-loglevel", "error",
    "-ss", str(START_SEC), "-t", str(DURATION_SEC),
    "-i", str(video_path),
    "-vf", f"fps={FPS},scale={FRAME_W}:{FRAME_H},format=gray",
    "-f", "rawvideo", "-",
], capture_output=True, check=True)
frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, FRAME_H, FRAME_W) / 255.0
print("화면 장 수:", len(frames))

steps = list(range(0, len(bands), FRAME_STEP))
num_frames = min(len(steps), len(frames))
record = np.zeros((num_frames, len(show_idx)), dtype=np.float32)


BATCH = 128

print("초파리 뇌에 보여 주고 들려주는 중...")
for start in range(0, num_frames, BATCH):
    end = min(start + BATCH, num_frames)
    size = end - start
    sense_input = torch.zeros(num_neuros, size)

    t_list = steps[start:end]
    for b, neuron_idx in enumerate(jo_groups):
        sense_input[neuron_idx] = bands[t_list, b]

    frame_batch = torch.tensor(frames[start:end], dtype=torch.float32)
    sense_input[photo_idx] = frame_batch[:, pixel_y, pixel_x].T

    activity = sense_input.clone()
    for hop in range(HOPS):
        activity = torch.tanh(torch.relu(SCALE * (W @ activity) + sense_input))
    record[start:end] = activity[show_idx].T.numpy()

    print(f"  {end}/{num_frames}")

low = record.min(axis=0)
high = record.max(axis=0)
change = (record - low) / (high - low + 1e-6)
change[:, high - low < 0.01] = 0
(change * 255).round().astype(np.uint8).tofile(VIEWER_DIR / "brain_data.bin")

pos = anno.loc[show_idx.numpy(), ["pos_x", "pos_y", "pos_z"]].values * VOXEL_NM
pos = pos - pos.mean(axis=0)
pos = pos / np.abs(pos).max()
pos[:, 1] = -pos[:, 1]

if source.startswith("http"):
    shutil.move(video_path, VIEWER_DIR / "video.mp4")
else:
    shutil.copy(video_path, VIEWER_DIR / "video.mp4")

meta = {
    "num_neurons": len(show_idx),
    "num_frames": num_frames,
    "fps": FPS,
    "start_sec": START_SEC,
    "video": "video.mp4",
    "positions": pos.round(4).flatten().tolist(),
    "groups": groups,
}
with open(VIEWER_DIR / "brain_meta.json", "w") as f:
    json.dump(meta, f)

print("완료! 반응하는 뉴런 비율:", f"{(high - low >= 0.01).mean():.1%}")