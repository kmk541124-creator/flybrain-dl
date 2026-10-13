import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa

DATA_DIR = "data"
ANNO_FILE = f"{DATA_DIR}/Supplemental_file1_neuron_annotations.tsv"
BRAIN_FILE = f"{DATA_DIR}/brain_weights.pt"
SONG_FILE = f"{DATA_DIR}/song_inputs.pt"
MODEL_FILE = f"{DATA_DIR}/final_model.pt"
SEPARATE_DIR = "판별_분리"

SAMPLE_RATE = 22050
HOP_LENGTH = 1024
START_SEC = 30
DURATION_SEC = 60
FMIN = 100
FMAX = 1000
SCALE = 0.05
HOPS = 4

song_path = Path(sys.argv[1])

print("1/4 목소리 분리 중...")
subprocess.run([
    "demucs", "--two-stems=vocals", "--mp3",
    "-o", SEPARATE_DIR,
    "--filename", "{stem}/{track}.{ext}",
    str(song_path),
], check=True)
vocal_path = Path(SEPARATE_DIR) / "htdemucs" / "vocals" / f"{song_path.stem}.mp3"

print("2/4 소리를 숫자로 변환 중...")
y, sr = librosa.load(vocal_path, sr=SAMPLE_RATE, offset=START_SEC, duration=DURATION_SEC)
mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=2048, hop_length=HOP_LENGTH,
    n_mels=12, fmin=FMIN, fmax=FMAX,
)
db = librosa.power_to_db(mel, ref=np.max)
bands = torch.tensor(np.clip((db + 80) / 80, 0, 1).T, dtype=torch.float32)

print("3/4 초파리 뇌에 들려주는 중...")
brain = torch.load(BRAIN_FILE, weights_only=False)
W = brain["W"].to_sparse_csr()
id_to_index = brain["id_to_index"]
jo_groups = torch.load(SONG_FILE, weights_only=False)["jo_groups"]

anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False)
kc_ids = anno[anno["cell_class"] == "Kenyon_Cell"]["root_id"]
kc_idx = torch.tensor([id_to_index[i] for i in kc_ids if i in id_to_index])
num_neurons = W.shape[0]

kc_record = torch.zeros(len(bands), len(kc_idx))
for t in range(len(bands)):
    sound_input = torch.zeros(num_neurons, 1)
    for b, neuron_idx in enumerate(jo_groups):
        sound_input[neuron_idx] = bands[t, b]
    activity = sound_input.clone()
    for hop in range(HOPS):
        activity = torch.tanh(torch.relu(SCALE * (W @ activity) + sound_input))
    kc_record[t] = activity[kc_idx, 0]
    if t % 300 == 0:
        print(f"  {t}/{len(bands)}")

print("4/4 판별 중...")
saved = torch.load(MODEL_FILE, weights_only=False)
model = nn.Linear(saved["num_kc"], 1)
model.load_state_dict(saved["model"])

window = saved["window"]
num_win = len(kc_record) // window
x = kc_record[:num_win * window].reshape(num_win, window, -1).mean(dim=1)
x = (x - saved["mean"]) / saved["std"]

with torch.no_grad():
    prob = torch.sigmoid(model(x).squeeze(1)).mean().item()

verdict = saved["target"] if prob > 0.5 else f"{saved['target']} 아님"
print(f"\n{song_path.name}")
print(f"{saved['target']}일 확률: {prob:.2%} -> 판단: {verdict}")