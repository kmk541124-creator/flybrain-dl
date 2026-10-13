import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import gradio as gr

DATA_DIR = "data"
ANNO_FILE = f"{DATA_DIR}/Supplemental_file1_neuron_annotations.tsv"
BRAIN_FILE = f"{DATA_DIR}/brain_weights.pt"
SONG_FILE = f"{DATA_DIR}/song_inputs.pt"
MODEL_FILE = f"{DATA_DIR}/final_model.pt"

SAMPLE_RATE = 22050
HOP_LENGTH = 1024
START_SEC = 30
DURATION_SEC = 60
FMIN = 100
FMAX = 1000
SCALE = 0.05
HOPS = 4

print("뇌와 모델 불러오는 중...")
brain = torch.load(BRAIN_FILE, weights_only=False)
W = brain["W"].to_sparse_csr()
id_to_index = brain["id_to_index"]
jo_groups = torch.load(SONG_FILE, weights_only=False)["jo_groups"]

anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False)
kc_ids = anno[anno["cell_class"] == "Kenyon_Cell"]["root_id"]
kc_idx = torch.tensor([id_to_index[i] for i in kc_ids if i in id_to_index])
num_neurons = W.shape[0]

saved = torch.load(MODEL_FILE, weights_only=False)
model = nn.Linear(saved["num_kc"], 1)
model.load_state_dict(saved["model"])
target = saved["target"]
print("준비 완료")

def separate_vocals(song_path, out_dir):
    subprocess.run([
        "demucs", "--two-stems=vocals", "--mp3",
        "-o", out_dir,
        "--filename", "{stem}/{track}.{ext}",
        str(song_path),
    ], check=True)
    return Path(out_dir) / "htdemucs" / "vocals" / f"{Path(song_path).stem}.mp3"

def song_to_bands(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, offset=START_SEC, duration=DURATION_SEC)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=HOP_LENGTH,
        n_mels=len(jo_groups), fmin=FMIN, fmax=FMAX,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    bands = np.clip((db + 80) / 80, 0, 1)
    return torch.tensor(bands.T, dtype=torch.float32)

def listen(bands):
    kc_record = torch.zeros(len(bands), len(kc_idx))
    for t in range(len(bands)):
        sound_input = torch.zeros(num_neurons, 1)
        for b, neuron_idx in enumerate(jo_groups):
            sound_input[neuron_idx] = bands[t, b]
        activity = sound_input.clone()
        for hop in range(HOPS):
            activity = torch.tanh(torch.relu(SCALE * (W @ activity) + sound_input))
        kc_record[t] = activity[kc_idx, 0]
    return kc_record

def judge(song_path):
    if song_path is None:
        raise gr.Error("노래 파일을 올려 주세요.")
    if librosa.get_duration(path=song_path) < START_SEC + 1:
        raise gr.Error(f"노래가 너무 짧습니다. {START_SEC + 1}초 이상인 곡을 올려 주세요.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        vocal_path = separate_vocals(song_path, tmp_dir)
        bands = song_to_bands(vocal_path)

    kc_record = listen(bands)

    window = saved["window"]
    num_win = len(kc_record) // window
    x = kc_record[:num_win * window].reshape(num_win, window, -1).mean(dim=1)
    x = (x - saved["mean"]) / saved["std"]

    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze(1)).mean().item()

    return {target: prob, f"{target} 아님": 1 - prob}

demo = gr.Interface(
    fn=judge,
    inputs=gr.Audio(type="filepath", label="노래 파일"),
    outputs=gr.Label(label="판별 결과"),
    title="초파리 뇌 가수 판별기",
    description=f"노래를 올리면 초파리 뇌 연결 구조로 {target}의 노래인지 판별합니다. 한 곡에 2~5분 걸립니다.",
)

demo.launch()