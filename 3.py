from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa

DATA_DIR = "data"
ANNO_FILE = f"{DATA_DIR}/Supplemental_file1_neuron_annotations.tsv"
BRAIN_FILE = f"{DATA_DIR}/brain_weights.pt"
OUT_FILE = f"{DATA_DIR}/song_inputs.pt"

SINNGERS = {"김경호": 1, "윤하": 0}
SAMPLES_RATE = 22050
HOP_LENGTH = 1024
START_SEC = 30
DURATION_SEC = 120
FMIN = 100
FMAX = 1000

JO_TYPES = [
        "JO-B1_a", "JO-B1_b", "JO-B1_c", "JO-B2", "JO-B3", "JO-B4_a", "JO-B4_b",
    "JO-A1", "JO-A2", "JO-A3", "JO-A4", "JO-A5",
]

anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False)
id_to_index = torch.load(BRAIN_FILE, weights_only=False)["id_to_index"]

jo_groups = []
for jo_types in JO_TYPES:
    ids = anno[anno["cell_type"] == jo_types]["root_id"]
    indices = [id_to_index[i] for i in ids if i in id_to_index]
    jo_groups.append(torch.tensor(indices))
    print(f"{jo_types}: 뉴런 {len(indices)}개")


def song_to_bands(path):
    y, sr = librosa.load(path, sr=SAMPLES_RATE, offset=START_SEC, duration=DURATION_SEC)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=HOP_LENGTH,
        n_mels=len(JO_TYPES), fmin=FMIN, fmax=FMAX,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    bands = np.clip((db + 80)/80, 0, 1)
    return torch.tensor(bands.T, dtype=torch.float32)

sons = []

for singer, label in SINNGERS.items():
    for path in sorted(Path(f"{singer}_분리/htdemucs/vocals").glob("*.mp3")):
        bands = song_to_bands(path)
        sons.append({"singer": singer, "label": label, "name": path.name, "bands":bands})
        print(f"{singer} | {path.name} | {tuple(bands.shape)}")


torch.save({"songs": sons, "jo_groups": jo_groups, "jo_types": JO_TYPES}, OUT_FILE)
print("저장 완료:", OUT_FILE)