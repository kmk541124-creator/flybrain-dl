import pandas as pd
import torch
import matplotlib.pyplot as plt

DATA_DIR = "data"
ANNO_FILE = f"{DATA_DIR}/Supplemental_file1_neuron_annotations.tsv"
BRAIN_FILE = f"{DATA_DIR}/brain_weights.pt"
SONG_FILE = f"{DATA_DIR}/song_inputs.pt"
OUT_FILE = f"{DATA_DIR}/brain_response.pt"

SCALE = 0.05
HOPS = 4

plt.rcParams["font.family"] = "AppleGothic"

brain = torch.load(BRAIN_FILE, weights_only=False)
song_data = torch.load(SONG_FILE, weights_only=False)
anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False)

W = brain["W"].to_sparse_csr()
id_to_index = brain["id_to_index"]
songs = song_data["songs"]
jo_groups = song_data["jo_groups"]
num_neurous = W.shape[0]

kc_ids = anno[anno["cell_class"] == "Kenyon_Cell"]["root_id"]
kc_idx = torch.tensor([id_to_index[i] for i in kc_ids if i in id_to_index])
print("케년 세포 수:" , len(kc_idx))

brands = torch.stack([s["bands"] for s in songs])
num_songs, num_steps, num_bands = brands.shape
print("곡 수: ", num_songs, "| 순간 수:", num_steps)

kc_records = torch.zeros(num_steps, num_songs, len(kc_idx), dtype=torch.float16)

for t in range(num_steps):
    sound_input = torch.zeros(num_neurous, num_songs)
    for b, neurous_idx in enumerate(jo_groups):
        sound_input[neurous_idx] = brands[:, t, b]

    activity = sound_input.clone()
    for hop in range(HOPS):
        received = W @ activity
        activity = torch.tanh(torch.relu(SCALE * received + sound_input))

    kc_records[t] = activity[kc_idx].T.half()

    if t % 200 == 0:
        print(f"{t}/{num_steps} | 케년 세포 평균 활동: {activity[kc_idx].mean().item():.3f}")

torch.save({
    "kc_record": kc_records,
    "labels": torch.tensor([s["label"] for s in songs]),
    "names": [s["name"] for s in songs],
    "singers": [s["singer"] for s in songs],
}, OUT_FILE)
print("저장 완료:", OUT_FILE)

seconds = torch.arange(num_steps) * 1024 / 22050
fig, axes = plt.subplots(num_songs, 1, figsize=(10, 2 * num_songs), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(seconds, brands[i].mean(dim=1), label="소리 세기", alpha=0.6)
    ax.plot(seconds, kc_records[:, i].float().mean(dim=1), label="케년 세포 활동")
    ax.set_title(f"{songs[i]['singer']} | {songs[i]['name']}", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("시간 (초)")
plt.tight_layout()
plt.savefig("step4_response.png", dpi=120)
print("그래프 저장 완료: step4_response.png")