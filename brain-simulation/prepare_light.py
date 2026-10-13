"""9.py가 서버에서 쓸 가벼운 뇌 데이터(data/light_brain.npz)를 한 번만 미리 만들어 두는 파일입니다.
torch, librosa, cv2 없이 numpy와 scipy만으로 9.py가 돌아가도록, 무거운 준비 작업을 여기서 끝냅니다."""
import numpy as np
import pandas as pd

DATA_DIR = "data"
ANNO_FILE = f"{DATA_DIR}/Supplemental_file1_neuron_annotations.tsv"
CONN_FILE = f"{DATA_DIR}/Connectivity_783.parquet"
OUT_FILE = f"{DATA_DIR}/light_brain.npz"

JO_TYPES = [
    "JO-B1_a", "JO-B1_b", "JO-B1_c", "JO-B2", "JO-B3", "JO-B4_a", "JO-B4_b",
    "JO-A1", "JO-A2", "JO-A3", "JO-A4", "JO-A5",
]
PHOTO_TYPES = ["R1-6", "R7", "R8"]
NUM_OTHERS = 20000
VOXEL_NM = np.array([4, 4, 40])
FRAME_W = 64
FRAME_H = 48
EYE_OVERLAP = 0.1
SAMPLE_RATE = 22050
N_FFT = 2048
FMIN = 100
FMAX = 1000


def hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filters(n_mels):
    fftfreqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


print("연결 표 읽는 중...")
conn = pd.read_parquet(CONN_FILE, columns=["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index",
                                            "Postsynaptic_Index", "Excitatory x Connectivity"])
pairs = pd.concat([
    conn[["Presynaptic_ID", "Presynaptic_Index"]].set_axis(["id", "index"], axis=1),
    conn[["Postsynaptic_ID", "Postsynaptic_Index"]].set_axis(["id", "index"], axis=1),
]).drop_duplicates()
id_to_index = dict(zip(pairs["id"], pairs["index"]))
num_neurons = len(id_to_index)

print("가중치 행렬 만드는 중...")
from scipy.sparse import coo_matrix
W = coo_matrix(
    (conn["Excitatory x Connectivity"].to_numpy(np.float32),
     (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
    shape=(num_neurons, num_neurons),
).tocsr()
del conn, pairs

print("뉴런 정보 읽는 중...")
anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False,
                   usecols=["root_id", "cell_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"])
anno = anno[anno["root_id"].isin(list(id_to_index))].copy()
anno["index"] = anno["root_id"].map(id_to_index)
anno = anno.set_index("index")

jo_groups = []
for jo_type in JO_TYPES:
    ids = anno.loc[anno["cell_type"] == jo_type, "root_id"]
    jo_groups.append(np.array([id_to_index[i] for i in ids], dtype=np.int64))
jo_idx = np.concatenate(jo_groups)
jo_band = np.concatenate([np.full(len(g), b) for b, g in enumerate(jo_groups)])
kc_idx = anno.index[anno["cell_class"] == "Kenyon_Cell"].values

photo = anno[anno["cell_type"].isin(PHOTO_TYPES) & anno["side"].isin(["left", "right"])]
photo_idx = photo.index.values

rest = anno.drop(index=jo_idx.tolist() + kc_idx.tolist() + photo_idx.tolist())
other_idx = rest.sample(n=NUM_OTHERS, random_state=0).index.values

show_idx = np.concatenate([jo_idx, kc_idx, photo_idx, other_idx])
groups = np.array([0] * len(jo_idx) + [1] * len(kc_idx) + [2] * len(photo_idx) + [3] * len(other_idx), dtype=np.int8)


def eye_map(eye_df):
    p = eye_df[["pos_x", "pos_y", "pos_z"]].values * VOXEL_NM
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    uv = p @ vt[:2].T
    return (uv - uv.min(axis=0)) / (uv.max(axis=0) - uv.min(axis=0))


eye_x = np.zeros(len(photo))
eye_y = np.zeros(len(photo))
for side, x_from, x_to in [("left", 0.0, 0.5 + EYE_OVERLAP), ("right", 0.5 - EYE_OVERLAP, 1.0)]:
    is_side = (photo["side"] == side).values
    uv = eye_map(photo[is_side])
    eye_x[is_side] = x_from + uv[:, 0] * (x_to - x_from)
    eye_y[is_side] = uv[:, 1]

pos = anno.loc[show_idx, ["pos_x", "pos_y", "pos_z"]].values * VOXEL_NM
pos = pos - pos.mean(axis=0)
pos = pos / np.abs(pos).max()
pos[:, 1] = -pos[:, 1]

np.savez(
    OUT_FILE,
    num_neurons=num_neurons,
    W_indptr=W.indptr.astype(np.int32),
    W_indices=W.indices.astype(np.int32),
    W_data=W.data.astype(np.float32),
    jo_idx=jo_idx.astype(np.int32),
    jo_band=jo_band.astype(np.int8),
    photo_idx=photo_idx.astype(np.int32),
    pixel_x=(eye_x * (FRAME_W - 1)).round().astype(np.int16),
    pixel_y=(eye_y * (FRAME_H - 1)).round().astype(np.int16),
    show_idx=show_idx.astype(np.int32),
    groups=groups,
    positions=pos.round(4).astype(np.float32),
    mel_filters=mel_filters(len(JO_TYPES)),
)
print("저장 완료:", OUT_FILE, "| 보여줄 뉴런:", len(show_idx))
