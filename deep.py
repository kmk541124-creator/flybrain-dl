import pandas as pd
import torch

DATA_DIR = "data"
CONN_FILE = f"{DATA_DIR}/Connectivity_783.parquet"
OUT_FILE = f"{DATA_DIR}/brain_weights.pt"

conn = pd.read_parquet(CONN_FILE)

num_neurons = max(conn["Presynaptic_Index"].max(), conn["Postsynaptic_Index"].max()) + 1
print("뉴런 수: ", num_neurons)

rows = torch.tensor(conn["Postsynaptic_Index"].values)
cols = torch.tensor(conn["Presynaptic_Index"].values)
values = torch.tensor(conn["Excitatory x Connectivity"].values, dtype=torch.float32)

W = torch.sparse_coo_tensor(
    torch.stack([rows, cols]),
    values,
    size=(num_neurons, num_neurons),
).coalesce()

print("행렬 크기:", W.shape)
print("연결 수:", W._nnz())
print("흥분성 연결:", (values > 0).sum().item())
print("억제성 연결:", (values < 0).sum().item())

pre = conn[["Presynaptic_ID", "Presynaptic_Index"]].drop_duplicates()
post = conn[["Postsynaptic_ID", "Postsynaptic_Index"]].drop_duplicates()
id_to_index = dict(zip(pre["Presynaptic_ID"], pre["Presynaptic_Index"]))
id_to_index.update(zip(post["Postsynaptic_ID"], post["Postsynaptic_Index"]))
print("ID 사전 크기:", len(id_to_index))

torch.save({"W": W, "id_to_index": id_to_index}, OUT_FILE)
print("저장 완료:", OUT_FILE)