import numpy as np
import pandas as pd
from cube import MOVES, scramble, apply_move, is_solved, observe, solved_cube

DATA_DIR = "data"

NUM_PN = 288

anno = pd.read_csv(f"{DATA_DIR}/Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_class"])

conn = pd.read_parquet(f"{DATA_DIR}/Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])

pn_ids = anno.root_id[anno.cell_class == "ALPN"]
kc_ids = anno.root_id[anno.cell_class == "Kenyon_Cell"]

edges = conn[conn.Presynaptic_ID.isin(pn_ids) & conn.Postsynaptic_ID.isin(kc_ids)]
strength = edges.groupby("Presynaptic_ID").Connectivity.sum().sort_values(ascending=False)

pn_list = strength.index[:NUM_PN].to_numpy()

edges = edges[edges.Presynaptic_ID.isin(pn_list)]

kc_list = np.sort(edges.Postsynaptic_ID.unique())

pn_pos = {r : i for i, r in enumerate(pn_list)}
kc_pos = {r : i for i, r in enumerate(kc_list)}


W_pk = np.zeros((len(kc_list), NUM_PN), dtype=np.float32)
W_pk[edges.Postsynaptic_ID.map(kc_pos).to_numpy(), edges.Presynaptic_ID.map(pn_pos).to_numpy()] = edges.Connectivity.to_numpy()

NUM_KC = len(kc_list)

CENTER_SLOTS = [face * 9 + 4 for face in range(6)]

keep = np.ones(324, dtype=bool)

for s in CENTER_SLOTS:
    keep[s * 6 : s * 6 + 6] = False

NUM_ACTIVE = int(NUM_KC * 0.05)


def kc_response(x):
    drive = W_pk @ x[keep]
    threshold = np.partition(drive, -NUM_ACTIVE)[-NUM_ACTIVE]
    return (drive >= threshold).astype(np.float32)

LEARING_RATE = 0.5

GAMMA = 0.9

EPSION = 0.1

STEP_COST = -0.05

W_out = np.zeros((len(MOVES), NUM_KC), dtype=np.float32)

rng = np.random.default_rng(0)

def choice_act():
    