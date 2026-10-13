import torch
import torch.nn as nn

DATA_DIR = "data"
RESPONSE_FILE = f"{DATA_DIR}/brain_response.pt"
MODEL_FILE = f"{DATA_DIR}/final_model.pt"
TARGET_SINGER = "김경호"
WINDOW = 10
EPOCHS = 500
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01

torch.manual_seed(0)

response = torch.load(RESPONSE_FILE, weights_only=False)
singers = response["singers"]
labels = torch.tensor([1.0 if s == TARGET_SINGER else 0.0 for s in singers])

kc = response["kc_record"].permute(1, 0, 2).float()
num_songs, num_steps, num_kc = kc.shape
num_win = num_steps//WINDOW
kc = kc[:, :num_win * WINDOW].reshape(num_songs, num_win, WINDOW, num_kc).mean(dim=2)


x = kc.reshape(-1, num_kc)
y = labels.repeat_interleave(num_win)
mean = x.mean(dim=0)
std = x.std(dim=0) + 1e-6
x = (x - mean) / std

model = nn.Linear(num_kc, 1)
loss_fc = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for epoch in range(EPOCHS):
    logits = model(x).squeeze(1)
    loss = loss_fc(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        acc = ((logits > 0).float() == y).float().mean().item()
        print(f"epoch {epoch} | 손실 {loss.item():.4f} | 학습 정확도 {acc:.2%}")



torch.save({
    "model": model.state_dict(),
    "mean": mean,
    "std": std,
    "num_kc": num_kc,
    "window": WINDOW,
    "target": TARGET_SINGER,
}, MODEL_FILE)
print("최종 모델 저장 완료:", MODEL_FILE)