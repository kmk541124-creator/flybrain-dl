import torch
import torch.nn as nn

DATA_DIR = "data"
RESPONSE_FILE = f"{DATA_DIR}/brain_response.pt"
MODEL_FILE = f"{DATA_DIR}/singer_model.pt"

EPOCHS = 2000
LEARNING_RATE = 0.001
TEST_SONGS = [8, 17]

data = torch.load(RESPONSE_FILE, weights_only=False)
kc_record = data["kc_record"].float()
labels = data["labels"]
names = data["names"]
singers = data["singers"]
num_steps, num_songs, num_kc = kc_record.shape

train_songs = [i for i in range(num_songs) if i not in TEST_SONGS]
print("학습용: ", [names[i] for i in train_songs])
print("시험용: ", [names[i] for i in TEST_SONGS])

def make_dataset(song_list):
    x = torch.cat([kc_record[:, i] for i in song_list])
    y = torch.cat([torch.full((num_steps,), float(labels[i])) for i in song_list])
    return x, y

x_train, y_train = make_dataset(train_songs)
x_test, y_test = make_dataset(TEST_SONGS)
print("학습 데이터: ", tuple(x_train.shape), " | 시험 데이터: ", tuple(x_train.shape))

model = nn.Linear(num_kc, 1)
loss_fc = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    logits = model(x_train).squeeze(1)
    loss = loss_fc(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        train_acc = ((logits > 0).float() == y_train).float().mean().item()
        print(f"epoch {epoch} | 손실 {loss.item():.4f} | 학습 정확도 {train_acc:.2%}")


with torch.no_grad():
    test_prob = torch.sigmoid(model(x_test).squeeze(1))
    test_acc = ((test_prob > 0.5).float() == y_test).float().mean().item()
print(f"\n처음 듣는 곡 정확도: {test_acc:.2%}")

for i in TEST_SONGS:
    with torch.no_grad():
        song_prob = torch.sigmoid(model(kc_record[:, i]).squeeze(1)).mean().item()
    verdict = "김경호" if song_prob > 0.5 else "김경호 아님"
    print(f"[정답: {singers[i]}] {names[i]} -> 김경호일 확률 {song_prob:.2%} -> 판단 {verdict}")

torch.save(model.state_dict(), MODEL_FILE)
print("모델 저장 완료: ", MODEL_FILE)    