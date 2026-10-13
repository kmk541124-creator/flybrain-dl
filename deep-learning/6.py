import torch
import torch.nn as nn

DATA_DIR = "data"
RESPONSE_FILE = f"{DATA_DIR}/brain_response.pt"
SONG_FILE = f"{DATA_DIR}/song_inputs.pt"
TARGET_SINGER = "김경호"
WINDOW = 10
EPOCHS = 500
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01

torch.manual_seed(0)

response = torch.load(RESPONSE_FILE, weights_only=False)
song_data = torch.load(SONG_FILE, weights_only=False)

singers = response["singers"]
names = response["names"]
assert names == [s["name"] for s in song_data["songs"]], "3.py와 4.py 결과의 곡 순서가 다릅니다. 4.py를 다시 실행하세요."
labels = torch.tensor([1.0 if s == TARGET_SINGER else 0.0 for s in singers])

kc = response["kc_record"].permute(1, 0, 2).float()
bands = torch.stack([s["bands"] for s in song_data["songs"]])


def windowed(x):
    num_songs, num_steps, num_feat = x.shape
    num_win = num_steps // WINDOW
    return x[:, :num_win * WINDOW].reshape(num_songs, num_win, WINDOW, num_feat).mean(dim=2)


def cross_validate(features, title):
    num_songs, num_win, num_feat = features.shape
    correct = 0
    print(f"\n===== {title} =====")
    for test in range(num_songs):
        train = [i for i in range(num_songs) if i != test]
        x_train = features[train].reshape(-1, num_feat)
        y_train = labels[train].repeat_interleave(num_win)
        mean = x_train.mean(dim=0)
        std = x_train.std(dim=0) + 1e-6
        x_train = (x_train - mean) / std
        x_test = (features[test] - mean) / std

        model = nn.Linear(num_feat, 1)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS):
            loss = loss_fn(model(x_train).squeeze(1), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prob = torch.sigmoid(model(x_test).squeeze(1)).mean().item()
        hit = (prob > 0.5) == (labels[test].item() == 1.0)
        correct += int(hit)
        mark = "O" if hit else "X"
        print(f"{mark} [정답: {singers[test]}] {names[test]} -> {TARGET_SINGER}일 확률 {prob:.2%}")

    acc = correct / num_songs
    print(f"{title} 곡 단위 정확도: {correct}/{num_songs} ({acc:.2%})")
    return acc


kc_acc = cross_validate(windowed(kc), "초파리 뇌 (케년 세포)")
band_acc = cross_validate(windowed(bands), "소리 그대로 (비교용)")

print("\n===== 결과 =====")
print(f"초파리 뇌: {kc_acc:.2%} | 소리 그대로: {band_acc:.2%} | 아무렇게나 찍기: 50.00%")
