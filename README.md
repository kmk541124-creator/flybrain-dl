# flybrain-dl

초파리 뇌 커넥톰(FlyWire) 데이터로 신경망을 만들어, 초파리 뇌에 소리와 영상을 넣었을 때 어떤 뉴런이 반응하는지 실험하는 딥러닝 학습용 프로젝트입니다.

크게 두 가지 실험이 들어 있습니다.

1. **가수 판별** — 노래를 초파리의 청각 뉴런(JO)에 들려주고, 케년 세포(Kenyon Cell)의 반응만 보고 어떤 가수의 노래인지 맞힙니다.
2. **뇌 반응 시각화 웹** — 유튜브 영상(또는 영상 파일)의 소리는 청각 뉴런에, 화면은 광수용 뉴런(R1~R8)에 넣고, 뇌 전체의 반응을 영상과 함께 3D로 보여 줍니다.

---

## 준비

### 1. 데이터

`data/` 폴더에 FlyWire 데이터가 있어야 합니다.

| 파일 | 설명 |
| --- | --- |
| `Connectivity_783.parquet` | 뉴런끼리의 연결(시냅스) 정보 |
| `Supplemental_file1_neuron_annotations.tsv` | 뉴런 종류, 위치 정보 |

### 2. 환경

Python 3.11과 `ffmpeg`가 필요합니다.

```bash
# 웹 서버만 돌릴 때
pip install -r requirements.txt

# 가수 판별 실험까지 할 때 (추가 패키지)
pip install torch pandas pyarrow librosa matplotlib demucs gradio
```

---

## 사용법 1: 뇌 반응 시각화 웹

### 가벼운 뇌 데이터 만들기 (처음 한 번만)

```bash
python prepare_light.py
```

`data/light_brain.npz`가 만들어집니다. 서버는 이 파일만 쓰기 때문에 torch 없이 numpy, scipy만으로 돌아갑니다.

### 서버 실행

```bash
python server.py
```

브라우저에서 `http://localhost:8080`에 들어가 유튜브 링크를 넣거나 영상 파일(mp4, webm, 50MB 이하)을 올리고, 필요하면 시작·끝 시간을 정한 뒤 실행합니다. 처리가 끝나면 영상과 함께 뉴런 반응이 재생됩니다.

### 터미널에서 바로 실행

```bash
python 9.py <유튜브 링크 또는 영상 경로> [결과 폴더] [시작 초] [끝 초]
```

결과 폴더를 생략하면 `viewer/`에 저장되고, `viewer/index.html`로 확인할 수 있습니다.

### Docker로 실행

```bash
docker build -t flybrain .
docker run -p 8080:8080 flybrain
```

`prepare_light.py`로 `data/light_brain.npz`를 먼저 만들어 두어야 합니다.

---

## 사용법 2: 가수 판별 실험

노래 파일은 가수 이름 폴더(`김경호/`, `윤하/`)에 넣고, demucs로 목소리를 분리해 `<가수>_분리/htdemucs/vocals/`에 둡니다.

```bash
demucs --two-stems=vocals --mp3 -o 김경호_분리 김경호/*.mp3
```

그다음 아래 순서대로 실행합니다.

| 순서 | 파일 | 하는 일 | 결과 |
| --- | --- | --- | --- |
| 1 | `deep.py` | 연결 정보를 뇌 가중치 행렬로 변환 | `data/brain_weights.pt` |
| 2 | `3.py` | 목소리를 주파수 대역으로 나눠 청각 뉴런 입력으로 변환 | `data/song_inputs.pt` |
| 3 | `4.py` | 노래를 뇌에 들려주고 케년 세포 반응 기록 | `data/brain_response.pt`, `step4_response.png` |
| 4 | `5.py` | 케년 세포 반응으로 가수 분류 모델 학습 | `data/singer_model.pt` |
| 5 | `6.py` | 곡 하나씩 빼고 검증해서 "초파리 뇌 / 소리 그대로 / 찍기" 정확도 비교 | 터미널 출력 |
| 6 | `7.py` | 전체 곡으로 최종 모델 학습 | `data/final_model.pt` |

### 새 노래 판별

```bash
# 터미널
python 8.py <노래 파일 경로>

# 웹 화면 (Gradio)
python app.py
```

---

## 기타 파일

| 파일 | 설명 |
| --- | --- |
| `9_full.py` | `9.py`의 원래 버전. torch, librosa, cv2로 원본 데이터를 직접 읽음 (메모리 많이 사용) |
| `youtubedownload.py` | 유튜브 영상/음원 다운로드 (`다운로드 위치/`에 저장) |
| `cube.py`, `learn.py` | 실험: 초파리 PN→KC 연결로 루빅스 큐브를 강화학습시키는 시도 |
| `results/` | 서버가 처리한 결과 (최근 20개만 유지) |
| `flagged/` | Gradio(`app.py`)에서 신고(Flag)한 기록 |

## 사용 기술

Python, NumPy, SciPy, PyTorch, librosa, demucs, FastAPI, Gradio, Docker, FlyWire 커넥톰 데이터
