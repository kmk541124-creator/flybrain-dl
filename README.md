# flybrain-dl

초파리 뇌 커넥톰(FlyWire) 데이터를 이용한 두 가지 프로젝트를 모아 둔 저장소입니다.

| 폴더 | 프로젝트 | 한 줄 설명 |
| --- | --- | --- |
| [`brain-simulation/`](brain-simulation) | 초파리 뇌 시뮬레이션 | 영상의 소리와 화면을 초파리 뇌에 넣고, 뉴런 반응을 영상과 함께 3D로 보여 주는 웹 |
| [`deep-learning/`](deep-learning) | 초파리 뇌 딥러닝 (가수 판별) | 노래를 초파리 뇌에 들려주고, 케년 세포 반응만으로 어떤 가수의 노래인지 맞히는 모델 |

```
flybrain-dl/
├── data/                 # FlyWire 데이터와 중간 결과 (git에는 올라가지 않음)
├── brain-simulation/     # 초파리 뇌 시뮬레이션
│   ├── prepare_light.py  # 서버용 가벼운 뇌 데이터 만들기
│   ├── 9.py              # 영상 → 뇌 반응 기록
│   ├── 9_full.py         # 9.py의 원래 버전 (torch 사용)
│   ├── server.py         # 웹 서버 (FastAPI)
│   ├── youtubedownload.py
│   ├── viewer/           # 3D 뷰어 화면
│   ├── Dockerfile
│   └── requirements.txt
└── deep-learning/        # 초파리 뇌 딥러닝
    ├── deep.py, 3.py ~ 7.py   # 학습 단계
    ├── 8.py, app.py           # 새 노래 판별
    └── cube.py, learn.py      # 실험: 루빅스 큐브 강화학습
```

> 모든 명령은 **저장소 맨 위 폴더(flybrain-dl)에서** 실행합니다. 두 프로젝트가 `data/` 폴더를 함께 쓰기 때문입니다.

---

## 공통 준비

### 1. 데이터

[FlyWire](https://codex.flywire.ai/)에서 아래 두 파일을 받아 `data/` 폴더에 넣습니다.

| 파일 | 설명 |
| --- | --- |
| `Connectivity_783.parquet` | 뉴런끼리의 연결(시냅스) 정보 |
| `Supplemental_file1_neuron_annotations.tsv` | 뉴런 종류, 위치 정보 |

### 2. 환경

Python 3.11과 `ffmpeg`가 필요합니다.

---

## 1. 초파리 뇌 시뮬레이션 (`brain-simulation/`)

유튜브 영상(또는 영상 파일)의 **소리는 청각 뉴런(JO)**에, **화면은 광수용 뉴런(R1~R8)**에 넣고, 신호가 뇌 연결을 따라 퍼져 나가는 모습을 기록합니다. 결과는 영상과 함께 뉴런 3D 위치 위에 재생됩니다.

### 설치

```bash
pip install -r brain-simulation/requirements.txt
pip install pandas pyarrow   # prepare_light.py 실행용
```

### 가벼운 뇌 데이터 만들기 (처음 한 번만)

```bash
python brain-simulation/prepare_light.py
```

`data/light_brain.npz`가 만들어집니다. 서버는 이 파일만 쓰기 때문에 torch 없이 numpy, scipy만으로 돌아갑니다.

### 웹 서버 실행

```bash
python brain-simulation/server.py
```

브라우저에서 `http://localhost:8080`에 접속해 유튜브 링크를 넣거나 영상 파일(mp4, webm, 50MB 이하)을 올리고, 필요하면 시작·끝 시간을 정한 뒤 실행합니다.

### 터미널에서 바로 실행

```bash
python brain-simulation/9.py <유튜브 링크 또는 영상 경로> [결과 폴더] [시작 초] [끝 초]
```

결과 폴더를 생략하면 `brain-simulation/viewer/`에 저장됩니다.

### Docker

```bash
docker build -f brain-simulation/Dockerfile -t flybrain .
docker run -p 8080:8080 flybrain
```

`data/light_brain.npz`를 먼저 만들어 두어야 합니다.

---

## 2. 초파리 뇌 딥러닝 (`deep-learning/`)

노래에서 목소리만 분리해 초파리의 청각 뉴런에 들려주고, 학습·기억을 담당하는 **케년 세포(Kenyon Cell)**의 반응을 입력으로 가수를 분류합니다.

### 설치

```bash
pip install torch pandas pyarrow librosa matplotlib demucs gradio
```

### 노래 준비

가수 이름 폴더(`김경호/`, `윤하/`)에 노래를 넣고, demucs로 목소리를 분리합니다.

```bash
demucs --two-stems=vocals --mp3 -o 김경호_분리 김경호/*.mp3
demucs --two-stems=vocals --mp3 -o 윤하_분리 윤하/*.mp3
```

### 학습 순서

| 순서 | 명령 | 하는 일 | 결과 |
| --- | --- | --- | --- |
| 1 | `python deep-learning/deep.py` | 연결 정보를 뇌 가중치 행렬로 변환 | `data/brain_weights.pt` |
| 2 | `python deep-learning/3.py` | 목소리를 주파수 대역으로 나눠 청각 뉴런 입력으로 변환 | `data/song_inputs.pt` |
| 3 | `python deep-learning/4.py` | 노래를 뇌에 들려주고 케년 세포 반응 기록 | `data/brain_response.pt`, `step4_response.png` |
| 4 | `python deep-learning/5.py` | 케년 세포 반응으로 가수 분류 모델 학습 | `data/singer_model.pt` |
| 5 | `python deep-learning/6.py` | 곡 하나씩 빼고 검증해 "초파리 뇌 / 소리 그대로 / 찍기" 정확도 비교 | 터미널 출력 |
| 6 | `python deep-learning/7.py` | 전체 곡으로 최종 모델 학습 | `data/final_model.pt` |

### 새 노래 판별

```bash
python deep-learning/8.py <노래 파일 경로>   # 터미널
python deep-learning/app.py                  # 웹 화면 (Gradio)
```

### 실험: 루빅스 큐브 강화학습

`cube.py`는 큐브 시뮬레이터, `learn.py`는 초파리의 PN→케년 세포 연결을 특징 추출기로 써서 큐브 맞추기를 강화학습(Q-learning)시키는 실험입니다.

```bash
python deep-learning/learn.py
```

---

## 사용 기술

- **언어**: Python
- **뇌 시뮬레이션**: NumPy, SciPy(희소 행렬), FFmpeg, yt-dlp, FastAPI, Uvicorn, JavaScript(3D 뷰어), Docker
- **딥러닝**: PyTorch, pandas, librosa, demucs, Gradio, matplotlib
- **데이터**: FlyWire 초파리 뇌 커넥톰 (783 버전)
