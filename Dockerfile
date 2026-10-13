FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/light_brain.npz data/
COPY viewer/index.html viewer/
COPY viewer/empty/ viewer/empty/
COPY server.py 9.py youtubedownload.py ./

EXPOSE 8080

CMD ["python", "server.py"]
