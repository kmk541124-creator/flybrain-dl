FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY brain-simulation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/light_brain.npz data/
COPY brain-simulation/viewer/index.html brain-simulation/viewer/
COPY brain-simulation/viewer/empty/ brain-simulation/viewer/empty/
COPY brain-simulation/server.py brain-simulation/9.py brain-simulation/youtubedownload.py brain-simulation/

EXPOSE 8080

CMD ["python", "brain-simulation/server.py"]
