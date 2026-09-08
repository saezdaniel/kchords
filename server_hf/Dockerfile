FROM python:3.10-slim

# ffmpeg: necesario para que librosa y yt-dlp decodifiquen y extraigan audio
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Soporta tanto el puerto de Render ($PORT) como Hugging Face (7860)
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
