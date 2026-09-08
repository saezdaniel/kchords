FROM python:3.11-slim

# ffmpeg: decodificación de audio
# nodejs: requerido por yt-dlp para ejecutar el motor JS de descifrado de firmas de YouTube
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el contenido (app.py, cookies.txt si existe)
COPY . .

# Soporta tanto el puerto de Render ($PORT) como Hugging Face (7860)
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
