---
title: Kichwa Chord Detector
emoji: 🎸
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🎸 Kichwa Chord Detector — API de reconocimiento de acordes

Reconoce acordes de una canción subida por el usuario y devuelve un JSON con
marcas de tiempo (estilo Chordify), usando librosa (chroma CQT) + clasificación
por plantillas.

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /analyze` → multipart/form-data con campo `file` (audio) y opcional `name`.
  Devuelve el JSON de la progresión con `chords[{startSeconds, endSeconds, chord, confidence}]`.
- `GET /` → página web de prueba.

## Límites (tier gratuito)

- Tamaño máximo: 40 MB por archivo
- Duración máxima: 8 minutos
