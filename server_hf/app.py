"""
Kichwa Music — Detector de Acordes (Hugging Face Space)
========================================================
API REST para reconocer acordes de una canción en el servidor.

Endpoints:
  GET  /health   -> {"status": "ok"}
  POST /analyze  -> multipart/form-data: file (audio), name (opcional)
  GET  /         -> página de prueba en el navegador

Devuelve el mismo esquema JSON que el motor local de la app:
{
  "app": "Kichwa Music", "type": "chord-progression",
  "song": "...", "analysis": {...},
  "metadata": {"durationSeconds": 213.5, "estimatedKey": "C major"},
  "chords": [{"startSeconds": 0.0, "endSeconds": 1.87, "chord": "C", "confidence": 0.82}, ...]
}
"""

import io

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

SAMPLE_RATE = 22050
HOP = 2048  # ~93 ms por frame de análisis
MAX_SECONDS = 8 * 60
MAX_BYTES = 40 * 1024 * 1024

LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

KIND_SUFFIX = {
    'maj': '', 'min': 'm', 'dim': 'dim', 'aug': 'aug',
    'sus2': 'sus2', 'sus4': 'sus4',
    '7': '7', 'maj7': 'maj7', 'min7': 'm7', '6': '6', 'min6': 'm6',
}
KIND_INTERVALS = {
    'maj': [0, 4, 7], 'min': [0, 3, 7], 'dim': [0, 3, 6], 'aug': [0, 4, 8],
    'sus2': [0, 2, 7], 'sus4': [0, 5, 7],
    '7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'min7': [0, 3, 7, 10],
    '6': [0, 4, 7, 9], 'min6': [0, 3, 7, 9],
}


def _build_templates():
    """Plantillas binarias por clase tonal (una por tipo de acorde y fundamental)."""
    out = []
    for kind, intervals in KIND_INTERVALS.items():
        for root in range(12):
            v = np.zeros(12, dtype=np.float64)
            for iv in intervals:
                v[(root + iv) % 12] = 1.0
            norm = np.linalg.norm(v)
            out.append((LABELS[root] + KIND_SUFFIX[kind], v / norm))
    return out


TEMPLATES = _build_templates()

# Perfiles de Krumhansl-Kessler para la estimación de tonalidad
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _smooth(mat, win):
    """Media móvil a lo largo del tiempo. mat: (12, N)."""
    n = mat.shape[1]
    if win <= 1 or n == 0:
        return mat
    kernel = np.ones(win) / win
    out = np.empty_like(mat)
    for i in range(mat.shape[0]):
        out[i] = np.convolve(mat[i], kernel, mode='same')
    return out


def _classify(chroma, rms, threshold):
    """Clasifica cada frame por correlación coseno con las plantillas."""
    n = chroma.shape[1]
    norms = np.linalg.norm(chroma, axis=0)
    norms[norms == 0] = 1.0
    chroma_n = chroma / norms
    tpl_matrix = np.stack([v for _, v in TEMPLATES])
    sims = tpl_matrix @ chroma_n  # (M, N)
    best = np.argmax(sims, axis=0)

    labels, confs = [], []
    for i in range(n):
        if rms[i] < threshold:
            labels.append('N.C.')
            confs.append(0.0)
        else:
            labels.append(TEMPLATES[best[i]][0])
            confs.append(float(sims[best[i], i]))
    return labels, confs


def _median_filter(labels, win):
    """Votación mayoritaria en una ventana para eliminar cambios espurios."""
    if win <= 1:
        return labels[:]
    half = win // 2
    n = len(labels)
    out = labels[:]
    for i in range(half, n - half):
        counts = {}
        for lab in labels[i - half:i + half + 1]:
            counts[lab] = counts.get(lab, 0) + 1
        out[i] = max(counts, key=counts.get)
    return out


def _patch_short(labels, min_frames):
    """Fusiona segmentos más cortos que min_frames con su vecino estable."""
    labels = labels[:]
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        if (j - i) < min_frames:
            left = labels[i - 1] if i > 0 else None
            right = labels[j] if j < n else None
            current = labels[i]
            if left is not None and left != current:
                replacement = left
            elif right is not None and right != current:
                replacement = right
            elif right is not None:
                replacement = right
            elif left is not None:
                replacement = left
            else:
                replacement = current
            for k in range(i, j):
                labels[k] = replacement
        i = j
    return labels


def _segments(labels, confs, frame_sec, window_sec):
    """Construye los acordes con marcas de tiempo a partir de las etiquetas."""
    segs = []
    n = len(labels)
    if n == 0:
        return segs
    start = 0
    current = labels[0]
    for i in range(1, n + 1):
        lab = labels[i] if i < n else None
        if lab != current:
            conf = sum(confs[start:i]) / max(1, i - start)
            segs.append({
                'startSeconds': round(start * frame_sec, 3),
                'endSeconds': round(i * frame_sec + window_sec, 3),
                'chord': current,
                'confidence': round(conf, 3),
            })
            if i < n:
                start = i
                current = lab
    return segs


def _estimate_key(chroma):
    """Tonalidad estimada por correlación con perfiles Krumhansl-Kessler."""
    avg = chroma.mean(axis=1)
    best_name, best_score = 'C major', -2.0
    for root in range(12):
        score = float(np.corrcoef(avg, np.roll(MAJOR_PROFILE, root))[0, 1])
        if score > best_score:
            best_score, best_name = score, f'{LABELS[root]} major'
        score = float(np.corrcoef(avg, np.roll(MINOR_PROFILE, root))[0, 1])
        if score > best_score:
            best_score, best_name = score, f'{LABELS[root]} minor'
    return best_name


def run_analysis(y, sr, song):
    """Pipeline completo: cromagrama CQT -> plantillas -> segmentos JSON."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

    n = min(chroma.shape[1], len(rms))
    chroma, rms = chroma[:, :n], rms[:n]

    frame_sec = HOP / sr
    threshold = max(0.02 * float(rms.max()), 1e-5)

    # Suavizado temporal del cromagrama (~1.1 s)
    win = max(1, int(round(1.1 / frame_sec)))
    chroma_s = _smooth(chroma, win)

    labels, confs = _classify(chroma_s, rms, threshold)

    min_frames = max(1, int(round(0.4 / frame_sec)))
    med_win = min_frames if min_frames % 2 == 1 else min_frames + 1
    labels = _median_filter(labels, med_win)
    labels = _patch_short(labels, min_frames)

    segs = _segments(labels, confs, frame_sec, HOP / sr)
    key = _estimate_key(chroma_s)
    duration = len(y) / sr

    # Estimación de BPM / tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(np.atleast_1d(tempo)[0]), 1)
        if bpm <= 0 or np.isnan(bpm):
            bpm = 120.0
    except Exception:
        bpm = 120.0

    return {
        'app': 'Kichwa Music',
        'type': 'chord-progression',
        'song': song,
        'title': song,
        'analysis': {
            'sampleRate': sr,
            'hopSize': HOP,
            'bpm': bpm,
            'detector': 'librosa CQT chroma + plantillas (Render / Hugging Face v2)',
        },
        'metadata': {
            'durationSeconds': round(duration, 3),
            'estimatedKey': key,
        },
        'chords': segs,
    }


app = FastAPI(title='Kichwa Chord Detector', version='2.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'kichwa-chord-detector'}


@app.post('/analyze')
async def analyze(file: UploadFile = File(...), name: str = Form(default='')):
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail='Archivo vacío.')
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'Archivo demasiado grande ({len(raw) / 1e6:.1f} MB). '
                   f'Límite: {MAX_BYTES // (1024 * 1024)} MB.',
        )

    song = (name or file.filename or 'cancion').strip() or 'cancion'
    try:
        y, sr = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f'No se pudo decodificar el audio: {e}')

    if y.size == 0:
        raise HTTPException(status_code=400, detail='El audio está vacío o corrupto.')
    if len(y) / sr > MAX_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f'Canción demasiado larga ({len(y) / sr / 60:.1f} min). Máximo {MAX_SECONDS // 60} min.',
        )

    try:
        result = run_analysis(y, sr, song)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'Error durante el análisis: {e}')
    return JSONResponse(content=result)


def _extract_youtube_audio(video_id: str, temp_dir: str):
    """Descarga el audio de YouTube con múltiples capas de tolerancia a fallos:

    1. yt-dlp con cookies (si se configuró YOUTUBE_COOKIES en Render o existe cookies.txt)
    2. yt-dlp con clientes móviles (Android, iOS, Android VR, Web Embedded, MWeb)
    3. Cobalt API v10/v11
    4. Piped API
    5. Invidious API
    """
    import os
    import requests
    import yt_dlp

    title = f'Video de YouTube ({video_id})'
    uploader = 'YouTube'
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    errors = []

    # 0. Revisar si hay cookies configuradas en Render (Environment Variable: YOUTUBE_COOKIES o archivo cookies.txt)
    cookie_file = None
    cookies_env = os.environ.get('YOUTUBE_COOKIES') or os.environ.get('YTDL_COOKIES')
    if cookies_env:
        cookie_file = os.path.join(temp_dir, 'cookies.txt')
        content = cookies_env.replace('\\n', '\n').replace('\\r', '').replace('\\t', '\t').strip()
        with open(cookie_file, 'w', encoding='utf-8') as cf:
            cf.write(content)
    elif os.path.exists('cookies.txt'):
        cookie_file = os.path.abspath('cookies.txt')
    elif os.path.exists('/app/cookies.txt'):
        cookie_file = '/app/cookies.txt'

    # 1. Intentar yt-dlp con varias configuraciones
    AUDIO_FORMAT_SELECTOR = 'ba/ba*/bestaudio/bestaudio*/140/251/249/250/139/best/b'
    ydl_configs = []
    if cookie_file:
        ydl_configs.append({
            'name': 'yt-dlp (cookies web)',
            'opts': {
                'cookiefile': cookie_file,
                'format': AUDIO_FORMAT_SELECTOR,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web', 'web_creator', 'mweb'],
                    }
                },
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
                },
            },
        })
        ydl_configs.append({
            'name': 'yt-dlp (cookies mweb)',
            'opts': {
                'cookiefile': cookie_file,
                'format': AUDIO_FORMAT_SELECTOR,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['mweb', 'web_safari'],
                    }
                },
            },
        })
        ydl_configs.append({
            'name': 'yt-dlp (cookies default)',
            'opts': {
                'cookiefile': cookie_file,
                'format': AUDIO_FORMAT_SELECTOR,
            },
        })

    ydl_configs.extend([
        {
            'name': 'yt-dlp (android)',
            'opts': {
                'format': AUDIO_FORMAT_SELECTOR,
                'extractor_args': {'youtube': {'player_client': ['android']}},
            },
        },
        {
            'name': 'yt-dlp (ios)',
            'opts': {
                'format': AUDIO_FORMAT_SELECTOR,
                'extractor_args': {'youtube': {'player_client': ['ios']}},
            },
        },
        {
            'name': 'yt-dlp (android_vr,web_embedded)',
            'opts': {
                'format': AUDIO_FORMAT_SELECTOR,
                'extractor_args': {'youtube': {'player_client': ['android_vr', 'web_embedded']}},
            },
        },
        {
            'name': 'yt-dlp (mweb)',
            'opts': {
                'format': AUDIO_FORMAT_SELECTOR,
                'extractor_args': {'youtube': {'player_client': ['mweb']}},
            },
        },
        {
            'name': 'yt-dlp (default)',
            'opts': {
                'format': AUDIO_FORMAT_SELECTOR,
            },
        },
    ])

    for cfg in ydl_configs:
        try:
            out_template = os.path.join(temp_dir, f'{video_id}_%(id)s.%(ext)s')
            opts = {
                'outtmpl': out_template,
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }],
            }
            opts.update(cfg['opts'])
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                title = info.get('title') or title
                uploader = info.get('uploader') or info.get('channel') or uploader

            audio_files = [
                os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                if f.endswith(('.mp3', '.m4a', '.webm', '.opus', '.wav', '.aac'))
            ]
            if audio_files:
                for af in audio_files:
                    if os.path.exists(af) and os.path.getsize(af) > 1000:
                        return af, title, uploader
        except Exception as e:
            errors.append(f"{cfg['name']}: {str(e)[:120]}")

    # 2. Intentar instancias de Cobalt API (v10 / v11)
    cobalt_instances = [
        'https://cobalt-api.kwiatekm.tokyo',
        'https://api.cobalt.tools',
        'https://co.wuk.sh',
        'https://cobalt.api.scast.me',
    ]
    for cob in cobalt_instances:
        try:
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }
            payload = {
                'url': video_url,
                'downloadMode': 'audio',
                'audioFormat': 'mp3',
            }
            r = requests.post(f'{cob}/', json=payload, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                stream_url = data.get('url')
                if stream_url:
                    audio_res = requests.get(stream_url, timeout=20, stream=True)
                    if audio_res.status_code == 200:
                        target_file = os.path.join(temp_dir, f'{video_id}.mp3')
                        with open(target_file, 'wb') as f:
                            for chunk in audio_res.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                        if os.path.exists(target_file) and os.path.getsize(target_file) > 1000:
                            return target_file, title, uploader
        except Exception as e:
            errors.append(f'Cobalt: {str(e)[:80]}')

    # 3. Intentar Piped API
    piped_instances = [
        'https://pipedapi.kavin.rocks',
        'https://api.piped.privacydev.net',
        'https://piped-api.garudalinux.org',
        'https://api-piped.mha.fi',
        'https://pipedapi.leptons.xyz',
        'https://pipedapi.tokhmi.xyz',
    ]
    for base in piped_instances:
        try:
            r = requests.get(f'{base}/streams/{video_id}', timeout=4, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                data = r.json()
                title = data.get('title') or title
                uploader = data.get('uploader') or uploader
                audio_streams = data.get('audioStreams', [])
                if audio_streams:
                    audio_streams.sort(key=lambda x: int(x.get('bitrate', 0)), reverse=True)
                    stream_url = audio_streams[0].get('url')
                    if stream_url:
                        audio_res = requests.get(stream_url, timeout=20, stream=True)
                        if audio_res.status_code == 200:
                            target_file = os.path.join(temp_dir, f'{video_id}.m4a')
                            with open(target_file, 'wb') as f:
                                for chunk in audio_res.iter_content(chunk_size=65536):
                                    if chunk:
                                        f.write(chunk)
                            if os.path.exists(target_file) and os.path.getsize(target_file) > 1000:
                                return target_file, title, uploader
        except Exception as e:
            errors.append(f'Piped: {str(e)[:80]}')

    # 4. Intentar Invidious API
    invidious_instances = [
        'https://inv.nadeko.net',
        'https://invidious.nerdvpn.de',
        'https://invidious.projectsegfau.lt',
        'https://yt.artemislena.eu',
        'https://invidious.flokinet.to',
        'https://invidious.private.coffee',
        'https://invidious.asir.dev',
    ]
    for inst in invidious_instances:
        try:
            r = requests.get(f'{inst}/api/v1/videos/{video_id}', timeout=4, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                data = r.json()
                title = data.get('title') or title
                uploader = data.get('author') or uploader
                adaptive = data.get('adaptiveFormats', [])
                audio_streams = [f for f in adaptive if (f.get('type', '').startswith('audio/') or f.get('container') in ['m4a', 'webm'])]
                if audio_streams:
                    audio_streams.sort(key=lambda x: int(x.get('bitrate', 0)), reverse=True)
                    stream_url = audio_streams[0].get('url')
                    if stream_url:
                        audio_res = requests.get(stream_url, timeout=20, stream=True)
                        if audio_res.status_code == 200:
                            target_file = os.path.join(temp_dir, f'{video_id}.m4a')
                            with open(target_file, 'wb') as f:
                                for chunk in audio_res.iter_content(chunk_size=65536):
                                    if chunk:
                                        f.write(chunk)
                            if os.path.exists(target_file) and os.path.getsize(target_file) > 1000:
                                return target_file, title, uploader
        except Exception as e:
            errors.append(f'Invidious: {str(e)[:80]}')

    error_summary = ' \n• '.join(errors) if errors else 'No se pudo conectar a ningún servicio de extracción.'
    raise Exception(f'No se pudo extraer el audio de YouTube. Detalle de los intentos:\n• {error_summary}')


@app.get('/analyze-youtube')
@app.post('/analyze-youtube')
async def analyze_youtube(id: str = '', url: str = ''):
    import os
    import re
    import tempfile

    video_input = (id or url).strip()
    if not video_input:
        raise HTTPException(status_code=400, detail='Debes proporcionar el parámetro "id" o "url" de YouTube.')

    # Extraer ID de 11 caracteres
    yt_regex = re.compile(
        r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?|shorts)\/|.*[?&]v=)|youtu\.be\/|youtube\.com\/embed\/)?([a-zA-Z0-9_-]{11})',
        re.IGNORECASE,
    )
    match = yt_regex.search(video_input)
    if not match:
        raise HTTPException(status_code=400, detail='ID o URL de YouTube no válido.')
    video_id = match.group(1)

    temp_dir = tempfile.mkdtemp()

    try:
        audio_file, title, uploader = _extract_youtube_audio(video_id, temp_dir)

        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        if y.size == 0:
            raise HTTPException(status_code=400, detail='El audio del video está vacío o corrupto.')

        result = run_analysis(y, sr, title)
        result['title'] = title
        result['artist'] = uploader
        result['youtubeVideoId'] = video_id

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error procesando video de YouTube: {e}')
    finally:
        try:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
        except Exception:
            pass


@app.get('/debug-youtube')
async def debug_youtube(id: str = 'dPDo8TM7Zro'):
    import os
    import sys
    cookies_env = os.environ.get('YOUTUBE_COOKIES') or os.environ.get('YTDL_COOKIES')
    return {
        'pythonVersion': sys.version,
        'hasCookiesEnv': bool(cookies_env),
        'cookiesEnvLength': len(cookies_env) if cookies_env else 0,
        'localCookiesTxtExists': os.path.exists('cookies.txt'),
        'appCookiesTxtExists': os.path.exists('/app/cookies.txt'),
        'testVideoId': id,
    }


TEST_PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Kichwa Chord Detector</title>
<style>body{font-family:sans-serif;max-width:680px;margin:40px auto;padding:0 16px;background:#111;color:#eee}
button{background:#FF6B35;color:#fff;border:0;padding:10px 18px;border-radius:8px;cursor:pointer;font-size:15px}
#out{white-space:pre-wrap;font-size:12px;background:#1c1c1c;padding:12px;border-radius:8px;margin-top:16px;max-height:420px;overflow:auto}</style>
</head><body>
<h2>🎸 Kichwa Chord Detector</h2>
<p>Sube una canción (MP3/M4A/OGG/WAV) y obtén los acordes con marcas de tiempo en JSON.</p>
<input id="f" type="file" accept="audio/*"><br><br>
<button onclick="go()">Analizar</button>
<div id="out">Esperando archivo…</div>
<script>
async function go(){
  const f=document.getElementById('f').files[0];
  if(!f){alert('Elige un archivo');return;}
  const out=document.getElementById('out');
  out.textContent='Subiendo '+f.name+' …';
  const fd=new FormData();fd.append('file',f);fd.append('name',f.name);
  try{
    const r=await fetch('/analyze',{method:'POST',body:fd});
    const j=await r.json();
    out.textContent=JSON.stringify(j,null,2);
  }catch(e){out.textContent='Error: '+e;}
}
</script></body></html>"""


@app.get('/', response_class=HTMLResponse)
def index():
    return TEST_PAGE
