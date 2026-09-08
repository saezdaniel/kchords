# 🚀 Cómo montar el servidor en Hugging Face Spaces (GRATIS)

Sigue estos pasos (5 minutos, no requiere tarjeta ni instalar nada):

## 1. Crear cuenta
Entra en https://huggingface.co y regístrate gratis (solo email).

## 2. Crear el Space
1. Clic en tu avatar → **New Space** (https://huggingface.co/new-space)
2. **Space name**: `kichwa-chord-detector` (o el que quieras)
3. **License**: MIT (o la que prefieras)
4. **Select the Space SDK**: elige **Docker** → Blank
5. Visibilidad: **Public** (obligatorio para llamarlo gratis desde la app)
6. Clic en **Create Space**

## 3. Subir los archivos de esta carpeta
Dentro del Space: pestaña **Files** → **Add file** → **Upload files**.
Sube estos **4 archivos**:

- `README.md` (importante: contiene la configuración del Space)
- `Dockerfile`
- `requirements.txt`
- `app.py`

El Space se compilará solo (la primera vez tarda ~3-6 min porque instala
librosa/ffmpeg). Cuando veas **Running** (punto verde), está listo.

## 4. Probar en el navegador
Abre la web del Space: verás una página de prueba. Sube una canción y
comprueba que devuelve el JSON con los acordes.

## 5. Conectar la app
En la app: **Media → Herramientas → Detector de Acordes** → activa el modo
**Nube (HF)** → toca el icono de servidor ⚙ → pega la URL.

Acepta cualquiera de estas dos formas:
- `https://huggingface.co/spaces/TU-USUARIO/kichwa-chord-detector`
- `https://TU-USUARIO-kichwa-chord-detector.hf.space`

Toca **Probar** (debe decir "✅ Conexión correcta") y **Guardar**.

## 6. Analizar
Elige una canción y pulsa **Analizar**. La canción se sube al Space, se
analiza con librosa y vuelve el JSON de acordes. La reproducción
sincronizada funciona igual que con el motor local.

---

## ⚠️ Notas importantes

- **Arranque en frío**: si el Space lleva 48 h sin usarse, se "duerme"; la
  primera llamada tardará 1-3 min mientras despierta (luego va rápido).
- **Límites**: 40 MB por archivo, 8 minutos por canción (protege el tier gratis).
- **Privacidad**: la canción se sube a TU Space en Hugging Face. Si la app se
  publica en Google Play, menciona esto en la política de privacidad.
- **Costo**: 0 €. Si algún día necesitas más potencia, HF ofrece hardware de
  pago o puedes mover el mismo `app.py` a cualquier VPS.
