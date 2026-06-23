"""
Chatterbox TTS HTTP service wrapper.

Exposes the API contract expected by adapters/synthesize_voice/tts_adapter.py:
  GET  /health      → {"status": "ok"}
  POST /synthesize  → multipart/form-data: text, reference_audio (file),
                       language, exaggeration → raw WAV bytes

Run:
  uvicorn services.chatterbox_server:app --host 0.0.0.0 --port 8020
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_model = None
_sample_rate: int = 24000


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _sample_rate
    import torch
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading ChatterboxTTS on %s ...", device)
    _model = ChatterboxTTS.from_pretrained(device=device)
    _sample_rate = _model.sr
    logger.info("ChatterboxTTS ready — sample rate %d Hz", _sample_rate)
    yield
    _model = None


app = FastAPI(title="Chatterbox TTS", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    language: str = Form("en"),
    exaggeration: float = Form(0.5),
) -> Response:
    if _model is None:
        raise HTTPException(503, detail="Model not loaded yet")

    # Write reference audio to a temp file (Chatterbox needs a path, not bytes)
    suffix = os.path.splitext(reference_audio.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await reference_audio.read())
        ref_path = tmp.name

    try:
        wav = _model.generate(
            text,
            audio_prompt_path=ref_path,
            exaggeration=exaggeration,
        )
    except TypeError:
        # Older Chatterbox builds may not accept exaggeration
        wav = _model.generate(text, audio_prompt_path=ref_path)
    finally:
        os.unlink(ref_path)

    import torchaudio

    buf = io.BytesIO()
    torchaudio.save(buf, wav, _sample_rate, format="wav")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
