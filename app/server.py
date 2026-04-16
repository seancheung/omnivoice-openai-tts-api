from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .audio import CONTENT_TYPES, encode
from .config import get_settings
from .engine import TTSEngine
from .schemas import (
    DesignRequest,
    HealthResponse,
    SpeechRequest,
    VoiceInfo,
    VoiceList,
)
from .voices import VoiceCatalog

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app.state.settings = settings
    app.state.catalog = VoiceCatalog(settings.voices_path)
    app.state.engine = None
    try:
        app.state.engine = TTSEngine(settings)
    except Exception:
        log.exception("failed to load OmniVoice model")
        raise
    yield


app = FastAPI(
    title="OmniVoice OpenAI-TTS API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    engine: TTSEngine | None = request.app.state.engine
    if engine is None:
        return HealthResponse(status="loading", model=settings.omnivoice_model)
    return HealthResponse(
        status="ok",
        model=settings.omnivoice_model,
        device=engine.device,
        dtype=engine.dtype_str,
        sample_rate=engine.sample_rate,
    )


@app.get("/v1/audio/voices", response_model=VoiceList)
async def list_voices(request: Request) -> VoiceList:
    catalog: VoiceCatalog = request.app.state.catalog
    base = str(request.base_url).rstrip("/")
    voices = catalog.scan()
    data = [
        VoiceInfo(
            id=v.id,
            preview_url=f"{base}/v1/audio/voices/preview?id={quote(v.id, safe='')}",
            prompt_text=v.prompt_text,
        )
        for v in voices.values()
    ]
    return VoiceList(data=data)


@app.get("/v1/audio/voices/preview")
async def preview_voice(id: str, request: Request):
    catalog: VoiceCatalog = request.app.state.catalog
    voice = catalog.get(id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{id}' not found")
    return FileResponse(
        path=str(voice.wav_path), media_type="audio/wav", filename=f"{id}.wav"
    )


def _validate_text(raw: str, max_chars: int) -> str:
    text = (raw or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="input is empty")
    if len(text) > max_chars:
        raise HTTPException(
            status_code=413, detail=f"input exceeds {max_chars} chars",
        )
    return text


def _validate_format(fmt: str) -> None:
    if fmt not in CONTENT_TYPES:
        raise HTTPException(
            status_code=422, detail=f"unsupported response_format: {fmt}",
        )


@app.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine
    catalog: VoiceCatalog = request.app.state.catalog

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    voice = catalog.get(body.voice)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{body.voice}' not found")

    try:
        samples = await engine.synthesize_clone(
            text,
            ref_audio=str(voice.wav_path),
            ref_text=voice.prompt_text,
            ref_mtime=voice.mtime,
            language=body.language,
            duration=body.duration,
            speed=body.speed,
            num_step=body.num_step,
            guidance_scale=body.guidance_scale,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    try:
        audio_bytes, content_type = encode(samples, engine.sample_rate, body.response_format)
    except Exception as e:
        log.exception("encoding failed")
        raise HTTPException(status_code=500, detail=f"encoding failed: {e}") from e

    return Response(content=audio_bytes, media_type=content_type)


@app.post("/v1/audio/design")
async def create_design(body: DesignRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    instruct = (body.instruct or "").strip()
    if not instruct:
        raise HTTPException(status_code=422, detail="instruct is empty")

    try:
        samples = await engine.synthesize_design(
            text,
            instruct=instruct,
            language=body.language,
            duration=body.duration,
            speed=body.speed,
            num_step=body.num_step,
            guidance_scale=body.guidance_scale,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    try:
        audio_bytes, content_type = encode(samples, engine.sample_rate, body.response_format)
    except Exception as e:
        log.exception("encoding failed")
        raise HTTPException(status_code=500, detail=f"encoding failed: {e}") from e

    return Response(content=audio_bytes, media_type=content_type)
