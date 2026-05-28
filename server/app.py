"""Faster Whisper FastAPI service.

Designed to run inside the ``faster-whisper-gpu`` Docker image on the
GPU server (e.g. ws1-rtx5090, 10.80.39.41). The Whisper Large v3 model
is loaded once at startup and kept resident in GPU memory so subsequent
requests are fast.

The JSON response schema is intentionally identical to what the local
``sentence_recognition.SentenceRecognition.recognize`` method writes to
disk, so the rest of the pipeline (``stress_highlight.py``) can consume
remote and local outputs interchangeably.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get(
    "WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8"
)
API_KEY = os.environ.get("WHISPER_API_KEY", "").strip()
MAX_UPLOAD_MB = int(os.environ.get("WHISPER_MAX_UPLOAD_MB", "1024"))
DEFAULT_LANGUAGE = os.environ.get("WHISPER_DEFAULT_LANGUAGE", "en")

SENSITIVITY_LEVELS = ("off", "sensitive", "ultra")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("whisper-api")


# ---------------------------------------------------------------------------
# Transcribe kwargs (mirrors sentence_recognition._build_transcribe_kwargs)
# ---------------------------------------------------------------------------

def build_transcribe_kwargs(sensitivity: str, language: str) -> dict:
    base = dict(
        language=language,
        word_timestamps=True,
        vad_filter=False,
    )
    if sensitivity == "sensitive":
        base.update(
            no_speech_threshold=0.2,
            log_prob_threshold=-1.5,
            compression_ratio_threshold=2.6,
            condition_on_previous_text=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
    elif sensitivity == "ultra":
        base.update(
            no_speech_threshold=0.05,
            log_prob_threshold=-2.5,
            compression_ratio_threshold=3.0,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            beam_size=10,
            chunk_length=15,
            initial_prompt=(
                "The following is quiet, whispered, or low-energy "
                "English speech."
            ),
        )
    return base


def normalise_sensitivity(value: Optional[str]) -> str:
    if not value:
        return "off"
    v = value.strip().lower()
    return v if v in SENSITIVITY_LEVELS else "off"


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

_state: dict = {"model": None}


def _load_model():
    from faster_whisper import WhisperModel

    log.info(
        "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
        MODEL_SIZE, DEVICE, COMPUTE_TYPE,
    )
    started = time.time()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("Model ready in %.1fs", time.time() - started)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = _load_model()
    yield
    _state["model"] = None


app = FastAPI(
    title="Faster Whisper STT API",
    version="1.0.0",
    description=(
        "Speech-to-text via faster-whisper. Returns word-level timestamps "
        "in the same JSON shape as the local sentence_recognition module."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str]) -> None:
    """Reject the request if a shared-secret API key is configured but
    missing/incorrect on the request."""
    if not API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    expected = f"Bearer {API_KEY}"
    if authorization.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _safe_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".bin"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if len(ext) > 8 or not ext.startswith("."):
        return ".bin"
    return ext


def _segments_to_payload(segments_iter, info) -> dict:
    json_segments = []
    full_text_parts = []
    for segment in segments_iter:
        words = []
        if segment.words:
            for w in segment.words:
                words.append(
                    {
                        "word": w.word,
                        "start": float(w.start) if w.start is not None else 0.0,
                        "end": float(w.end) if w.end is not None else 0.0,
                        "probability": float(getattr(w, "probability", 0.0) or 0.0),
                    }
                )
        json_segments.append(
            {
                "id": len(json_segments),
                "start": float(segment.start) if segment.start is not None else 0.0,
                "end": float(segment.end) if segment.end is not None else 0.0,
                "text": segment.text,
                "words": words,
            }
        )
        full_text_parts.append(segment.text)

    return {
        "text": "".join(full_text_parts),
        "language": info.language,
        "segments": json_segments,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "faster-whisper-stt",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    ready = _state.get("model") is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "loading",
            "model": MODEL_SIZE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        },
    )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    sensitivity: str = Form("off"),
    language: str = Form(DEFAULT_LANGUAGE),
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)

    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    sensitivity = normalise_sensitivity(sensitivity)
    suffix = _safe_suffix(file.filename)
    request_id = uuid.uuid4().hex[:8]

    tmp_dir = tempfile.mkdtemp(prefix=f"whisper-{request_id}-")
    tmp_path = os.path.join(tmp_dir, f"input{suffix}")
    bytes_written = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024

    try:
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {MAX_UPLOAD_MB} MB limit",
                    )
                out.write(chunk)

        log.info(
            "[%s] transcribe start file=%s size=%.2fMB sensitivity=%s lang=%s",
            request_id, file.filename, bytes_written / (1024 * 1024),
            sensitivity, language,
        )

        kwargs = build_transcribe_kwargs(sensitivity, language)
        started = time.time()
        try:
            segments_iter, info = model.transcribe(tmp_path, **kwargs)
            payload = _segments_to_payload(segments_iter, info)
        except Exception as exc:
            log.exception("[%s] transcription failed", request_id)
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {exc}",
            ) from exc

        elapsed = time.time() - started
        log.info(
            "[%s] transcribe done segments=%d language=%s elapsed=%.2fs",
            request_id, len(payload["segments"]), payload["language"], elapsed,
        )
        return payload
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
